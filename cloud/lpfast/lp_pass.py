"""Real-time LivePortrait face pass on top of JoyAI output (TensorRT 11 on the laptop RTX 5080).

Per output frame O (JoyAI, e.g. G-Star) and the exact input frame I it was made from (webcam frame, matched via
source_seq), re-render O's face so that its mouth and eyes follow I. Head pose, scale, position and identity stay
JoyAI's; only the lip / eye expression keypoints change.

Split for the live pipeline:
  - `Driver.analyze(I)`: driver landmarks + LivePortrait expression. Runs when the frame is captured/sent (its own
    engines and CUDA stream), so it is off the critical path: ~3-4 ms, overlapped with the previous warp.
  - `LPPass.render(O, drv)`: on the critical path, when the JoyAI frame arrives: track the generated face (203-pt
    landmarks), 256 px crop, motion + appearance, expression transfer, stitching, warp + SPADE decoder (FP16, 512 px),
    paste into the face ROI only (GPU). All keypoint math stays on the GPU; one sync for the landmarks, one for the
    ROI download.
Expression transfer, mode "rel" (default, best for cartoon refs):
    exp[k] = base_out[k] + gain_r * (exp_in[k] - neutral_in[k])   for k in region r (lip / eye / brow / rest)
  base_out = slow EMA of JoyAI's own expression (removes its chunk-periodic mouth wobble), neutral_in = slow EMA of
  the driver's (their resting face). Mode "abs": exp[k] = exp_out[k] + gain_r * (exp_in[k] - exp_out[k]) (gain 1 =
the human expression as is). Gains per region, live-adjustable: LPPass.set_gain(lip=1, eye=1, brow=1, rest=0).
"""

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import flp_compat
from src.utils.crop import _estimate_similar_transform_from_pts
from trt_runtime import Engine

ENG = flp_compat.ENG
# LivePortrait implicit-keypoint groups (official LivePortrait index sets; brows/rest from its "exp" region):
# (keypoint, axes) per region. Each region has its own gain, adjustable live (LPPass.set_gain).
REGIONS = {
    "lip": [(k, (0, 1, 2)) for k in (6, 12, 14, 17, 19, 20)],
    "eye": [(k, (0, 1, 2)) for k in (11, 13, 15, 16, 18)],
    "brow": [(1, (0, 1, 2)), (2, (0, 1, 2))],
    "brow2": [(9, (1, 2))],  # part of LivePortrait's full-expression set near the brows / upper lids
    "rest": [(3, (1,)), (4, (1,)), (5, (2,)), (8, (2,))],  # cheeks / jaw parts of LivePortrait's full expression
}
# defaults per mode (tested 2026-11-03, expr_sync.py): rel = the G-Star preset (his cartoon face maps LivePortrait's
# brow keypoints the other way round: brow -1 -> brow corr +0.58, +1 -> -0.54; mouth 1.3 = the user's pick);
# abs = human refs, LivePortrait's full expression set (lip/eye/brow corr 0.98/0.98/0.97).
# Soft limits per region, (knee, max) on the norm of the region's expression change (gain applied): linear up to the
# knee, then tanh-compressed towards max. Calibrated 2026-11-04 by sweeping a real driver's most extreme frame on
# G-Star (assets/char_front.png): lips fine to 0.18, stretching at 0.24, torn at 0.31 (the "double mouth");
# eyes shut cleanly at 0.075, squashed lids at 0.12; brows fine to 0.11, forehead/eyes distort at 0.17.
# brow2 / rest are not calibrated (off in the gstar preset): conservative guesses.
DEFAULT_LIMITS = {"lip": (0.14, 0.21), "eye": (0.07, 0.10), "brow": (0.10, 0.14), "brow2": (0.05, 0.08),
                  "rest": (0.06, 0.10),
                  # closing past the character's resting mouth (lips pressed tighter than the speaker's neutral,
                  # times the gain): on G-Star the lips thin at 0.02 and merge / pass each other from 0.04
                  "lipclose": (0.01, 0.025)}
# unit "mouth opens" direction in the lip keypoints' expression space (6 x 3, from a real driver: mean of the most
# open quarter minus the most closed quarter); the closing limit acts on the component along it
LIP_OPEN_DIR = Path(__file__).resolve().parent / "lip_open_dir.npy"
DEFAULT_EURO = (1.0, 5.0)  # driver-expression One-Euro (min cutoff Hz, beta): -30 % output jitter at the user's gains

# Since 2026-11-03 the defaults are the user's live-tuned presets (presets.json): rel = "gstar", abs = "girl".
PRESETS_FILE = Path(__file__).resolve().parent / "presets.json"


def load_presets():
    import json
    return json.loads(PRESETS_FILE.read_text()) if PRESETS_FILE.exists() else {}


def save_preset(name, mode, gains, pose=False, note="", limits=None, gates=None, euro="unset"):
    import json
    p = load_presets()
    p[name] = {"mode": mode, "pose": bool(pose), "gains": dict(gains), "note": note}
    if limits is not None:
        p[name]["limits"] = {k: (list(v) if v else None) for k, v in limits.items()}
    if gates is not None:
        p[name]["gates"] = dict(gates)
    if euro != "unset":
        p[name]["euro"] = list(euro) if euro else None
    PRESETS_FILE.write_text(json.dumps(p, indent=2) + "\n")


_P = load_presets()
DEFAULT_GAINS = {
    "rel": dict(_P.get("gstar", {}).get("gains") or {"lip": 1.3, "eye": 1.0, "brow": -1.0, "brow2": 0.0, "rest": 0.0}),
    "abs": dict(_P.get("girl", {}).get("gains") or {"lip": 1.0, "eye": 1.0, "brow": 1.0, "brow2": 1.0, "rest": 1.0}),
}
IDX66 = None
_UNSET = object()
DET_STREAM = None
DET_LOCK = threading.Lock()  # the RetinaFace wrapper (shared by driver + source) is not thread-safe; used rarely


def _deg(logits):
    """66-bin head-pose logits -> degrees (LivePortrait: softmax expectation * 3 - 97.5)."""
    global IDX66
    if IDX66 is None:
        IDX66 = torch.arange(66, device="cuda", dtype=torch.float32)
    return (torch.softmax(logits.float(), dim=1) * IDX66).sum(1) * 3 - 97.5


def rot_matrix(pitch, yaw, roll):
    """Same as FasterLivePortrait get_rotation_matrix (degrees in, (B,3,3) out, transposed), on torch."""
    x, y, z = (v * (math.pi / 180) for v in (pitch, yaw, roll))
    o, n = torch.ones_like(x), torch.zeros_like(x)
    rx = torch.stack([o, n, n, n, x.cos(), -x.sin(), n, x.sin(), x.cos()], 1).view(-1, 3, 3)
    ry = torch.stack([y.cos(), n, y.sin(), n, o, n, -y.sin(), n, y.cos()], 1).view(-1, 3, 3)
    rz = torch.stack([z.cos(), -z.sin(), n, z.sin(), z.cos(), n, n, n, o], 1).view(-1, 3, 3)
    return (rz @ ry @ rx).transpose(1, 2)


class Face:
    """Landmark tracking + LivePortrait motion (and optionally appearance) for one video stream."""

    def __init__(self, detector, appearance=False, max_jump=0.35, retry=6):
        self.stream = torch.cuda.Stream()
        self.lm = Engine(ENG / "landmark.engine", self.stream)
        self.me = Engine(ENG / "motion_extractor.engine", self.stream)
        self.app = Engine(ENG / "appearance_feature_extractor.engine", self.stream) if appearance else None
        self.det, self.max_jump = detector, max_jump
        self.lmk = None
        self.detections = 0
        self.retry, self.lost = retry, 0  # while lost, re-detect only every `retry` frames (no per-frame stalls)
        self.stable_crop, self.crop_ref = False, None

    def _detect(self, bgr):
        self.detections += 1
        self.detected_now = True
        # the wrapper launches on the stream it was built on (DET_STREAM): run it there; fence only that stream
        with DET_LOCK, torch.cuda.stream(DET_STREAM):
            faces = self.det.predict(bgr)
            if len(faces) == 0:  # cartoon faces score lower (char_front 0.3, the no-hat G-Star close-up 0.2): step down
                th = self.det.det_thresh
                try:
                    for t in (0.3, 0.2):
                        self.det.det_thresh = t
                        faces = self.det.predict(bgr)
                        if len(faces):
                            break
                finally:
                    self.det.det_thresh = th
            DET_STREAM.synchronize()
        if len(faces) == 0:
            return None
        return self._landmarks(bgr, faces[0])

    def _landmarks(self, bgr, pts):
        M, _ = _estimate_similar_transform_from_pts(pts, dsize=224, scale=1.5, vy_ratio=-0.1)
        crop = cv2.warpAffine(bgr, M, (224, 224), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x = torch.from_numpy(crop[..., ::-1].copy()).cuda().permute(2, 0, 1)[None].float().div_(255)
        out = self.lm.run(input=x)["856"].float().cpu().numpy().reshape(-1, 2) * 224  # 203 points in the crop
        Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]]))
        return out @ Mi[:2, :2].T + Mi[:2, 2]

    def landmarks(self, bgr):
        """Track from the previous landmarks; re-detect only when lost or when the track jumps."""
        self.detected_now = False
        if self.lmk is None:
            self.lost += 1
            if self.retry > 1 and self.lost % self.retry != 1:
                return None
            lmk = self._detect(bgr)
            self.lmk = lmk if lmk is not None and np.isfinite(lmk).all() else None
            return self.lmk
        prev = self.lmk
        lmk = self._landmarks(bgr, prev)
        size = np.ptp(prev[:, 1]) + 1e-6
        if not np.isfinite(lmk).all() or np.abs(lmk.mean(0) - prev.mean(0)).max() / size > self.max_jump \
                or np.ptp(lmk[:, 1]) < 0.4 * size:
            lmk = self._detect(bgr)
            if lmk is not None and not np.isfinite(lmk).all():
                lmk = None
        self.lmk = lmk
        if lmk is not None:
            self.lost = 0
        return lmk

    def _stable_crop(self, lmk, M):
        """Jaw-independent crop: LivePortrait sizes/centres the crop on the landmark box, which includes the chin, so
        an open mouth grows the box (face looks smaller -> scale drops, the animated face shrinks). Keep the first
        frame's crop relative to the eyes instead: size ~ inter-ocular distance, centre/angle tied to the eye line."""
        el, er = lmk[[0, 12]].mean(0), lmk[[24, 36]].mean(0)
        ce, d = (el + er) / 2, er - el
        iod, ang = float(np.hypot(*d)) + 1e-6, float(np.arctan2(d[1], d[0]))
        if self.crop_ref is None:
            sc = float(np.hypot(M[0, 0], M[0, 1]))
            th = float(np.arctan2(M[0, 1], M[0, 0]))
            center = np.linalg.solve(M[:, :2], np.array([128.0, 128.0]) - M[:, 2])
            c, s_ = np.cos(-ang), np.sin(-ang)
            rel = np.array([[c, -s_], [s_, c]]) @ (center - ce) / iod
            self.crop_ref = {"rel": rel, "size": (256.0 / sc) / iod, "dth": th - ang}
            return M
        r = self.crop_ref
        c, s_ = np.cos(ang), np.sin(ang)
        center = ce + (np.array([[c, -s_], [s_, c]]) @ r["rel"]) * iod
        sc = 256.0 / (r["size"] * iod)
        th = ang + r["dth"]  # crop angle follows the eye line (checked: d(crop angle) = d(eye angle))
        ct, st = np.cos(th), np.sin(th)
        return np.array([[sc * ct, sc * st, 128 - sc * (ct * center[0] + st * center[1])],
                         [-sc * st, sc * ct, 128 - sc * (-st * center[0] + ct * center[1])]], np.float32)

    def motion(self, bgr, lmk):
        """256 px LivePortrait crop -> dict of GPU tensors (+ M_c2o of the equivalent 512 crop, for paste-back)."""
        M256, _ = _estimate_similar_transform_from_pts(lmk, dsize=256, scale=2.3, vy_ratio=-0.125)
        if self.stable_crop:
            M256 = self._stable_crop(lmk, M256)
        crop = cv2.warpAffine(bgr, M256, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x = torch.from_numpy(crop[..., ::-1].copy()).cuda().permute(2, 0, 1)[None].float().div_(255)
        o = self.me.run(img=x)  # Engine.run orders itself after the caller's stream and back
        m = {"pitch": _deg(o["pitch"]), "yaw": _deg(o["yaw"]), "roll": _deg(o["roll"]),
             "t": o["t"].float().clone(), "exp": o["exp"].float().view(1, 21, 3).clone(),
             "scale": o["scale"].float().clone(), "kp": o["kp"].float().view(1, 21, 3).clone()}
        if self.app is not None:
            m["f"] = self.app.run(img=x)["output"].clone()
        M512_o2c = np.vstack([M256 * 2.0, [0, 0, 1]])  # 512 crop = 256 crop scaled by 2
        m["M_c2o"] = np.linalg.inv(M512_o2c)
        return m


class Visibility:
    """Generic occlusion + 'turned away' signals for the driver frame (any occluder: hand, cup, mic, hair...).

    - mouth_vis: a face-parsing net (BiSeNet, CelebAMask-HQ classes, FP16 TensorRT, 0.6 ms) labels the 512 px face
      crop; the landmarks say where the mouth is even when it's hidden, so mouth_vis = fraction of the mouth's landmark
      region (dilated) labelled lips or mouth interior. Calibrated 2026-11-04 on 3 people: normal frames >= 0.66
      (pressed lips included), a covered mouth 0.0, glasses held over the mouth 0.32.
      Not done for the eyes: the parser doesn't label eyes/brows reliably at webcam scale (normal frames read 0).
    - face_score: RetinaFace's top anchor confidence (no NMS, ~2.5 ms): frontal 0.76-0.90, turned / profile
      0.04-0.5. It does NOT drop for partial occlusion (detectors are trained to see through it).
    Both are computed every `every` frames (default 2) and held in between: gating needs ~12 Hz, the GPU is busy.
    """

    def __init__(self, detector, stream, every=2):
        self.det, self.stream, self.every = detector, stream, every
        self.parse = Engine(ENG / "faceparse_fp16.engine", stream)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        self.n, self.last = 0, (1.0, 1.0)

    def reset(self):
        self.n, self.last = 0, (1.0, 1.0)

    def face_score(self, bgr):
        H, W = bgr.shape[:2]
        r = min(512 / W, 512 / H)
        im = cv2.resize(bgr, (int(W * r), int(H * r)))
        canvas = np.zeros((512, 512, 3), np.uint8)
        canvas[:im.shape[0], :im.shape[1]] = im
        x = (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) - 127.5) / 128.0
        fd = self.det.face_det
        with DET_LOCK, torch.cuda.stream(DET_STREAM):
            out = fd.predict({fd.inputs[0]["name"]: torch.from_numpy(x).cuda()}, DET_STREAM.cuda_stream)
            return max(float(out[k].max()) for k in ("448", "471", "494"))  # the three stride levels' score maps

    def mouth_vis(self, bgr, lmk):
        M, _ = _estimate_similar_transform_from_pts(lmk, dsize=512, scale=1.3, vy_ratio=-0.05)
        crop = cv2.warpAffine(bgr, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x = torch.from_numpy(crop[..., ::-1].copy()).cuda().permute(2, 0, 1)[None].float().div_(255)
        lab = self.parse.run(input=(x - self.mean) / self.std)["out"].argmax(1)[0].byte().cpu().numpy()
        m = np.zeros((512, 512), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull((lmk[48:106] @ M[:, :2].T + M[:, 2]).astype(np.int32)), 1)
        m = cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0
        return float(np.isin(lab[m], (11, 12, 13)).mean()) if m.any() else 1.0

    def __call__(self, bgr, lmk):
        """-> (mouth_vis, face_score), refreshed every `every` frames."""
        self.n += 1
        if self.n % self.every == 1 or self.every <= 1:
            self.last = (self.mouth_vis(bgr, lmk), self.face_score(bgr))
        return self.last


class Driver:
    """Webcam side: call `analyze(I)` when the frame is captured/sent; keep the result with the frame's seq."""

    def __init__(self, detector, visibility=True):
        self.face = Face(detector, retry=3)  # parallel thread: back off a little so it doesn't contend
        self.face.stable_crop = True  # the driver's mouth must not change its crop (scale / position)
        self.vis = Visibility(detector, self.face.stream) if visibility else None
        self.bad, self.since_redetect = 0, 99

    def reset(self):
        """Re-take the crop reference and the visibility baseline and re-detect the face (LPPass neutral reset /
        a new person / a stuck track)."""
        self.face.crop_ref = None
        self.face.lmk = None
        self.bad = 0
        if self.vis is not None:
            self.vis.reset()

    def analyze(self, bgr):
        with torch.cuda.stream(self.face.stream):  # own stream: overlaps with the render of the previous frame
            lmk = self.face.landmarks(bgr)
            if lmk is None:
                return None
            m = self.face.motion(bgr, lmk)
            m["R"] = rot_matrix(m["pitch"], m["yaw"], m["roll"])
            self.face.stream.synchronize()
        el, er = lmk[[0, 12]].mean(0), lmk[[24, 36]].mean(0)
        # head size from the inter-ocular distance (expression-independent; LivePortrait's own scale estimate moves
        # with the jaw and brows, which shrank the animated face when the mouth opened / brows went up)
        mv, fs = self.vis(bgr, lmk) if self.vis is not None else (1.0, 1.0)
        # self-check: the landmark tracker follows the previous frame and only re-detects on a jump, so after a hand
        # over the face / a turn it can lock onto the wrong thing and keep the gates closed forever. After 8 bad
        # frames in a row (mouth not visible, low face confidence, turned far) re-detect, at most every 15 frames.
        self.since_redetect += 1
        self.bad = self.bad + 1 if (mv < 0.45 or fs < 0.6 or abs(float(m["yaw"])) > 40) else 0
        if self.bad >= 8 and self.since_redetect >= 15:
            self.face.lmk, self.bad, self.since_redetect = None, 0, 0
            if self.vis is not None:
                self.vis.reset()  # fresh visibility values on the next frame
        return {"exp": m["exp"], "lmk": lmk, "R": m["R"], "iod": float(np.hypot(*(er - el))),
                "yaw": float(m["yaw"]), "pitch": float(m["pitch"]), "mouth_vis": mv, "face_score": fs}


class OneEuro:
    """One-Euro filter on a GPU tensor (per element): smooths hard when the signal is still (kills webcam/landmark
    jitter, which high gains amplify), little when it moves fast (keeps lip sync responsive). fps = call rate."""

    def __init__(self, min_cutoff=1.0, beta=5.0, d_cutoff=1.0, fps=25.0):
        self.min_cutoff, self.beta, self.d_cutoff, self.te = min_cutoff, beta, d_cutoff, 1.0 / fps
        self.x = self.dx = None

    def _a(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / self.te)

    def __call__(self, x):
        if self.x is None:
            self.x, self.dx = x.clone(), torch.zeros_like(x)
            return self.x
        dx = (x - self.x) / self.te
        self.dx = self.dx + self._a(self.d_cutoff) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * self.dx.abs()
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / self.te)
        self.x = self.x + a * (x - self.x)
        return self.x


class LPPass:
    def __init__(self, mode="rel", gains=None, base_ema=0.08,
                 neutral_ema=0.01, fade=0.25, drv_smooth=0.0, euro=DEFAULT_EURO, yaw_gate=(30.0, 45.0), pitch_gate=(20.0, 35.0),
                 pose=False, limits=None, drv_yaw_gate=(25.0, 40.0), drv_pitch_gate=(18.0, 30.0), visibility=True,
                 cover_gate=(0.55, 0.35), score_gate=(0.65, 0.5), cover_hold=6):
        global DET_STREAM
        DET_STREAM = DET_STREAM or torch.cuda.Stream()
        with torch.cuda.stream(DET_STREAM):  # the wrappers capture the current stream at construction
            m = flp_compat.load_models()  # only RetinaFace (+106-pt) detection is used through the wrappers
        self.detector = m["face_analysis"]
        # lost generated face: re-detect every frame (a lost frame renders nothing, so a detection fits its budget)
        self.src = Face(self.detector, appearance=True, retry=1)
        self.warp = Engine(ENG / "warping_fp16.engine", self.src.stream)
        self.stitch = Engine(ENG / "stitching.engine", self.src.stream)
        mt = cv2.imread(str(flp_compat.FLP / "assets" / "mask_template.png"), cv2.IMREAD_GRAYSCALE)
        self.mask = torch.from_numpy(mt.astype(np.float32) / 255).cuda()[None]  # 1x512x512
        self.mode = mode
        self.gains = dict(DEFAULT_GAINS[mode])
        self.set_gain(**(gains or {}))
        self.base_ema, self.neutral_ema, self.fade, self.drv_smooth = base_ema, neutral_ema, fade, drv_smooth
        self.set_euro(euro)
        self.yaw_gate, self.pitch_gate = yaw_gate, pitch_gate
        # driver gates: the speaker turned too far (yaw absolute, pitch relative to their usual pitch) or a hand over
        # the face -> the effect fades out completely; after a cover it waits `cover_hold` clear frames
        self.drv_yaw_gate, self.drv_pitch_gate, self.use_vis = drv_yaw_gate, drv_pitch_gate, visibility
        self.cover_gate, self.score_gate = cover_gate, score_gate
        self.cover_hold, self.clear_frames, self.pitch0, self.gate_info = cover_hold, 99, None, {}
        self.limits = dict(DEFAULT_LIMITS)
        self.set_limit(**(limits or {}))
        # pose=True (still-image sources): also transfer the driver's head rotation / shift / scale, relative to the
        # driver's first frame (LivePortrait image animation). Off for JoyAI output, which already moves the head.
        self.pose, self.drv0 = pose, None
        self.base_out = self.neutral_in = self.exp_in = None
        self.pose_s = None
        self.w = 0.0
        self.timing = {}

        self._warmup()

    def set_euro(self, euro):
        """Driver-expression One-Euro filter: None/off, or (min_cutoff Hz, beta). Lower min_cutoff = calmer when
        still; higher beta = less lag on fast motion."""
        self.euro = tuple(euro) if euro else None
        self._euro = OneEuro(*self.euro) if self.euro else None
        return self.euro

    def apply_preset(self, name):
        """Switch mode, gains and pose transfer to a saved preset (presets.json); safe while rendering."""
        p = load_presets().get(name)
        if p is None:
            raise ValueError(f"no preset {name!r}; have {sorted(load_presets())}")
        self.mode = p.get("mode", self.mode)
        self.pose = bool(p.get("pose", self.pose))
        self.drv0 = None
        self.gains = dict(DEFAULT_GAINS[self.mode])
        self.limits = dict(DEFAULT_LIMITS)
        self.set_limit(**{k: (tuple(v) if v else None) for k, v in p.get("limits", {}).items()})
        if p.get("gates"):
            self.set_gate(**p["gates"])
        if "euro" in p:
            self.set_euro(p["euro"])
        return self.set_gain(**p.get("gains", {}))

    def gates(self):
        return {"drv_yaw": list(self.drv_yaw_gate), "drv_pitch": list(self.drv_pitch_gate), "cover": list(self.cover_gate),
                "score": list(self.score_gate), "visibility": self.use_vis, "cover_hold": self.cover_hold,
                "src_yaw": list(self.yaw_gate), "src_pitch": list(self.pitch_gate)}

    def set_gate(self, **g):
        """drv_yaw / drv_pitch / src_yaw / src_pitch = (start fading, fully off) in degrees; cover = (full, off) of the
        mouth visibility (0..1, normal >= 0.66, covered ~0); score = (full, off) of the face confidence (frontal
        0.76-0.9, turned away < 0.5); visibility = bool (only if the driver was built with it); cover_hold = clear
        frames before the effect returns."""
        for k, v in g.items():
            if k in ("drv_yaw", "drv_pitch", "src_yaw", "src_pitch", "cover", "score"):
                lo, hi = float(v[0]), float(v[1])
                down = k in ("cover", "score")  # these fade out as the value DROPS
                if (hi >= lo) if down else (hi <= lo):
                    raise ValueError(f"{k}: need {'full > off' if down else 'start < off'}, got {lo} {hi}")
                setattr(self, {"drv_yaw": "drv_yaw_gate", "drv_pitch": "drv_pitch_gate", "src_yaw": "yaw_gate",
                               "src_pitch": "pitch_gate", "cover": "cover_gate", "score": "score_gate"}[k], (lo, hi))
            elif k in ("visibility", "hands"):
                self.use_vis = bool(v)
            elif k == "cover_hold":
                self.cover_hold = int(v)
            else:
                raise ValueError(f"unknown gate {k!r}; use drv_yaw, drv_pitch, cover, score, src_yaw, src_pitch, "
                                 "visibility, cover_hold")
        return self.gates()

    def set_gain(self, **g):
        """Per-region motion gain (lip / eye / brow / brow2 / rest; `all` scales every region). 0 = JoyAI's own
        expression there, 1 = the driver's motion as is, 0.5 = half, 2 = double, negative = mirrored (G-Star's brows). abs mode: 1 = the driver's expression, 0.5 = a
        50/50 blend with JoyAI's. Safe to call from another thread while rendering."""
        if "all" in g:
            f = g.pop("all")
            g = {k: v * f for k, v in self.gains.items()} | g
        bad = set(g) - set(REGIONS)
        if bad:
            raise ValueError(f"unknown region(s) {sorted(bad)}; use {sorted(REGIONS)} or all")
        self.gains.update({k: float(v) for k, v in g.items()})
        W = torch.zeros(1, 21, 3, device="cuda")
        for name, kps in REGIONS.items():
            for k, axes in kps:
                W[0, k, list(axes)] = self.gains[name]
        self.W, self.Wm = W, (W != 0).float()  # swapped atomically (render reads self.W once per frame)
        return dict(self.gains)

    def set_limit(self, **lim):
        """Soft limits per region: value = (knee, max) or a single max (knee = 2/3 max); None/0 = unlimited."""
        for k, v in lim.items():
            if k not in REGIONS and k != "lipclose":
                raise ValueError(f"unknown region {k!r}; use {sorted(REGIONS)} or lipclose")
            if not v:
                self.limits[k] = None
            elif isinstance(v, (tuple, list)):
                self.limits[k] = (float(v[0]), float(v[1]))
            else:
                self.limits[k] = (2.0 * float(v) / 3.0, float(v))
        masks = []
        self._close = None
        if self.limits.get("lipclose") and LIP_OPEN_DIR.exists():
            U = torch.zeros(1, 21, 3, device="cuda")
            U[0, [k for k, _ in REGIONS["lip"]]] = torch.from_numpy(np.load(LIP_OPEN_DIR)).float().cuda()
            self._close = (U, *self.limits["lipclose"])
        for name, kps in REGIONS.items():
            if self.limits.get(name):
                M = torch.zeros(1, 21, 3, device="cuda")
                for k, axes in kps:
                    M[0, k, list(axes)] = 1.0
                masks.append((M, *self.limits[name]))
        self._limit_masks = masks
        return dict(self.limits)

    def _soft_limit(self, D):
        """Compress each region's expression change D (1x21x3) past its knee (tanh), never beyond its max; the lip
        component along 'mouth closes' gets its own (tight) limit."""
        if self._close is not None:
            U, knee, mx = self._close
            a = (D * U).sum()                       # >0 opening, <0 closing past the resting mouth
            if float(a) < -knee:
                c = -a
                c_new = knee + (mx - knee) * torch.tanh((c - knee) / (mx - knee))
                D = D + (c - c_new) * U
        for M, knee, mx in self._limit_masks:
            n = (D * M).norm()
            if float(n) > knee:
                f = (knee + (mx - knee) * torch.tanh((n - knee) / (mx - knee))) / n
                D = D * (1 - M) + D * M * f
        return D

    def _driver_gate(self, drv):
        """0..1 from the speaker's face: turned too far, or covered by a hand."""
        ramp = lambda v, g: min(1.0, max(0.0, (g[1] - v) / (g[1] - g[0])))  # noqa: E731
        self.pitch0 = drv["pitch"] if self.pitch0 is None else self.pitch0 + 0.01 * (drv["pitch"] - self.pitch0)
        g_pose = min(ramp(abs(drv["yaw"]), self.drv_yaw_gate), ramp(abs(drv["pitch"] - self.pitch0),
                                                                        self.drv_pitch_gate))
        mv = drv.get("mouth_vis", 1.0) if self.use_vis else 1.0
        fs = drv.get("face_score", 1.0) if self.use_vis else 1.0
        c0, c1 = self.cover_gate                    # mouth visibility: full above c0, off below c1, + hold after
        g_cov = min(1.0, max(0.0, (mv - c1) / (c0 - c1)))
        # hold after a cover with hysteresis: a frame counts as clear above the midpoint (a value hovering near the
        # top of the ramp, e.g. in low light, must not keep resetting the hold and leave the effect off for good)
        self.clear_frames = 0 if g_cov < 0.5 else self.clear_frames + 1
        g_cover = g_cov if self.clear_frames >= self.cover_hold else 0.0
        s0, s1 = self.score_gate                   # face confidence: full above s0, off below s1 (turned away)
        g_score = min(1.0, max(0.0, (fs - s1) / (s0 - s1)))
        self.gate_info = {"drv_yaw": round(drv["yaw"], 1), "drv_pitch_rel": round(drv["pitch"] - self.pitch0, 1),
                          "mouth_vis": round(mv, 2), "face_score": round(fs, 2), "g_pose": round(g_pose, 2),
                          "g_cover": round(g_cover, 2), "g_score": round(g_score, 2)}
        return min(g_pose, g_cover, g_score)

    def _warmup(self):
        """Run every engine once (first executions cost 50-120 ms: lazy TensorRT/CUDA init)."""
        with torch.cuda.stream(DET_STREAM):
            for pred in (self.detector.face_det, self.detector.face_pose):
                pred.predict({b["name"]: torch.zeros(b["shape"], device="cuda") for b in pred.inputs},
                             torch.cuda.current_stream().cuda_stream)
            # a real face: the first detection that finds one pays lazy imports / alignment init (~100 ms)
            img = cv2.imread(str(flp_compat.FLP / "assets" / "examples" / "source" / "s0.jpg"))
            if img is not None:
                self.detector.predict(img)
        for e in (self.src.lm, self.src.me, self.src.app, self.warp, self.stitch):
            e.run(**{n: torch.zeros_like(e.buf[n]) for n in e.inputs})
        torch.cuda.synchronize()
        if img is not None:  # full render path once (first use of each CUDA kernel is slow), then reset the state
            img = cv2.resize(img, (480, 640))
            d = Driver(self.detector, visibility=False)
            for _ in range(3):
                self.render(img, d.analyze(img))
            torch.cuda.synchronize()
            self.src.lmk, self.src.lost, self.src.detections = None, 0, 0
            self.base_out = self.neutral_in = self.exp_in = self.pose_s = self.drv0 = self.pitch0 = None
            self.clear_frames, self.gate_info = 99, {}
            self.w, self.timing = 0.0, {}

    def driver(self):
        d = Driver(self.detector, visibility=self.use_vis)
        for e in (d.face.lm, d.face.me):
            e.run(**{n: torch.zeros_like(e.buf[n]) for n in e.inputs})
        torch.cuda.synchronize()
        return d

    def _t(self, key, t0, sync=True):
        if sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        self.timing[key] = self.timing.get(key, (t - t0) * 1000) * 0.9 + (t - t0) * 1000 * 0.1
        return t

    def analyze(self, out_bgr):
        """Stage 1 (critical path, can overlap with stage 2 of the previous frame): track + motion + appearance."""
        with torch.cuda.stream(self.src.stream):
            lo = self.src.landmarks(out_bgr)
            s = None if lo is None else self.src.motion(out_bgr, lo)
            self.src.stream.synchronize()
        return s

    def render(self, out_bgr, drv, profile=False, s=_UNSET):
        """Stage 2: expression transfer, stitching, warp, paste. Pass `s` from `analyze` to pipeline the stages."""
        t0 = time.perf_counter()
        if s is _UNSET:
            s = self.analyze(out_bgr)
        ok = s is not None and drv is not None
        if self.src.detected_now:  # detection (~23 ms) already used this frame's budget: pass through, render next
            info = {"ok": False, "w": self.w, "skipped": "detect"}
            return out_bgr, info
        target = 0.0
        if ok:  # pose gate on the generated face: a frontal re-render warps profile views
            yaw, pitch = abs(float(s["yaw"])), abs(float(s["pitch"]))
            self.pose_s = (yaw, pitch) if self.pose_s is None else \
                tuple(0.6 * p + 0.4 * v for p, v in zip(self.pose_s, (yaw, pitch)))  # no flicker at the threshold
            yaw, pitch = self.pose_s
            ramp = lambda v, g: min(1.0, max(0.0, (g[1] - v) / (g[1] - g[0])))  # noqa: E731
            target = min(ramp(yaw, self.yaw_gate), ramp(pitch, self.pitch_gate), self._driver_gate(drv))
        fade_out = 2 * self.fade if self.gate_info.get("g_cover", 1.0) < 1 else self.fade  # covers: drop fast
        self.w = min(target, self.w + self.fade) if target > self.w else max(target, self.w - fade_out)
        ok = ok and self.w > 0
        info = {"ok": ok, "w": self.w, **self.gate_info}
        if not ok:  # no face on one side: pass JoyAI through
            return out_bgr, info
        t0 = self._t("src", t0, profile)
        # slow anchors (GPU)
        e_in = drv["exp"]
        if self._euro is not None:
            e_in = self._euro(e_in)
        elif self.drv_smooth > 0 and self.exp_in is not None:
            e_in = self.exp_in + (1 - self.drv_smooth) * (e_in - self.exp_in)
        self.exp_in = e_in
        self.base_out = s["exp"].clone() if self.base_out is None else self.base_out.lerp_(s["exp"], self.base_ema)
        self.neutral_in = e_in.clone() if self.neutral_in is None else self.neutral_in.lerp_(e_in, self.neutral_ema)
        W, Wm = self.W, self.Wm
        if self.mode == "abs":  # blend towards the driver's expression
            exp = s["exp"] + self._soft_limit(W * (e_in - s["exp"]))
        else:  # character's resting expression + scaled driver motion; gain-0 regions keep JoyAI's own expression
            exp = s["exp"] * (1 - Wm) + Wm * (self.base_out + self._soft_limit(W * (e_in - self.neutral_in)))
        R = rot_matrix(s["pitch"], s["yaw"], s["roll"])
        sc, t, kp = s["scale"][..., None], s["t"], s["kp"]
        x_s = (kp @ R + s["exp"]) * sc
        x_s[..., :2] += t[:, None, :2]
        if self.pose:
            if self.drv0 is None:
                self.drv0 = {"R": drv["R"].clone(), "iod": drv["iod"]}
            d0 = self.drv0
            R_d = drv["R"] @ d0["R"].transpose(1, 2) @ R  # (big turns are gated out above)
            sc_d = sc * min(1.5, max(0.67, drv["iod"] / d0["iod"]))
            t_d = t  # head translation isn't transferred (stitching cancels it; the character stays in place)
            x_d = (kp @ R_d + exp) * sc_d
            x_d[..., :2] += t_d[:, None, :2]
        else:
            x_d = (kp @ R + exp) * sc
            x_d[..., :2] += t[:, None, :2]
        d = self.stitch.run(input=torch.cat([x_s.view(1, -1), x_d.view(1, -1)], 1))["output"].float()
        x_d = x_d + d[:, :63].view(1, 21, 3)
        x_d[..., :2] += d[:, 63:65].view(1, 1, 2)
        x_d = x_s + (x_d - x_s) * self.w
        crop = self.warp.run(feature_3d=s["f"], kp_driving=x_d, kp_source=x_s)["out"]  # 1x3x512x512 RGB 0..1
        t0 = self._t("warp", t0, profile)
        res = self._paste(out_bgr, crop, s["M_c2o"])
        self._t("paste", t0, False)
        return res, info

    def _paste(self, frame, crop, M_c2o):
        """Blend the 512 render into the frame inside the face ROI only (GPU grid_sample, one download)."""
        H, W = frame.shape[:2]
        corners = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]], np.float64) @ M_c2o[:2].T
        x0, y0 = np.floor(corners.min(0)).astype(int)
        x1, y1 = np.ceil(corners.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame
        Mo2c = torch.from_numpy(np.linalg.inv(M_c2o)[:2].astype(np.float32)).cuda()
        ys, xs = torch.meshgrid(torch.arange(y0, y1, device="cuda", dtype=torch.float32),
                                torch.arange(x0, x1, device="cuda", dtype=torch.float32), indexing="ij")
        u = Mo2c[0, 0] * xs + Mo2c[0, 1] * ys + Mo2c[0, 2]
        v = Mo2c[1, 0] * xs + Mo2c[1, 1] * ys + Mo2c[1, 2]
        grid = torch.stack([(2 * u + 1) / 512 - 1, (2 * v + 1) / 512 - 1], -1)[None]
        src = torch.cat([crop.float()[0].flip(0), self.mask], 0)[None]  # RGB -> BGR, + mask
        smp = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        roi = torch.from_numpy(np.ascontiguousarray(frame[y0:y1, x0:x1])).cuda().permute(2, 0, 1).float()
        a = smp[3:4]
        blend = (a * smp[:3] * 255 + (1 - a) * roi).clamp_(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out = frame.copy()
        out[y0:y1, x0:x1] = blend
        return out
