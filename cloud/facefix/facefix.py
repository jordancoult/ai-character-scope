"""Face post-process for the JoyAI character swap: lip/eye retargeting from the real (input) face onto the
generated (output) face with LivePortrait, plus optional face restoration. Offline prototype: driving video +
generated video -> refined video, side-by-side review video and timing numbers.

  .venv/bin/python facefix.py --input in_human.mp4 --output human.mp4 --out refined.mp4 \
      --sbs sbs.mp4 --stats stats.json [--restore gfpgan] [--model animal] [--box in]

Per output frame: find the input frame it was made from (DTW over motion features; live you would use the
frame's seq), MediaPipe landmarks on the input (and on the output when a face is found there) -> LivePortrait
crop -> motion extractor on both crops -> replace the output's lip (and optionally eye) expression keypoints
with the input's -> stitching -> warp + decode the output face -> colour-match -> feathered paste-back.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions, vision

from track import paste_mask_from_seg

HERE = Path(__file__).resolve().parent
LP_DIR = HERE / "third_party" / "LivePortrait"
WEIGHTS = HERE / "weights"
FACE_TASK = HERE.parent / "client" / "models" / "face_landmarker.task"
LIP_IDX = [6, 12, 14, 17, 19, 20]      # LivePortrait implicit keypoints that move the lips
EYE_IDX = [11, 13, 15, 16, 18]          # ... and the eyes


# ---------------------------------------------------------------- video io
def read_video(path, max_frames=0):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(f)
    return frames


class Writer:
    def __init__(self, path, w, h, fps=25):
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset",
                                   "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def write(self, f):
        self.p.stdin.write(np.ascontiguousarray(f).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


# ---------------------------------------------------------------- alignment (offline only)
def motion_feats(frames, g=8, s=64):
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
             for f in frames]
    feats = [np.zeros(g * g, np.float32)]
    for a, b in zip(small, small[1:]):
        feats.append(np.abs(b - a).reshape(g, s // g, g, s // g).mean((1, 3)).ravel())
    f = np.array(feats)
    f -= f.mean(1, keepdims=True)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-6)


def align_dtw(in_frames, out_frames, max_step=60, max_start=90, jump_cost=0.01):
    """For each output frame the index of the input frame it was made from: monotonic DTW over grid
    motion-energy features (identity-independent). Output steps by 1, input by 1..max_step."""
    fi, fo = motion_feats(in_frames), motion_feats(out_frames)
    cost = 1.0 - fo @ fi.T                                    # (n_out, n_in)
    n_o, n_i = cost.shape
    inf = 1e9
    D = np.full((n_o, n_i), inf)
    B = np.zeros((n_o, n_i), np.int32)
    D[0, :max_start] = cost[0, :max_start]
    for j in range(1, n_o):
        best, arg = np.full(n_i, inf), np.zeros(n_i, np.int32)
        for s in range(1, max_step + 1):
            cand = np.full(n_i, inf)
            cand[s:] = D[j - 1, :-s] + jump_cost * max(0, s - 2)
            m = cand < best
            best[m], arg[m] = cand[m], s
        D[j], B[j] = best + cost[j], arg
    i = int(np.argmin(D[-1]))
    idx = [i]
    for j in range(n_o - 1, 0, -1):
        i -= B[j, i]
        idx.append(i)
    return np.array(idx[::-1])


def align_linear(n_in, n_out):
    return np.round(np.arange(n_out) * (n_in - 1) / max(1, n_out - 1)).astype(int)


# ---------------------------------------------------------------- landmarks + crop
HAND_TASK = WEIGHTS / "hand_landmarker.task"


def euler(m):
    """yaw, pitch, roll in degrees from a MediaPipe facial transformation matrix (4x4)."""
    R = np.asarray(m)[:3, :3]
    yaw = np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


class Face:
    __slots__ = ("pts", "jaw", "yaw", "pitch", "roll", "center", "size")

    def __init__(self, pts, jaw, ypr):
        self.pts, self.jaw = pts, jaw
        self.yaw, self.pitch, self.roll = ypr
        lo, hi = pts.min(0), pts.max(0)
        self.center, self.size = (lo + hi) / 2, float((hi - lo).max())


class Landmarker:
    """MediaPipe FaceLandmarker (up to 3 faces) + track selection: the face nearest the previous one, else the
    largest. Video mode (tracks between frames, cheaper than per-frame detection)."""

    def __init__(self, num_faces=3):
        self.det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_TASK)), output_face_blendshapes=True,
            output_facial_transformation_matrixes=True, num_faces=num_faces, running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3, min_tracking_confidence=0.3))
        self.t = 0
        self.prev = None

    def detect(self, bgr):
        self.t += 33
        r = self.det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), self.t)
        h, w = bgr.shape[:2]
        faces = []
        for k, lm in enumerate(r.face_landmarks):
            pts = np.array([[p.x * w, p.y * h] for p in lm[:468]], np.float32)
            jaw = next((b.score for b in r.face_blendshapes[k] if b.category_name == "jawOpen"), np.nan)
            faces.append(Face(pts, jaw, euler(r.facial_transformation_matrixes[k])))
        return faces

    def __call__(self, bgr, near=None):
        """Main face: nearest to `near` (or the previous pick) among faces >= 50 % of the largest; else largest."""
        faces = self.detect(bgr)
        if not faces:
            self.prev = None
            return None
        big = max(f.size for f in faces)
        cand = [f for f in faces if f.size >= 0.5 * big]
        ref = near if near is not None else self.prev
        if ref is not None:
            f = min(cand, key=lambda f: np.linalg.norm(f.center - ref))
            if np.linalg.norm(f.center - ref) > 1.5 * f.size:      # previous face gone: take the largest
                f = max(cand, key=lambda f: f.size)
        else:
            f = max(cand, key=lambda f: f.size)
        self.prev = f.center
        return f


class Hands:
    def __init__(self):
        self.det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_TASK)), num_hands=2,
            running_mode=vision.RunningMode.VIDEO, min_hand_detection_confidence=0.3))
        self.t = 0

    def __call__(self, bgr):
        self.t += 33
        r = self.det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), self.t)
        h, w = bgr.shape[:2]
        return [np.array([[p.x * w, p.y * h] for p in hl], np.float32) for hl in r.hand_landmarks]


def hand_over_face(hands, face):
    """Fraction-like score 0..1: how much a hand covers the lower face (mouth/jaw ellipse)."""
    if face is None or not hands:
        return 0.0
    mouth = face.pts[[13, 14, 61, 291, 17, 0]].mean(0)
    jaw_w = np.linalg.norm(face.pts[234] - face.pts[454]) + 1e-3     # cheek to cheek
    best = 0.0
    for h in hands:
        d = np.linalg.norm(h - mouth, axis=1) / jaw_w                   # hand points near the mouth
        best = max(best, float(np.clip((0.6 - d.min()) / 0.25, 0, 1)))   # full at 0.35 jaw widths
    return best


def ramp_down(x, lo, hi):
    """1 below lo, 0 above hi, linear between."""
    return float(np.clip((hi - abs(x)) / (hi - lo), 0, 1))


def crop_rect(pts, scale, vy=-0.125):
    """LivePortrait-style crop (center, size, angle) from MediaPipe landmarks: axis eye-center -> lip-center."""
    eye = (pts[[33, 133, 159, 145]].mean(0) + pts[[362, 263, 386, 374]].mean(0)) / 2
    lip = pts[[13, 14, 78, 308]].mean(0)
    uy = lip - eye
    uy /= max(np.linalg.norm(uy), 1e-3)
    ux = np.array([uy[1], -uy[0]], np.float32)
    R = np.stack([ux, uy])
    c0 = pts.mean(0)
    r = (pts - c0) @ R.T
    lo, hi = r.min(0), r.max(0)
    size = float((hi - lo).max()) * scale
    center = c0 + R.T @ ((lo + hi) / 2) + uy * vy * size
    return np.array([center[0], center[1], size, np.arctan2(ux[1], ux[0])], np.float64)


def rect_to_M(rect, dsize):
    """original -> crop 2x3 similarity matrix."""
    cx, cy, size, ang = rect
    s = dsize / size
    c, sn = np.cos(ang), np.sin(ang)
    return np.array([[s * c, s * sn, dsize / 2 - s * (c * cx + sn * cy)],
                     [-s * sn, s * c, dsize / 2 - s * (-sn * cx + c * cy)]], np.float32)


# ---------------------------------------------------------------- LivePortrait retargeting
def channels_last(module):
    """cuDNN on sm_120 runs LivePortrait's fp16 conv3d (esp. the 7x7x7 occlusion head) on a slow vol2col path in
    NCDHW; channels-last weights make it pick tensor-core kernels (W+G ~97 -> see README)."""
    for m in module.modules():
        if isinstance(m, torch.nn.Conv3d):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
        elif isinstance(m, torch.nn.Conv2d):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)


class Retargeter:
    def __init__(self, model="human", region="lip", alpha=1.0, compile=False, lp_size=256, lip_mode="abs",
                 anchor=0.0, in_smooth=0.0):
        sys.path.insert(0, str(LP_DIR))
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.camera import get_rotation_matrix
        self.rot = get_rotation_matrix
        base = WEIGHTS / "liveportrait" / ("liveportrait/base_models" if model == "human"
                                           else "liveportrait_animals/base_models_v1.1")
        cfg = InferenceConfig()
        cfg.checkpoint_F = str(base / "appearance_feature_extractor.pth")
        cfg.checkpoint_M = str(base / "motion_extractor.pth")
        cfg.checkpoint_W = str(base / "warping_module.pth")
        cfg.checkpoint_G = str(base / "spade_generator.pth")
        cfg.checkpoint_S = str(WEIGHTS / "liveportrait/liveportrait/retargeting_models/stitching_retargeting_module.pth")
        cfg.flag_do_torch_compile = False
        self.w = LivePortraitWrapper(cfg)
        self.idx = {"lip": LIP_IDX, "eyes": EYE_IDX, "lip+eyes": LIP_IDX + EYE_IDX, "exp": list(range(21))}[region]
        self.alpha = alpha
        self.lip_mode = lip_mode
        self.neutral = torch.from_numpy(cfg.lip_array).float().to(self.w.device).reshape(1, 21, 3)
        self.anchor, self.in_smooth, self.base, self.e_in = anchor, in_smooth, None, None
        torch.backends.cudnn.benchmark = True
        self.G = self.w.spade_generator.half()
        if compile:   # ~35 -> 20 ms on the 5080 Laptop; compiling W or the whole step made it slower
            self.G = torch.compile(self.G, mode="max-autotune-no-cudagraphs")
        for mod in (w for w in (self.w.warping_module, self.w.spade_generator, self.w.appearance_feature_extractor)):
            channels_last(mod)
        mask = cv2.imread(str(LP_DIR / "src/utils/resources/mask_template.png"), cv2.IMREAD_GRAYSCALE)
        self.lp, self.gen = lp_size, 2 * lp_size     # F/W/G run at lp_size (a multiple of 128), G renders 2x
        self.face_mask = cv2.resize(mask, (self.gen, self.gen)).astype(np.float32) / 255.0

    @torch.no_grad()
    def __call__(self, in_crop, out_crop, out_lp=None):
        """in_crop, out_crop: 256x256 RGB uint8 (motion extractor), out_lp: lp_size crop of the output (appearance)
        -> (2*lp_size)^2 RGB uint8: output identity, input lips."""
        w = self.w
        x = torch.from_numpy(np.stack([in_crop, out_crop])).to(w.device).permute(0, 3, 1, 2).float() / 255.0
        kp = w.get_kp_info(x)
        xa = x[1:] if out_lp is None else \
            torch.from_numpy(out_lp).to(w.device).permute(2, 0, 1)[None].float() / 255.0
        f_s = w.extract_feature_3d(xa)
        s = {k: v[1:] for k, v in kp.items()}
        R = self.rot(s["pitch"], s["yaw"], s["roll"])
        x_s = w.transform_keypoint(s)
        exp = s["exp"].clone()
        e_in = kp["exp"][0:1]
        if self.in_smooth > 0:         # EMA on the driving keypoints: motion-extractor noise -> mouth shimmer
            self.e_in = e_in if self.e_in is None else self.in_smooth * self.e_in + (1 - self.in_smooth) * e_in
            e_in = self.e_in
        if self.lip_mode == "abs":     # the input's expression keypoints replace the output's
            exp[:, self.idx] = self.alpha * e_in[:, self.idx] + (1 - self.alpha) * exp[:, self.idx]
        else:                          # rel: input's deviation from a neutral closed mouth added to the output's mouth
            base = exp
            if self.anchor > 0:        # ... to a slow average of it, so JoyAI's own per-chunk mouth wobble is
                self.base = exp.clone() if self.base is None else \
                    (1 - self.anchor) * self.base + self.anchor * exp       # warped back to a stable mouth
                base = self.base
            exp[:, self.idx] = base[:, self.idx] + self.alpha * (e_in[:, self.idx] - self.neutral[:, self.idx])
        x_d = (s["kp"] @ R + exp) * s["scale"][..., None]
        x_d[:, :, :2] += s["t"][:, None, :2]
        x_d = w.stitching(x_s, x_d)
        with torch.autocast("cuda", dtype=torch.float16):
            feat = w.warping_module(f_s, kp_source=x_s, kp_driving=x_d)["out"]
        out = self.G(feature=feat.half().contiguous(memory_format=torch.channels_last)).float()
        return (out[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()


# ---------------------------------------------------------------- face restoration
FFHQ5 = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                  [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)


def five_points(pts):
    return np.stack([pts[[33, 133]].mean(0), pts[[362, 263]].mean(0), pts[1], pts[61], pts[291]]).astype(np.float32)


class Restorer:
    def __init__(self, kind, fidelity=0.7):
        import spandrel
        import spandrel_extra_arches
        spandrel_extra_arches.install()
        f = {"gfpgan": "GFPGANv1.4.pth", "codeformer": "codeformer.pth", "restoreformer": "RestoreFormer.pth"}[kind]
        self.kind, self.fid = kind, fidelity
        self.fp16 = True
        self.m = spandrel.ModelLoader().load_from_file(str(WEIGHTS / "restore" / f)).model.cuda().eval()
        # fp16 GFPGAN overflows (flat brown face); CodeFormer runs under autocast
        m = np.zeros((512, 512), np.float32)
        cv2.rectangle(m, (40, 40), (472, 472), 1.0, -1)
        self.mask = cv2.GaussianBlur(m, (0, 0), 20)

    @torch.no_grad()
    def __call__(self, frame, pts, blend=1.0):
        M, _ = cv2.estimateAffinePartial2D(five_points(pts), FFHQ5, method=cv2.LMEDS)
        face = cv2.warpAffine(frame, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        x = torch.from_numpy(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).cuda().permute(2, 0, 1)[None].float() / 127.5 - 1
        if self.kind == "codeformer":
            with torch.autocast("cuda", dtype=torch.float16):
                y = self.m(x, weight=self.fid)[0]
        elif self.kind == "gfpgan":
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                y = self.m(x, return_rgb=False, randomize_noise=False)[0]
        else:
            y = self.m(x)[0]
        y = ((y[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy()
        y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
        Mi = cv2.invertAffineTransform(M)
        h, w = frame.shape[:2]
        back = cv2.warpAffine(y, Mi, (w, h), flags=cv2.INTER_LINEAR)
        mk = cv2.warpAffine(self.mask, Mi, (w, h))[..., None] * blend
        return (back * mk + frame * (1 - mk)).astype(np.uint8)


def sharpen(frame, rect, amount=0.8, sigma=1.2):
    """Unsharp mask inside a feathered ellipse around the face (cheap baseline; only the face ROI is filtered)."""
    cx, cy, size, _ = rect
    h, w = frame.shape[:2]
    r = int(size * 0.4)
    x0, y0, x1, y1 = max(0, int(cx) - r), max(0, int(cy) - r), min(w, int(cx) + r), min(h, int(cy) + r)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return frame
    roi = frame[y0:y1, x0:x1]
    m = np.zeros(roi.shape[:2], np.float32)
    cv2.ellipse(m, (int(cx) - x0, int(cy) - y0), (int(size * 0.28), int(size * 0.36)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), size * 0.05 + 1)[..., None]
    sharp = cv2.addWeighted(roi, 1 + amount, cv2.GaussianBlur(roi, (0, 0), sigma), -amount, 0)
    out = frame.copy()
    out[y0:y1, x0:x1] = (sharp * m + roi * (1 - m)).astype(np.uint8)
    return out


# ---------------------------------------------------------------- the per-frame pass (offline and live)
def lower_ramp(n, lo=0.36, hi=0.58):
    y = np.linspace(0, 1, n, dtype=np.float32)
    return np.clip((y - lo) / (hi - lo), 0, 1)[:, None] * np.ones((1, n), np.float32)


def mouth_geom(pts, width_k=1.0, top_k=1.5):
    """Mouth/chin mask geometry from face landmarks: [mx, my, half_width, y_top, y_bottom, angle] with (mx, my) the
    mouth centre, y offsets along the mouth's down axis (top: towards the nose, bottom: below the chin + 25 %)."""
    L, R, top, chin, c13, c14 = pts[61], pts[291], pts[2], pts[152], pts[13], pts[14]
    mw = np.linalg.norm(R - L)
    ux = (R - L) / (mw + 1e-6)
    uy = np.array([-ux[1], ux[0]])
    if uy @ (chin - top) < 0:
        uy = -uy
    mc = (L + R + c13 + c14) / 4
    y_top = (top - mc) @ uy * top_k
    y_bot = (chin - mc) @ uy
    y_bot += 0.25 * (y_bot - (top - mc) @ uy)
    return np.array([mc[0], mc[1], width_k * mw, y_top, y_bot, np.arctan2(ux[1], ux[0])])


def mask_from_geom(g, w, h, feather=5.0, shape="box"):
    """Soft mask (ROI + float mask) for a mouth geometry. shape box = rounded rectangle (superellipse p=5), wide
    mouths like G-Star's; ellipse = p=2."""
    mx, my, hw, yt, yb, ang = g
    ux = np.array([np.cos(ang), np.sin(ang)])
    uy = np.array([-ux[1], ux[0]])
    c = np.array([mx, my]) + uy * (yt + yb) / 2
    ax, ay = hw, (yb - yt) / 2
    r = int(np.hypot(ax, ay) + 2 * feather + 4)
    x0, y0 = max(int(c[0]) - r, 0), max(int(c[1]) - r, 0)
    x1, y1 = min(int(c[0]) + r, w), min(int(c[1]) + r, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0, 0, 1, 1, np.zeros((1, 1), np.float32)
    sh = 1.5 * feather                               # shrink so the blur tail stays inside the outline
    p = 5.0 if shape == "box" else 2.0
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ct, st_ = np.cos(t), np.sin(t)
    u = max(ax - sh, 1) * np.sign(ct) * np.abs(ct) ** (2 / p)
    v = max(ay - sh, 1) * np.sign(st_) * np.abs(st_) ** (2 / p)
    poly = np.stack([c[0] + u * ux[0] + v * uy[0] - x0, c[1] + u * ux[1] + v * uy[1] - y0], 1)
    inside = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(inside, [np.round(poly * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
    m = cv2.GaussianBlur(inside, (0, 0), feather / 2)
    m[m < 1e-3] = 0
    return x0, y0, x1, y1, m


class RelTrack:
    """Geometry tracked relative to the INPUT face (stable, real) and filtered hard. A measurement on the generated
    face is rejected as a tracking error if it jumps more than `jump` face sizes from the previous measurement, or
    lands more than 2.5x that from the current state (or its size changes > 25 %); the state is then held and the
    caller fades the effect out. Accepted measurements are blended in with an adaptive EMA (alpha .. 0.6 when it
    moves consistently). A new position that persists `persist` frames is taken as real (a "switch")."""

    def __init__(self, alpha=0.15, jump=0.08, persist=10, confirm=3, size_idx=2):
        self.alpha, self.jump, self.persist, self.confirm, self.si = alpha, jump, persist, confirm, size_idx
        self.v, self.prev, self.cand, self.bad = None, None, [], 0
        self.stats = {"accepted": 0, "rejected": 0, "switches": 0, "missing": 0}

    def dist(self, a, b):
        return np.linalg.norm(a[:2] - b[:2]), abs(a[self.si] / (b[self.si] + 1e-9) - 1)

    def close(self, a, b, k=1.0):
        d, ds = self.dist(a, b)
        return d < k * self.jump and ds < 0.25

    def update(self, m):
        if m is None:
            self.stats["missing"] += 1
            self.bad += 1
            self.prev = None
            return self.v, False
        prev, self.prev = self.prev, m
        if self.v is None:
            self.cand = [c for c in self.cand if self.close(c, m)] + [m]
            if len(self.cand) >= self.confirm:
                self.v = np.mean(self.cand, 0)
                self.cand = []
            return self.v, self.v is not None
        jumped = prev is not None and not self.close(m, prev)
        if not jumped and self.close(m, self.v, 2.5):
            d = m - self.v
            d[-1] = (d[-1] + np.pi) % (2 * np.pi) - np.pi
            k = float(np.clip(self.alpha + 0.3 * self.dist(m, self.v)[0] / self.jump, self.alpha, 0.6))
            self.v = self.v + k * d
            self.bad = 0
            self.cand = []
            self.stats["accepted"] += 1
            return self.v, True
        self.stats["rejected"] += 1
        self.cand = [c for c in self.cand if self.close(c, m)] + [m]
        if len(self.cand) >= self.persist:                # it stayed there: a real change
            self.v, self.cand = np.mean(self.cand[-self.confirm:], 0), []
            self.stats["switches"] += 1
            return self.v, True
        self.bad += 1
        return self.v, False


def face_frame(f):
    """Reference frame of the (input) face: origin at the nose, scale = face size, angle = eye line."""
    p = f.pts
    el, er = p[[33, 133]].mean(0), p[[362, 263]].mean(0)
    return p[[1, 4, 6, 168]].mean(0), f.size, np.arctan2(er[1] - el[1], er[0] - el[0])   # nose: rigid, not the jaw


def to_rel(v, fr, pos=(0, 1), lens=(), ang_idx=-1):
    o, s, a = fr
    c, sn = np.cos(-a), np.sin(-a)
    r = np.array(v, float).copy()
    d = r[list(pos)] - o
    r[list(pos)] = np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]]) / s
    for i in lens:
        r[i] /= s
    r[ang_idx] = (r[ang_idx] - a + np.pi) % (2 * np.pi) - np.pi
    return r


def from_rel(r, fr, pos=(0, 1), lens=(), ang_idx=-1):
    o, s, a = fr
    c, sn = np.cos(a), np.sin(a)
    v = np.array(r, float).copy()
    d = v[list(pos)] * s
    v[list(pos)] = o + np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])
    for i in lens:
        v[i] *= s
    v[ang_idx] = v[ang_idx] + a
    return v


GEOM_LENS, RECT_LENS = (2, 3, 4), (2,)


class OneEuro:
    """One-Euro filter on arrays (landmarks): smooths jitter at rest, follows fast moves without lag."""

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0, fps=25.0):
        self.mc, self.beta, self.dc, self.dt = min_cutoff, beta, d_cutoff, 1.0 / fps
        self.x = self.dx = None

    def alpha(self, cutoff):
        r = 2 * np.pi * cutoff * self.dt
        return r / (r + 1)

    def __call__(self, x, scale=1.0):
        """scale: normalise speed (e.g. by face size) so the filter is resolution independent."""
        if x is None:
            self.x = self.dx = None
            return None
        if self.x is None or x.shape != self.x.shape:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return x
        dx = (x - self.x) / self.dt / scale
        self.dx = self.dx + self.alpha(self.dc) * (dx - self.dx)
        cutoff = self.mc + self.beta * np.abs(self.dx)
        a = self.alpha(cutoff)
        self.x = self.x + a * (x - self.x)
        return self.x.copy()


class FacePass:
    """Input (real) frame + JoyAI output frame -> refined output frame.

    Robustness: the effect strength `weight` (0..1) is the product of gates (driving face found, output face
    found or mapped from the input, head pose inside the reliable range, no hand over the mouth, input/output
    faces agree in position) and moves towards that target with an EMA, so dropouts, turns and occlusions fade the
    effect out/in over a few frames instead of popping. weight ~0 -> the JoyAI frame passes through untouched
    (and the GPU work is skipped)."""

    def __init__(self, a):
        self.a = a
        self.lm_in, self.lm_out = Landmarker(a.num_faces), Landmarker(a.num_faces)
        self.hands = Hands() if a.hands else None
        self.rt = Retargeter(a.model, a.region, a.alpha, a.compile, a.lp_size, a.lip_mode,
                             a.rel_anchor, a.in_smooth) if a.region != "none" else None
        self.rs = Restorer(a.restore, a.fidelity) if a.restore in ("gfpgan", "codeformer", "restoreformer") else None
        g = self.rt.gen if self.rt else 512
        self.mask = (self.rt.face_mask if self.rt else np.ones((g, g), np.float32)) * \
            (lower_ramp(g) if a.mask == "lower" else 1.0)
        self.weight, self.rect, self.last_in, self.rin = 0.0, None, None, None   # last_in: (in256, rin) for fading out on loss
        self.euro_in, self.euro_out = OneEuro(a.euro_cutoff, a.euro_beta), OneEuro(a.euro_cutoff, a.euro_beta)
        self.trk_rect = RelTrack(a.track_alpha, a.track_jump, a.track_persist)
        self.trk_mask = RelTrack(a.track_alpha, a.track_jump, a.track_persist)
        self.mask_geom = None
        self.last_mask = None
        self.paste_mask = None
        if a.track == "flow":
            from track import FlowTracker
            self.flow = FlowTracker()
        elif a.track == "sam2" and a.method == "lp":
            from track import SamTracker
            self.sam = SamTracker()

    def smooth_rect(self, rect):
        if rect is None or self.rect is None:
            return rect
        d = np.linalg.norm(rect[:2] - self.rect[:2]) / rect[2]
        if d > 0.25 or abs(rect[2] / self.rect[2] - 1) > 0.3:     # jump (new face / fast move): no lag
            return rect
        k = self.a.smooth * (1 - d / 0.25)                       # less smoothing the faster it moves
        da = (rect[3] - self.rect[3] + np.pi) % (2 * np.pi) - np.pi
        out = k * self.rect + (1 - k) * rect
        out[3] = rect[3] - k * da
        return out

    def analyze_input(self, I):
        """Input-side analysis (face + hand-over-mouth score). Live: run at capture time, keyed by seq."""
        fi = self.lm_in(I)
        hand = hand_over_face(self.hands(I), fi) if self.hands is not None and fi is not None else 0.0
        return fi, hand

    def analyze(self, I, O, pre=None):
        """CPU part: landmarks, gates, weight EMA, crop rects and the input crop. Runs in order (stateful), can
        overlap with render() of the previous frame. `pre`: analyze_input(I) computed earlier (live: at capture)."""
        a, info = self.a, {}
        t0 = time.perf_counter()
        fi, hand = pre if pre is not None else self.analyze_input(I)
        fo = self.lm_out(O, near=fi.center if fi is not None else None)
        if a.euro_cutoff > 0:                              # One-Euro smoothing of the landmarks (jitter -> shimmer)
            if fi is not None:
                fi = Face(self.euro_in(fi.pts, fi.size), fi.jaw, (fi.yaw, fi.pitch, fi.roll))
            else:
                self.euro_in(None)
            if fo is not None:
                fo = Face(self.euro_out(fo.pts, fo.size), fo.jaw, (fo.yaw, fo.pitch, fo.roll))
            else:
                self.euro_out(None)

        gates = {"in_face": fi is not None}
        if fi is not None:
            gates["pose_in"] = ramp_down(fi.yaw, a.yaw_lo, a.yaw_hi) * ramp_down(fi.pitch, a.pitch_lo, a.pitch_hi) \
                * ramp_down(fi.roll, 40, 60)
            gates["hand"] = 1.0 - hand
            info.update(yaw=round(fi.yaw, 1), pitch=round(fi.pitch, 1))
        rect, src = None, "none"
        if fo is not None and (fi is None or np.linalg.norm(fo.center - fi.center) < a.max_offset * fi.size):
            rect, src = crop_rect(fo.pts, a.crop_scale), "out"
            gates["pose_out"] = ramp_down(fo.yaw, a.yaw_lo, a.yaw_hi) * ramp_down(fo.pitch, a.pitch_lo, a.pitch_hi)
        elif fi is not None:                                   # output face not found / disagrees: map input box
            rect, src = crop_rect(fi.pts, a.crop_scale), "in"
            rect[2] *= a.box_scale
            rect[1] += a.box_dy * rect[2]
            gates["mapped"] = a.mapped_weight
        meas_geom = None
        if a.stabilize and fi is not None:
            # crop rect and mouth mask relative to the input face, hard-filtered (G-Star's landmarks jump around)
            fr = face_frame(fi)
            ok_out = fo is not None and np.linalg.norm(fo.center - fi.center) < a.max_offset * fi.size
            mr = crop_rect(fo.pts, a.crop_scale) if ok_out else None
            meas_geom = mouth_geom(fo.pts, a.chin_width, a.chin_top) if ok_out else None
            vr, okr = self.trk_rect.update(None if mr is None else to_rel(mr, fr, lens=RECT_LENS))
            vg, okg = self.trk_mask.update(None if meas_geom is None else to_rel(meas_geom, fr, lens=GEOM_LENS))
            if a.track == "sam2":                              # bind to G-Star's actual mouth: SAM 2 segment
                ts = time.perf_counter()
                seg, oks = self.sam.update(O, fo.pts if ok_out else None)
                info["ms_sam"] = round(1000 * (time.perf_counter() - ts), 1)
                pm = paste_mask_from_seg(seg, a.feather) if seg is not None else None
                if vr is not None and pm is not None:
                    rect = from_rel(vr, fr, lens=RECT_LENS)
                    self.paste_mask = pm
                    ys, xs = np.nonzero(seg)
                    self.mask_geom = np.array([xs.mean(), ys.mean(), np.ptp(xs) / 2, 0, 0, 0])   # for stats
                    gates["track"] = 1.0 if oks or self.sam.bad <= a.track_hold else 0.0
                    gates.pop("mapped", None)
                else:
                    gates["track"] = 0.0
            elif a.track == "flow":                            # bind to G-Star's pixels: optical-flow tracker
                fg, okf = self.flow.update(O, meas_geom)
                if fg is not None:                             # flow can accumulate rotation (head turns, hands):
                    da = (fg[5] - fr[2] + np.pi) % (2 * np.pi) - np.pi      # keep the mouth line within 12 deg
                    lim = np.radians(12)                       # of the real face's eye line
                    if abs(da) > lim:
                        fg[5] = fr[2] + np.clip(da, -lim, lim)
                        self.flow.g[5] = fg[5]
                if vr is not None and fg is not None:
                    rect = from_rel(vr, fr, lens=RECT_LENS)
                    self.mask_geom = fg.copy()
                    gates["track"] = 1.0 if okf or self.flow.bad <= a.track_hold else 0.0
                    gates.pop("mapped", None)
                else:
                    gates["track"] = 0.0
            elif vr is not None and vg is not None:
                rect = from_rel(vr, fr, lens=RECT_LENS)
                self.mask_geom = from_rel(vg, fr, lens=GEOM_LENS)
                # rejected (jumped / wrong place): hold the geometry and fade out; output face merely not found
                # for a few frames: the input-anchored geometry is a good prior, keep going for up to 8 frames
                held = mr is None and self.trk_mask.bad <= a.track_hold
                gates["track"] = 1.0 if (okr and okg) or held else 0.0
                gates.pop("mapped", None)
            else:
                gates["track"] = 0.0                               # not locked yet
            info["fsize"] = round(float(fi.size), 1)
            info["mask_geom"] = None if self.mask_geom is None else [round(float(x), 2) for x in self.mask_geom]
            info["meas_geom"] = None if meas_geom is None else [round(float(x), 2) for x in meas_geom]
        target = float(np.prod([float(v) for v in gates.values()])) if fi is not None and rect is not None else 0.0
        rate = a.fade_in if target > self.weight else a.fade_out
        self.weight += rate * (target - self.weight)
        if self.weight < 0.02 and target == 0:
            self.weight = 0.0
        if not a.stabilize:
            rect = self.smooth_rect(rect)
        if rect is not None:
            self.rect = rect
        if fi is not None and self.rt is not None:
            rin = crop_rect(fi.pts, a.crop_scale)
            self.rin = rin
            self.last_in = cv2.cvtColor(cv2.warpAffine(I, rect_to_M(rin, 256), (256, 256),
                                                       borderMode=cv2.BORDER_REPLICATE), cv2.COLOR_BGR2RGB)
        info.update(src=src, target=round(target, 2), weight=round(self.weight, 2),
                    gates={k: round(float(v), 2) for k, v in gates.items()},
                    jaw_in=fi.jaw if fi is not None else np.nan, jaw_out=fo.jaw if fo is not None else np.nan)
        info["ms_cpu"] = 1000 * (time.perf_counter() - t0)
        rpts = fo.pts if fo is not None else (fi.pts if fi is not None else None)
        return {"O": O, "I": I, "fi_pts": fi.pts if fi is not None else None,
                "fo_pts": fo.pts if fo is not None else None, "info": info, "weight": self.weight, "rect": None if self.rect is None else self.rect.copy(),
                "cin": self.last_in, "pts": rpts,
                "mask_geom": None if self.mask_geom is None else self.mask_geom.copy(),
                "paste_mask": self.paste_mask if a.track == "sam2" else None}

    def render(self, st):
        """GPU render + CPU compose (the live bridge runs the two in separate threads)."""
        return self.compose(self.render_gpu(st))

    def render_gpu(self, st):
        """GPU part: LivePortrait render of the output face with the input's lips (skipped at weight ~0)."""
        st = dict(st)
        st["t1"] = time.perf_counter()
        st["gen"] = None
        if self.rt is not None and st["weight"] > 0.01 and st["rect"] is not None and st["cin"] is not None:
            g = self.rt.gen
            Mo = rect_to_M(st["rect"], g)
            cg = cv2.warpAffine(st["O"], Mo, (g, g), borderMode=cv2.BORDER_REPLICATE)
            c256 = cv2.cvtColor(cv2.resize(cg, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            clp = cv2.cvtColor(cv2.resize(cg, (self.rt.lp, self.rt.lp), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB) if self.rt.lp != 256 else None
            st["gen"], st["Mo"] = cv2.cvtColor(self.rt(st["cin"], c256, clp), cv2.COLOR_RGB2BGR), Mo
        return st

    def compose(self, st):
        """CPU part: mask, colour match, detail transfer, paste, optional restoration."""
        a, O, info = self.a, st["O"], st["info"]
        h, w = O.shape[:2]
        wgt, rect = st["weight"], st["rect"]
        self.last_mask = None
        t1 = st["t1"]
        res = O
        if st["gen"] is not None:
            g, gen, Mo = self.rt.gen, st["gen"], st["Mo"]
            Minv = cv2.invertAffineTransform(Mo)
            fpts = st["fo_pts"] if st["fo_pts"] is not None else st["fi_pts"]
            if a.mask == "chin" and st.get("paste_mask") is not None:   # SAM 2 segment-based mask
                x0, y0, x1, y1, m = st["paste_mask"]
                self.last_mask = (x0, y0, m)
            elif a.mask == "chin" and (st.get("mask_geom") is not None or fpts is not None):
                # tight mask below the nose in frame pixels: everything else stays JoyAI's, pixel-identical
                geom = st["mask_geom"] if st.get("mask_geom") is not None else \
                    mouth_geom(fpts, a.chin_width, a.chin_top)
                x0, y0, x1, y1, m = mask_from_geom(geom, w, h, a.feather, a.mask_shape)
                self.last_mask = (x0, y0, m)
            else:
                mk = self.mask
                if a.mask == "mouth" and fpts is not None:
                    mk = self.mouth_mask(fpts, Mo, g)
                x0, y0, x1, y1 = self.roi(Minv, g, w, h)
                Mr0 = Minv.copy()
                Mr0[:, 2] -= (x0, y0)
                m = cv2.warpAffine(mk, Mr0, (x1 - x0, y1 - y0))
            Mr = Minv.copy()
            Mr[:, 2] -= (x0, y0)
            back = cv2.warpAffine(gen, Mr, (x1 - x0, y1 - y0), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE).astype(np.float32)   # one resampling step
            roi = O[y0:y1, x0:x1].astype(np.float32)
            m4 = m[::4, ::4, None]                             # colour-match the render to JoyAI inside the mask
            ws = m4.sum() + 1e-6                               # (stats on a 4x subsampled grid: cheap)
            back += ((roi[::4, ::4] * m4).sum((0, 1)) - (back[::4, ::4] * m4).sum((0, 1))) / ws
            if a.detail == "sep":
                # frequency separation: low band from the render, high band (texture, edges) from JoyAI where
                # both agree at low frequency; where they differ (open-mouth interior, moved lips) the render's
                # full band is used
                sg = a.detail_sigma
                lb, lo = cv2.GaussianBlur(back, (0, 0), sg), cv2.GaussianBlur(roi, (0, 0), sg)
                dd = cv2.cvtColor(cv2.absdiff(lb, lo), cv2.COLOR_BGR2GRAY)
                k = np.clip(1 - dd * (1.0 / a.detail_thr), 0, 1)[..., None]
                back = lb + k * (roi - lo) + (1 - k) * (1 + a.render_sharpen) * (back - lb)
            mm = (m * wgt)[..., None]
            res = O.copy()
            res[y0:y1, x0:x1] = np.clip(back * mm + roi * (1 - mm) + 0.5, 0, 255).astype(np.uint8)
            if a.mask == "chin":                               # outside the mask: exactly JoyAI's pixels
                out = mm[..., 0] <= 0
                res[y0:y1, x0:x1][out] = O[y0:y1, x0:x1][out]
                ep = st["fo_pts"] if st["fo_pts"] is not None else fpts
                if ep is not None:                             # verify: eyes/brows untouched
                    e = ep[[70, 105, 107, 336, 334, 300, 33, 133, 145, 362, 263, 374]]
                    ex0, ey0 = np.maximum(e.min(0).astype(int) - 5, 0)
                    ex1, ey1 = e.max(0).astype(int) + 5
                    if ex1 > ex0 and ey1 > ey0:
                        info["eye_diff"] = int(np.abs(res[ey0:ey1, ex0:ex1].astype(int) - O[ey0:ey1, ex0:ex1]).max())
        t2 = time.perf_counter()
        if a.restore == "sharpen" and rect is not None and wgt > 0.01:
            res = sharpen(res, rect, a.sharpen * wgt)
        elif self.rs is not None and wgt > 0.01 and st["pts"] is not None:
            res = self.rs(res, st["pts"], a.restore_blend * wgt)
        t3 = time.perf_counter()
        info["ms"] = (info["ms_cpu"], 1000 * (t2 - t1), 1000 * (t3 - t2))
        return res, info

    def __call__(self, I, O):
        return self.render(self.analyze(I, O))

    @staticmethod
    def chin_mask(pts, w, h, feather=8.0, top_k=1.0, width_k=0.85):
        """Ellipse from just below the nose (subnasale) to below the chin, as wide as the mouth + margin, rotated
        with the mouth line, feathered by `feather` px inside its edge. Returns ROI (x0, y0, x1, y1) and the mask."""
        L, R, top, chin, c13, c14 = pts[61], pts[291], pts[2], pts[152], pts[13], pts[14]
        mw = np.linalg.norm(R - L)
        ux = (R - L) / (mw + 1e-6)
        uy = np.array([-ux[1], ux[0]])
        if uy @ (chin - top) < 0:
            uy = -uy
        mc = (L + R + c13 + c14) / 4
        y_top = (top - mc) @ uy * top_k              # negative: nose is above the mouth
        y_bot = (chin - mc) @ uy
        y_bot += 0.25 * (y_bot - y_top)              # room for the jaw to drop
        cy = (y_top + y_bot) / 2
        ax, ay = width_k * mw, (y_bot - y_top) / 2
        c = mc + uy * cy
        r = int(max(ax, ay) + 2 * feather + 4)
        x0, y0 = max(int(c[0]) - r, 0), max(int(c[1]) - r, 0)
        x1, y1 = min(int(c[0]) + r, w), min(int(c[1]) + r, h)
        m = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1)), np.float32)
        ang = np.degrees(np.arctan2(ux[1], ux[0]))
        sh = 1.5 * feather                           # shrink so the blur tail stays inside the ellipse
        cv2.ellipse(m, (int(round(c[0] - x0)), int(round(c[1] - y0))),
                    (max(int(ax - sh), 1), max(int(ay - sh), 1)), ang, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), feather / 2)
        m[m < 1e-3] = 0
        return x0, y0, x1, y1, m

    @staticmethod
    def mouth_mask(pts, Mo, g):
        """Feathered ellipse around the mouth (lips + 60 % margin, down to the chin) in crop coords: only the mouth
        is replaced, cheeks/jaw stay JoyAI's (less visible warping)."""
        p = pts[[61, 291, 0, 17, 13, 14, 152]] @ Mo[:, :2].T + Mo[:, 2]
        c = p[:6].mean(0)
        mw = np.linalg.norm(p[1] - p[0])
        mh = max(np.linalg.norm(p[3] - p[2]), 0.35 * mw)
        m = np.zeros((g, g), np.float32)
        ang = np.degrees(np.arctan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0]))
        cv2.ellipse(m, (int(c[0]), int(c[1] + 0.15 * mh)), (int(0.8 * mw), int(0.5 * mh + 0.35 * mw)), ang, 0, 360,
                    1.0, -1)
        return cv2.GaussianBlur(m, (0, 0), 0.12 * mw + 1)

    @staticmethod
    def roi(Minv, g, w, h):
        c = np.array([[0, 0, 1], [g, 0, 1], [0, g, 1], [g, g, 1]], np.float32) @ Minv.T
        x0, y0 = np.clip(np.floor(c.min(0)).astype(int), 0, [w - 1, h - 1])
        x1, y1 = np.clip(np.ceil(c.max(0)).astype(int) + 1, 1, [w, h])
        return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def make_pass(a):
    """The face pass selected by --method (all share analyze_input / analyze / render)."""
    if a.method == "swap":
        from swap import SwapPass
        return SwapPass(a)
    return FacePass(a)


def add_args(p):
    p.add_argument("--method", default="lp", choices=["lp", "swap"],
                   help="lp: LivePortrait lip retargeting on the generated face (any ref, incl. cartoons); swap: face "
                        "swap of the real face with the reference identity, composited into the output (human refs)")
    p.add_argument("--ref", default=None, help="reference image (identity) for --method swap")
    p.add_argument("--swap-model", default="hyperswap", choices=["hyperswap", "inswapper"])
    p.add_argument("--swap-mask", default="lower", choices=["face", "lower"],
                   help="swap: paste below the nose (default: keeps JoyAI's eyes, no glasses carried over) or the inner face")
    p.add_argument("--euro-cutoff", type=float, default=1.5, help="One-Euro min cutoff (Hz) on landmarks, 0 = off")
    p.add_argument("--euro-beta", type=float, default=0.3)
    p.add_argument("--region", default="lip", choices=["none", "lip", "eyes", "lip+eyes", "exp"])
    p.add_argument("--lip-mode", default="abs", choices=["abs", "rel"],
                   help="abs: input lips replace the output's (human refs); rel: input's mouth motion is added to "
                        "the output's own mouth shape (cartoon refs keep their grin)")
    p.add_argument("--rel-anchor", type=float, default=0.01, help="rel: EMA rate of the output-mouth base "
                   "(0 = per-frame: JoyAI's mouth wobble passes through; 0.01 = ~4 s average)")
    p.add_argument("--in-smooth", type=float, default=0.3, help="EMA weight of the previous driving keypoints")
    p.add_argument("--alpha", type=float, default=1.0, help="blend of the input's expression keypoints")
    p.add_argument("--model", default="human", choices=["human", "animal"])
    p.add_argument("--lp-size", type=int, default=256, choices=[128, 256, 384],
                   help="appearance/warp/decoder crop size (128 = 4x less compute, renders 256 px)")
    p.add_argument("--crop-scale", type=float, default=2.0)
    p.add_argument("--box-scale", type=float, default=1.0, help="size factor for an input-derived box on the output")
    p.add_argument("--box-dy", type=float, default=0.0, help="shift of an input-derived box, in box sizes")
    p.add_argument("--mapped-weight", type=float, default=0.7, help="effect weight when the box is mapped from input")
    p.add_argument("--max-offset", type=float, default=0.5, help="output face must be within this many face sizes "
                   "of the input face, else the input box is used")
    p.add_argument("--mask", default="chin", choices=["chin", "face", "lower", "mouth"],
                   help="lp: chin = below the nose only, rest pixel-identical to JoyAI (default); face / lower half / "
                        "mouth ellipse = older variants")
    p.add_argument("--feather", type=float, default=5.0, help="chin mask feather in frame pixels")
    p.add_argument("--chin-width", type=float, default=1.2, help="mask half-width in mouth widths (G-Star's grin runs past the detected corners)")
    p.add_argument("--mask-shape", default="box", choices=["box", "ellipse"], help="chin mask: rounded rectangle "
                   "(wide mouths, G-Star) or ellipse")
    p.add_argument("--no-stabilize", dest="stabilize", action="store_false", help="per-frame crop/mask from the "
                   "output landmarks (old behaviour) instead of input-anchored, jump-rejecting tracks")
    p.add_argument("--track", default="flow", choices=["flow", "sam2", "anchor"],
                   help="mouth-mask tracker: flow = optical flow on the generated face + slow MediaPipe re-anchoring; "
                        "anchor = MediaPipe geometry relative to the input face (previous version)")
    p.add_argument("--track-alpha", type=float, default=0.15, help="EMA rate of accepted crop/mask measurements")
    p.add_argument("--track-jump", type=float, default=0.08, help="reject measurements that move more than this "
                   "many face sizes (tracking errors)")
    p.add_argument("--track-hold", type=int, default=8, help="frames to keep the effect on the held geometry while "
                   "the generated face is not detected")
    p.add_argument("--track-persist", type=int, default=10, help="frames a jump must persist to be taken as real")
    p.add_argument("--chin-top", type=float, default=1.5, help="mask top as a multiple of the mouth->subnasale "
                   "distance (1 = at the nose; >1 reaches above it)")
    p.add_argument("--detail", default="sep", choices=["sep", "none"],
                   help="sep: keep JoyAI's high frequencies inside the mask where the render agrees at low frequency")
    p.add_argument("--detail-sigma", type=float, default=2.0)
    p.add_argument("--render-sharpen", type=float, default=0.6, help="boost of the render's own high band where it "
                   "is used (open-mouth interior); the 128 px render is soft")
    p.add_argument("--detail-thr", type=float, default=20.0, help="low-band difference (grey levels) at which the "
                   "render's full band takes over")
    p.add_argument("--yaw-lo", type=float, default=25), p.add_argument("--yaw-hi", type=float, default=40)
    p.add_argument("--pitch-lo", type=float, default=20), p.add_argument("--pitch-hi", type=float, default=35)
    p.add_argument("--no-hands", dest="hands", action="store_false", help="skip the hand-over-mouth check")
    p.add_argument("--num-faces", type=int, default=3)
    p.add_argument("--fade-in", type=float, default=0.25, help="EMA rate of the weight towards a higher target")
    p.add_argument("--fade-out", type=float, default=0.4)
    p.add_argument("--smooth", type=float, default=0.5, help="EMA weight of the previous crop rect (slow motion)")
    p.add_argument("--restore", default="none", choices=["none", "gfpgan", "codeformer", "restoreformer", "sharpen"])
    p.add_argument("--fidelity", type=float, default=0.7, help="CodeFormer w (1 = keep input, 0 = max restore)")
    p.add_argument("--restore-blend", type=float, default=1.0)
    p.add_argument("--sharpen", type=float, default=0.8)
    p.add_argument("--compile", action="store_true", help="torch.compile the decoder (first frames slow)")


def overlay(img, info):
    img = img.copy()
    wgt = info.get("weight", 0)
    cv2.rectangle(img, (4, img.shape[0] - 14), (4 + int(150 * wgt), img.shape[0] - 4), (0, 255, 0), -1)
    txt = f"w={wgt:.2f} {info.get('src')} yaw={info.get('yaw', '-')} " + \
        " ".join(f"{k}={v}" for k, v in info.get("gates", {}).items() if v < 0.99)
    cv2.putText(img, txt, (4, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="driving video (the real webcam frames as sent)")
    p.add_argument("--output", required=True, help="generated video (JoyAI output)")
    p.add_argument("--out", required=True, help="refined video (frame-aligned with --output)")
    p.add_argument("--sbs", default=None, help="side-by-side review video: input | JoyAI output | refined (+ gates)")
    p.add_argument("--zoom", default=None, help="face close-up review video: input | JoyAI output | refined")
    p.add_argument("--stats", default=None, help="write timings, alignment, weights as JSON")
    p.add_argument("--align", default="dtw", choices=["dtw", "linear"])
    p.add_argument("--seq-json", default=None, help="joyai_client --stats JSON with source_seq: exact pairing "
                   "(output i was made from input source_seq[i]-1); overrides --align")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--debug-mask", action="store_true", help="draw the paste mask outline in --sbs / --zoom")
    p.add_argument("--eval-id", action="store_true", help="swap: ArcFace cosine to the ref every 5th frame")
    p.add_argument("--pipeline", action="store_true", help="overlap analyze (CPU, next frame) with render (GPU) "
                   "in two threads, like the live bridge would; reports wall-clock fps")
    p.add_argument("--fps", type=float, default=25)
    add_args(p)
    a = p.parse_args()

    ins, outs = read_video(a.input), read_video(a.output, a.max_frames)
    print(f"[facefix] {len(ins)} input / {len(outs)} output frames", flush=True)
    if a.seq_json:
        seq = json.load(open(a.seq_json))["source_seq"][:len(outs)]
        idx = np.clip(np.array(seq) - 1, 0, len(ins) - 1)
        a.align = "source_seq"
    else:
        idx = align_dtw(ins, outs) if a.align == "dtw" else align_linear(len(ins), len(outs))
    fp = make_pass(a)
    h, w = outs[0].shape[:2]
    wr = Writer(a.out, w, h, a.fps)
    sbs = Writer(a.sbs, w * 3, h, a.fps) if a.sbs else None
    zoom = Writer(a.zoom, 3 * 320, 320, a.fps) if a.zoom else None
    zoom_last = np.zeros((320, 960, 3), np.uint8)
    infos, T = [], []
    from concurrent.futures import ThreadPoolExecutor
    pool, pool_in = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    if a.pipeline:      # 3 stages like live: input analysis (capture time) | output analysis | GPU render
        pre = [pool_in.submit(fp.analyze_input, ins[i]) for i in idx]   # sequential in its own thread
        stage_b = lambda k: fp.analyze(ins[idx[k]], outs[k], pre[k].result())
        nxt = pool.submit(stage_b, 0)
    t_wall = None
    for j, O in enumerate(outs):
        I = ins[idx[j]]
        if j == 10:
            t_wall = time.perf_counter()
        t = time.perf_counter()
        if a.pipeline:
            st = nxt.result()
            st["info"]["n"] = j
            if j + 1 < len(outs):
                nxt = pool.submit(stage_b, j + 1)
            res, info = fp.render(st)
        else:
            res, info = fp(I, O)
        T.append(1000 * (time.perf_counter() - t))
        infos.append(info)
        wr.write(res)
        shown = res
        if a.debug_mask and getattr(fp, "last_mask", None) is not None:
            x0, y0, m = fp.last_mask
            shown = res.copy()
            for lvl, col in ((0.05, (0, 200, 255)), (0.5, (0, 255, 0))):
                cs, _ = cv2.findContours((m > lvl).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(shown, [c + (x0, y0) for c in cs], -1, col, 1, cv2.LINE_AA)
        if sbs:
            sbs.write(np.hstack([I, O, overlay(shown, info)]))
        if zoom:
            if fp.rect is not None:
                zr = fp.rect.copy()
                zr[1] += 0.07 * zr[2]                          # centre on nose/mouth, not forehead
                zr[2] *= 0.6
                Mz = rect_to_M(zr, 320)
                zi = fp.rin.copy() if fp.rin is not None else zr.copy()
                if fp.rin is not None:
                    zi[1] += 0.07 * zi[2]
                    zi[2] *= 0.6
                zoom_last = np.hstack([cv2.warpAffine(I, rect_to_M(zi, 320), (320, 320)), cv2.warpAffine(O, Mz, (320, 320)),
                                       overlay(cv2.warpAffine(shown, Mz, (320, 320)), info)])
            zoom.write(zoom_last)
        if j % 100 == 0:
            print(f"[facefix] {j}/{len(outs)} {info['src']} w={info['weight']} {np.mean(T[-100:]):.1f} ms/frame",
                  flush=True)
    wall_fps = (len(outs) - 10) / (time.perf_counter() - t_wall) if t_wall else None
    for x in (wr, sbs, zoom):
        if x:
            x.close()
    skip = 10
    ms = np.array([i["ms"] for i in infos[skip:]])
    active = np.array([i["weight"] > 0.01 for i in infos[skip:]])
    med = lambda v: round(float(np.median(v)), 1) if len(v) else None
    res = {"frames": len(outs), "align": a.align,
           "ms_median_active": {"landmarks": med(ms[active, 0]), "retarget": med(ms[active, 1]),
                                "restore": med(ms[active, 2]), "total": med(np.array(T[skip:])[active])},
           "active_frac": round(float(active.mean()), 3),
           "mean_weight": round(float(np.mean([i["weight"] for i in infos])), 3),
           "src": {k: sum(i["src"] == k for i in infos) for k in ("out", "in", "none")}}
    if hasattr(fp, "trk_mask") and a.stabilize:
        res["track_mask"], res["track_rect"] = fp.trk_mask.stats, fp.trk_rect.stats
        if hasattr(fp, "flow"):
            res["track_mask"] = fp.flow.stats
        if hasattr(fp, "sam"):
            res["track_mask"] = fp.sam.stats
            res["ms_sam_median"] = med([i["ms_sam"] for i in infos[skip:] if "ms_sam" in i])
    ed = [i["eye_diff"] for i in infos if "eye_diff" in i]
    if ed:
        res["eye_region_max_abs_diff"] = int(max(ed))
    ids = [(i["id_joyai"], i["id_refined"]) for i in infos if "id_refined" in i]
    if ids:
        res["identity_cos_joyai_refined"] = [round(float(np.mean([x[0] for x in ids])), 3),
                                             round(float(np.mean([x[1] for x in ids])), 3)]
    res["wall_fps_incl_io"] = round(wall_fps, 1) if wall_fps else None
    res["fps_active"] = round(1000 / res["ms_median_active"]["total"], 1) if res["ms_median_active"]["total"] else None
    print("[facefix] RESULT", json.dumps(res), flush=True)
    if a.stats:
        res.update(args=vars(a), idx=idx.tolist(), frames_info=[{k: v for k, v in i.items() if k != "ms"} | {
            "ms": [round(x, 1) for x in i["ms"]]} for i in infos])
        Path(a.stats).write_text(json.dumps(res, default=lambda x: None))


if __name__ == "__main__":
    main()
