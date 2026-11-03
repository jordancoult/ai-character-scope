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
DEFAULT_GAINS = {
    "rel": {"lip": 1.3, "eye": 1.0, "brow": -1.0, "brow2": 0.0, "rest": 0.0},
    "abs": {"lip": 1.0, "eye": 1.0, "brow": 1.0, "brow2": 1.0, "rest": 1.0},
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

    def _detect(self, bgr):
        self.detections += 1
        self.detected_now = True
        # the wrapper launches on the stream it was built on (DET_STREAM): run it there; fence only that stream
        with DET_LOCK, torch.cuda.stream(DET_STREAM):
            faces = self.det.predict(bgr)
            if len(faces) == 0:  # cartoon faces (G-Star close-up) score lower: retry at a lower threshold
                th = self.det.det_thresh
                self.det.det_thresh = 0.3
                try:
                    faces = self.det.predict(bgr)
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

    def motion(self, bgr, lmk):
        """256 px LivePortrait crop -> dict of GPU tensors (+ M_c2o of the equivalent 512 crop, for paste-back)."""
        M256, _ = _estimate_similar_transform_from_pts(lmk, dsize=256, scale=2.3, vy_ratio=-0.125)
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


class Driver:
    """Webcam side: call `analyze(I)` when the frame is captured/sent; keep the result with the frame's seq."""

    def __init__(self, detector):
        self.face = Face(detector, retry=3)  # parallel thread: back off a little so it doesn't contend

    def analyze(self, bgr):
        with torch.cuda.stream(self.face.stream):  # own stream: overlaps with the render of the previous frame
            lmk = self.face.landmarks(bgr)
            if lmk is None:
                return None
            m = self.face.motion(bgr, lmk)
            m["R"] = rot_matrix(m["pitch"], m["yaw"], m["roll"])
            self.face.stream.synchronize()
        return {"exp": m["exp"], "lmk": lmk, "R": m["R"], "t": m["t"], "scale": m["scale"]}


class LPPass:
    def __init__(self, mode="rel", gains=None, base_ema=0.08,
                 neutral_ema=0.01, fade=0.25, drv_smooth=0.0, yaw_gate=(30.0, 45.0), pitch_gate=(20.0, 35.0),
                 pose=False):
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
        self.yaw_gate, self.pitch_gate = yaw_gate, pitch_gate
        # pose=True (still-image sources): also transfer the driver's head rotation / shift / scale, relative to the
        # driver's first frame (LivePortrait image animation). Off for JoyAI output, which already moves the head.
        self.pose, self.drv0 = pose, None
        self.base_out = self.neutral_in = self.exp_in = None
        self.pose_s = None
        self.w = 0.0
        self.timing = {}

        self._warmup()

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
            d = Driver(self.detector)
            for _ in range(3):
                self.render(img, d.analyze(img))
            torch.cuda.synchronize()
            self.src.lmk, self.src.lost, self.src.detections = None, 0, 0
            self.base_out = self.neutral_in = self.exp_in = self.pose_s = self.drv0 = None
            self.w, self.timing = 0.0, {}

    def driver(self):
        d = Driver(self.detector)
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
            target = min(ramp(yaw, self.yaw_gate), ramp(pitch, self.pitch_gate))
        self.w = min(target, self.w + self.fade) if target > self.w else max(target, self.w - self.fade)
        ok = ok and self.w > 0
        info = {"ok": ok, "w": self.w}
        if not ok:  # no face on one side: pass JoyAI through
            return out_bgr, info
        t0 = self._t("src", t0, profile)
        # slow anchors (GPU)
        e_in = drv["exp"]
        if self.drv_smooth > 0 and self.exp_in is not None:
            e_in = self.exp_in + (1 - self.drv_smooth) * (e_in - self.exp_in)
        self.exp_in = e_in
        self.base_out = s["exp"].clone() if self.base_out is None else self.base_out.lerp_(s["exp"], self.base_ema)
        self.neutral_in = e_in.clone() if self.neutral_in is None else self.neutral_in.lerp_(e_in, self.neutral_ema)
        W, Wm = self.W, self.Wm
        if self.mode == "abs":  # blend towards the driver's expression
            exp = s["exp"] + W * (e_in - s["exp"])
        else:  # character's resting expression + scaled driver motion; gain-0 regions keep JoyAI's own expression
            exp = s["exp"] * (1 - Wm) + Wm * (self.base_out + W * (e_in - self.neutral_in))
        R = rot_matrix(s["pitch"], s["yaw"], s["roll"])
        sc, t, kp = s["scale"][..., None], s["t"], s["kp"]
        x_s = (kp @ R + s["exp"]) * sc
        x_s[..., :2] += t[:, None, :2]
        if self.pose:
            if self.drv0 is None:
                self.drv0 = {k: drv[k].clone() for k in ("R", "t", "scale")}
            R_d = drv["R"] @ self.drv0["R"].transpose(1, 2) @ R
            sc_d = sc * (drv["scale"] / self.drv0["scale"])[..., None]
            t_d = t + (drv["t"] - self.drv0["t"])
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
