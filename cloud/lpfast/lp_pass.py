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
    exp[k] = base_out[k] + gain * (exp_in[k] - neutral_in[k])   for k in the lip / eye keypoints
  base_out = slow EMA of JoyAI's own expression (removes its chunk-periodic mouth wobble), neutral_in = slow EMA of
  the driver's (their resting face). Mode "abs": exp[k] = exp_in[k] (human mouth shape).
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
LIP_IDX = [6, 12, 14, 17, 19, 20]
EYE_IDX = [11, 13, 15, 16, 18]
IDX66 = None
_UNSET = object()
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

    def __init__(self, detector, appearance=False, max_jump=0.35):
        self.stream = torch.cuda.Stream()
        self.lm = Engine(ENG / "landmark.engine", self.stream)
        self.me = Engine(ENG / "motion_extractor.engine", self.stream)
        self.app = Engine(ENG / "appearance_feature_extractor.engine", self.stream) if appearance else None
        self.det, self.max_jump = detector, max_jump
        self.lmk = None
        self.detections = 0

    def _detect(self, bgr):
        self.detections += 1
        # the wrapper launches on the default stream: run it there, fenced on both sides (it is rare)
        with DET_LOCK, torch.cuda.stream(torch.cuda.default_stream()):
            torch.cuda.synchronize()
            faces = self.det.predict(bgr)
            torch.cuda.synchronize()
        if len(faces) == 0:
            return None
        return self._landmarks(bgr, faces[0])

    def _landmarks(self, bgr, pts):
        M, _ = _estimate_similar_transform_from_pts(pts, dsize=224, scale=1.5, vy_ratio=-0.1)
        crop = cv2.warpAffine(bgr, M, (224, 224), flags=cv2.INTER_LINEAR)
        x = torch.from_numpy(crop[..., ::-1].copy()).cuda().permute(2, 0, 1)[None].float().div_(255)
        out = self.lm.run(input=x)["856"].float().cpu().numpy().reshape(-1, 2) * 224  # 203 points in the crop
        Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]]))
        return out @ Mi[:2, :2].T + Mi[:2, 2]

    def landmarks(self, bgr):
        """Track from the previous landmarks; re-detect only when lost or when the track jumps."""
        if self.lmk is None:
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
        return lmk

    def motion(self, bgr, lmk):
        """256 px LivePortrait crop -> dict of GPU tensors (+ M_c2o of the equivalent 512 crop, for paste-back)."""
        M256, _ = _estimate_similar_transform_from_pts(lmk, dsize=256, scale=2.3, vy_ratio=-0.125)
        crop = cv2.warpAffine(bgr, M256, (256, 256), flags=cv2.INTER_LINEAR)
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
        self.face = Face(detector)

    def analyze(self, bgr):
        with torch.cuda.stream(self.face.stream):  # own stream: overlaps with the render of the previous frame
            lmk = self.face.landmarks(bgr)
            if lmk is None:
                return None
            m = self.face.motion(bgr, lmk)
            self.face.stream.synchronize()
        return {"exp": m["exp"], "lmk": lmk}


class LPPass:
    def __init__(self, mode="rel", regions=("lip", "eyes"), gain=1.0, eye_gain=1.0, base_ema=0.08,
                 neutral_ema=0.01, fade=0.25, drv_smooth=0.0):
        m = flp_compat.load_models()  # only RetinaFace (+106-pt) detection is used through the wrappers
        self.detector = m["face_analysis"]
        self.src = Face(self.detector, appearance=True)
        self.warp = Engine(ENG / "warping_fp16.engine", self.src.stream)
        self.stitch = Engine(ENG / "stitching.engine", self.src.stream)
        mt = cv2.imread(str(flp_compat.FLP / "assets" / "mask_template.png"), cv2.IMREAD_GRAYSCALE)
        self.mask = torch.from_numpy(mt.astype(np.float32) / 255).cuda()[None]  # 1x512x512
        self.mode, self.gain, self.eye_gain = mode, gain, eye_gain
        self.idx = [(LIP_IDX, gain)] if "lip" in regions else []
        self.idx += [(EYE_IDX, eye_gain)] if "eyes" in regions else []
        self.base_ema, self.neutral_ema, self.fade, self.drv_smooth = base_ema, neutral_ema, fade, drv_smooth
        self.base_out = self.neutral_in = self.exp_in = None
        self.w = 0.0
        self.timing = {}

    def driver(self):
        return Driver(self.detector)

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
        self.w = min(1.0, self.w + self.fade) if ok else max(0.0, self.w - self.fade)
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
        exp = s["exp"].clone()
        for idx, g in self.idx:
            exp[:, idx] = e_in[:, idx] if self.mode == "abs" else \
                self.base_out[:, idx] + g * (e_in[:, idx] - self.neutral_in[:, idx])
        R = rot_matrix(s["pitch"], s["yaw"], s["roll"])
        sc, t, kp = s["scale"][..., None], s["t"], s["kp"]
        x_s = (kp @ R + s["exp"]) * sc
        x_s[..., :2] += t[:, None, :2]
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
