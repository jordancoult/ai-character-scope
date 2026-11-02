"""Face-swap method for human references: the REAL input face is swapped to the reference identity (HyperSwap 256
or InSwapper 128, ONNX converted to torch), optionally restored (GFPGAN / CodeFormer), mapped onto the generated
face with a similarity transform fitted on stable landmarks (JoyAI keeps the pose), colour-matched (LAB) and
pasted with a feathered inner-face mask. The mouth (teeth, tongue, jaw) is real video, not a warp.

Same interface as facefix.FacePass (analyze_input / analyze / render); gating/fades are inherited.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from facefix import WEIGHTS, FacePass, Landmarker, five_points

ARC112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655],
                   [70.7299, 92.2041]], np.float32)
ARC128 = ARC112 + np.array([8.0, 0.0], np.float32)       # insightface "arcface_128" template
STABLE = [33, 133, 362, 263, 168, 6, 197, 1, 70, 300, 105, 334]   # eyes, nose bridge, brows: input -> output fit
INNER = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300,          # brows
         454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]


def load_onnx(path):
    from onnx2torch import convert
    return convert(str(path)).cuda().eval()


class SwapPass(FacePass):
    def __init__(self, a):
        a.region = "none"                                   # no LivePortrait
        if a.fade_in == 0.25 and a.fade_out == 0.4:         # faster fades: a half-blended mouth shows twice
            a.fade_in, a.fade_out = 0.4, 0.6
        super().__init__(a)
        if not a.ref:
            raise SystemExit("--method swap needs --ref (the identity image)")
        torch.backends.cudnn.benchmark = True
        d = WEIGHTS / "swap"
        self.kind = a.swap_model
        self.size = 256 if self.kind == "hyperswap" else 128
        self.net = load_onnx(d / ("hyperswap_1a_256.onnx" if self.kind == "hyperswap" else "inswapper_128_fp16.onnx"))
        # (torch.compile and CUDA-graph capture both fail on the onnx2torch graph: host-side shape ops)
        arc = load_onnx(d / "arcface_w600k_r50.onnx")
        ref = cv2.imread(a.ref)
        f = Landmarker(1).detect(ref)
        if not f:
            raise SystemExit(f"no face found in {a.ref}")
        f = max(f, key=lambda x: x.size)
        M, _ = cv2.estimateAffinePartial2D(five_points(f.pts), ARC112, method=cv2.LMEDS)
        x = cv2.cvtColor(cv2.warpAffine(ref, M, (112, 112)), cv2.COLOR_BGR2RGB).astype(np.float32)
        x = torch.from_numpy((x - 127.5) / 127.5).permute(2, 0, 1)[None].cuda()
        with torch.no_grad():
            emb = arc(x)[0].float().cpu().numpy()
        self.ref_emb = emb / np.linalg.norm(emb)
        if self.kind == "inswapper":
            lat = self.ref_emb @ np.load(d / "inswapper_emap.npy")
            lat = lat / np.linalg.norm(lat)
        else:
            lat = self.ref_emb
        self.latent = torch.from_numpy(lat.astype(np.float32))[None].cuda()
        self.arc = arc
        self.tmpl = ARC128 * (self.size / 128.0)
        m = np.zeros((self.size, self.size), np.float32)      # valid area of the swapped crop, feathered
        b = self.size // 16
        m[b:-b, b:-b] = 1
        self.crop_valid = cv2.GaussianBlur(m, (0, 0), self.size / 32)

    def net_call(self, x):
        return self.net(self.latent, x * 2 - 1) if self.kind == "hyperswap" else (self.net(x, self.latent), None)

    @torch.no_grad()
    def swap(self, crop):
        x = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).cuda().permute(2, 0, 1)[None].float() / 255.0
        y, mask = self.net_call(x)
        if self.kind == "hyperswap":
            y = (y + 1) / 2
            mask = mask[0, 0].clamp(0, 1).float().cpu().numpy().copy()
        y = (y[0].clamp(0, 1) * 255).permute(1, 2, 0).byte().cpu().numpy().copy()
        return cv2.cvtColor(y, cv2.COLOR_RGB2BGR), mask

    @torch.no_grad()
    def identity(self, face_bgr, pts):
        """Cosine similarity of a face (frame + landmarks) to the reference identity (ArcFace w600k)."""
        M, _ = cv2.estimateAffinePartial2D(five_points(pts), ARC112, method=cv2.LMEDS)
        x = cv2.cvtColor(cv2.warpAffine(face_bgr, M, (112, 112)), cv2.COLOR_BGR2RGB).astype(np.float32)
        e = self.arc(torch.from_numpy((x - 127.5) / 127.5).permute(2, 0, 1)[None].cuda())[0].float().cpu().numpy()
        return float(e @ self.ref_emb / np.linalg.norm(e))

    def render(self, st):
        import time
        a, O, info = self.a, st["O"], st["info"]
        t1 = time.perf_counter()
        wgt, I, pi, po = st["weight"], st["I"], st["fi_pts"], st["fo_pts"]
        res = O
        if wgt > 0.01 and pi is not None:
            h, w = O.shape[:2]
            S = self.size
            M, _ = cv2.estimateAffinePartial2D(five_points(pi), self.tmpl, method=cv2.LMEDS)    # input -> crop
            crop = cv2.warpAffine(I, M, (S, S), borderMode=cv2.BORDER_REPLICATE)
            face, smask = self.swap(crop)
            valid = self.crop_valid if smask is None else self.crop_valid * cv2.GaussianBlur(smask, (0, 0), 2)
            Ci = cv2.invertAffineTransform(M)                                                   # crop -> input
            if po is not None:                                                                  # input -> output
                T, _ = cv2.estimateAffinePartial2D(pi[STABLE], po[STABLE], method=cv2.LMEDS)
            else:
                T = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
            T3 = np.vstack([T, [0, 0, 1]])
            C = (T3 @ np.vstack([Ci, [0, 0, 1]]))[:2]                                           # crop -> output
            pts_o = pi @ T[:, :2].T + T[:, 2]                    # the real face's landmarks on the output
            # paste mask: inner face of the real face (follows the real jaw), feathered, in output coords
            x0, y0 = np.maximum(np.floor(pts_o.min(0) - 0.25 * np.ptp(pts_o, 0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(pts_o.max(0) + 0.25 * np.ptp(pts_o, 0)).astype(int), [w, h])
            if x1 - x0 > 8 and y1 - y0 > 8:
                fs = float(np.ptp(pts_o, 0).max())
                m = np.zeros((y1 - y0, x1 - x0), np.float32)
                hull = cv2.convexHull((pts_o[INNER] - (x0, y0)).astype(np.int32))
                if a.swap_mask == "lower":
                    cut = int(pts_o[[1, 4]].mean(0)[1] - y0 - 0.05 * fs)
                    cv2.fillConvexPoly(m, hull, 1.0)
                    m[:max(cut, 0)] = 0
                else:
                    cv2.fillConvexPoly(m, hull, 1.0)
                k = max(3, int(fs * 0.06)) | 1
                m = cv2.erode(m, np.ones((k, k), np.uint8))
                m = cv2.GaussianBlur(m, (0, 0), fs * 0.04 + 1)
                Cr = C.copy()
                Cr[:, 2] -= (x0, y0)
                face_o = cv2.warpAffine(face, Cr, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
                m *= cv2.warpAffine(valid, Cr, (x1 - x0, y1 - y0))
                roi = O[y0:y1, x0:x1]
                face_o = lab_match(face_o, roi, m)
                if self.rs is not None:                     # restore on the composited face region
                    tmp = O.copy()
                    tmp[y0:y1, x0:x1] = (face_o * m[..., None] + roi * (1 - m[..., None])).astype(np.uint8)
                    tmp = self.rs(tmp, pts_o, a.restore_blend)
                    face_o = tmp[y0:y1, x0:x1].astype(np.float32)
                m = (m * wgt)[..., None]
                res = O.copy()
                res[y0:y1, x0:x1] = np.clip(face_o * m + roi * (1 - m), 0, 255).astype(np.uint8)
        t2 = time.perf_counter()
        info["ms"] = (info["ms_cpu"], 1000 * (t2 - t1), 0.0)
        if getattr(a, "eval_id", False) and po is not None and info.get("n", 0) % 5 == 0:
            info["id_joyai"] = round(self.identity(O, po), 3)
            info["id_refined"] = round(self.identity(res, po), 3)
        return res, info


def lab_match(src, ref, m):
    """Match mean/std of src to ref in LAB inside mask m (lighting + skin tone of the generated face). Stats on a
    4x downsampled ROI (cheap); the transform is applied at full size."""
    if m.sum() < 10:
        return src.astype(np.float32)
    sm = (slice(None, None, 4), slice(None, None, 4))
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    r4 = cv2.cvtColor(np.ascontiguousarray(ref[sm]), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    s4 = s[sm].reshape(-1, 3).astype(np.float32)
    w = m[sm].reshape(-1, 1)
    w = w / (w.sum() + 1e-6)
    ms, mr = (s4 * w).sum(0), (r4 * w).sum(0)
    ss = np.sqrt((((s4 - ms) ** 2) * w).sum(0)) + 1e-3
    sr = np.sqrt((((r4 - mr) ** 2) * w).sum(0)) + 1e-3
    gain = np.clip(sr / ss, 0.7, 1.4)
    gain[0] = np.clip(gain[0], 0.85, 1.15)                  # keep the swap's own shading (teeth, mouth interior)
    lut = np.stack([np.clip((np.arange(256) - ms[c]) * gain[c] + mr[c], 0, 255) for c in range(3)], 1).astype(np.uint8)
    out = np.stack([lut[s[..., c], c] for c in range(3)], -1)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR).astype(np.float32)
