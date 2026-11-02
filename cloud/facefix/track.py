"""Mouth-region trackers that bind to the GENERATED face's pixels (G-Star), instead of per-frame MediaPipe.

FlowTracker (--track flow): the mouth geometry [mx, my, half_width, y_top, y_bottom, angle] is propagated frame to
frame with dense optical flow (OpenCV DIS) on a downscaled ROI around it: a similarity transform is fitted (RANSAC)
to the flow of points around the mouth (lip motion inside is mostly rejected as outliers) and applied to the
geometry. MediaPipe measurements on the generated face only re-anchor it slowly when they agree (fixes drift); a
disagreeing measurement is ignored unless it persists while the flow is unreliable. ~2-3 ms per frame on the CPU.
"""
import cv2
import numpy as np


def geom_box(g):
    """Centre and axes of the geometry's box (centre between y_top and y_bottom)."""
    mx, my, hw, yt, yb, ang = g
    ux = np.array([np.cos(ang), np.sin(ang)])
    uy = np.array([-ux[1], ux[0]])
    return np.array([mx, my]) + uy * (yt + yb) / 2, hw, (yb - yt) / 2, ux, uy


class FlowTracker:
    def __init__(self, roi_px=160, anchor=0.08, confirm=2, persist=12, min_inliers=0.35):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.roi_px, self.anchor, self.confirm, self.persist, self.min_inl = roi_px, anchor, confirm, persist, min_inliers
        self.g, self.prev, self.cand, self.disagree, self.bad = None, None, [], 0, 0
        self.stats = {"flow_ok": 0, "flow_bad": 0, "anchored": 0, "ignored_meas": 0, "reinit": 0, "missing": 0}
        self.conf = 0.0

    def agree(self, m, g):
        return np.hypot(m[0] - g[0], m[1] - g[1]) < 0.45 * g[2] and abs(m[2] / g[2] - 1) < 0.3

    def flow_step(self, gray):
        """Similarity transform prev -> current around the mouth, or None if unreliable."""
        c, ax, ay, ux, uy = geom_box(self.g)
        S = 2.6 * max(ax, ay)                                # ROI side in frame px (mouth + cheeks + chin)
        k = self.roi_px / S
        M = np.array([[k, 0, self.roi_px / 2 - k * c[0]], [0, k, self.roi_px / 2 - k * c[1]]], np.float32)
        a = cv2.warpAffine(self.prev, M, (self.roi_px, self.roi_px), flags=cv2.INTER_AREA)
        b = cv2.warpAffine(gray, M, (self.roi_px, self.roi_px), flags=cv2.INTER_AREA)
        fl = self.dis.calc(a, b, None)
        n = 14                                               # grid points over the box, expanded 1.25x
        uu, vv = np.meshgrid(np.linspace(-1.25, 1.25, n), np.linspace(-1.25, 1.25, n))
        pts = (c + (uu.ravel() * ax)[:, None] * ux + (vv.ravel() * ay)[:, None] * uy)
        q = pts @ M[:, :2].T + M[:, 2]
        ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < self.roi_px - 1) & (q[:, 1] < self.roi_px - 1)
        q = q[ok]
        if len(q) < 20:
            return None, 0.0
        d = fl[q[:, 1].astype(int), q[:, 0].astype(int)]
        T, inl = cv2.estimateAffinePartial2D(q.astype(np.float32), (q + d).astype(np.float32),
                                             method=cv2.RANSAC, ransacReprojThreshold=1.0, maxIters=200)
        if T is None:
            return None, 0.0
        ratio = float(inl.mean())
        # back to frame coordinates: F = M^-1 * T * M
        M3, T3 = np.vstack([M, [0, 0, 1]]), np.vstack([T, [0, 0, 1]])
        F = (np.linalg.inv(M3) @ T3 @ M3)[:2]
        return F, ratio

    def apply(self, F):
        mx, my, hw, yt, yb, ang = self.g
        p = F[:, :2] @ np.array([mx, my]) + F[:, 2]
        s = float(np.sqrt(abs(np.linalg.det(F[:, :2]))))
        rot = float(np.arctan2(F[1, 0], F[0, 0]))
        s = float(np.clip(s, 0.9, 1.1))                     # per-frame scale change is small
        return np.array([p[0], p[1], hw * s, yt * s, yb * s, ang + rot])

    def update(self, frame, meas):
        """frame: BGR output frame; meas: MediaPipe mouth geometry on it (or None). -> (geometry or None, ok)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if meas is None:
            self.stats["missing"] += 1
        if self.g is None:                                   # wait for consistent measurements to lock
            if meas is not None:
                self.cand = [x for x in self.cand if self.agree(x, meas)] + [meas]
                if len(self.cand) >= self.confirm:
                    self.g, self.cand = np.mean(self.cand, 0), []
            self.prev = gray
            return self.g, self.g is not None
        F, ratio = self.flow_step(gray)
        self.prev = gray
        self.conf = 0.7 * self.conf + 0.3 * ratio
        flow_ok = F is not None and ratio >= self.min_inl
        if flow_ok:
            self.g = self.apply(F)
            self.stats["flow_ok"] += 1
        else:
            self.stats["flow_bad"] += 1
        if meas is not None:
            if self.agree(meas, self.g):
                d = meas - self.g
                d[-1] = (d[-1] + np.pi) % (2 * np.pi) - np.pi
                self.g = self.g + (self.anchor if flow_ok else 0.3) * d     # re-anchor (drift)
                self.disagree, self.cand = 0, []
                self.stats["anchored"] += 1
            else:
                self.stats["ignored_meas"] += 1
                self.cand = [x for x in self.cand if self.agree(x, meas)] + [meas]
                self.disagree += 1
                if len(self.cand) >= self.persist and (not flow_ok or self.disagree >= 3 * self.persist):
                    self.g, self.cand, self.disagree = np.mean(self.cand[-3:], 0), [], 0   # lost: re-lock
                    self.stats["reinit"] += 1
        ok = flow_ok or (meas is not None and self.agree(meas, self.g))
        self.bad = 0 if ok else self.bad + 1
        return self.g, ok


class SamTracker:
    """SAM 2.1 tiny in streaming mode (camera-predictor fork, third_party/sam2rt) at 512 px: segments G-Star's
    actual mouth (the opening of his grin) frame to frame, prompted once with a box around MediaPipe's lip
    landmarks on a confident frame; re-prompted when the mask vanishes, its area jumps, or it leaves the MediaPipe
    mouth estimate for several frames. The paste mask is the segment grown around the lips and down for the jaw
    drop, feathered. ~8 ms per frame on the RTX 5080 Laptop."""

    def __init__(self, image_size=512):
        import sys
        from pathlib import Path
        import torch
        d = Path(__file__).resolve().parent
        sys.path.insert(0, str(d / "third_party" / "sam2rt"))
        from sam2.build_sam import build_sam2_camera_predictor
        cfg = "sam2.1_hiera_t_512.yaml" if image_size == 512 else "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.torch = torch
        self.pred = build_sam2_camera_predictor(cfg, str(d / "weights/sam2/sam2.1_hiera_tiny.pt"), device="cuda")
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]

        def prep(img, image_size=1024, **_):           # the fork resizes + normalises in float64 on the CPU (15 ms)
            h, w = img.shape[:2]
            t = torch.from_numpy(img).cuda().permute(2, 0, 1)[None].float() / 255.0
            t = torch.nn.functional.interpolate(t, (image_size, image_size), mode="bilinear", antialias=True)[0]
            return (t - mean) / std, w, h
        self.pred.perpare_data = prep
        self.started, self.area, self.off, self.bad = False, None, 0, 0
        self.stats = {"prompts": 0, "lost": 0, "ok": 0, "drift_reprompt": 0}
        self.seg = None

    def prompt(self, frame, meas_pts):
        L = meas_pts[[61, 291, 0, 17, 13, 14, 78, 308]]
        lo, hi = L.min(0), L.max(0)
        d = (hi - lo) * np.array([0.08, 0.25])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            self.pred.load_first_frame(rgb)
            _, _, logits = self.pred.add_new_prompt(frame_idx=0, obj_id=1, bbox=np.array([lo - d, hi + d]))
        self.started, self.area, self.off = True, None, 0
        self.stats["prompts"] += 1
        return (logits[0, 0] > 0).cpu().numpy()

    def clean(self, seg, meas_pts):
        """Keep only the mouth: SAM sometimes adds fragments (cap, forehead). Keep the connected component that
        overlaps the previous mouth (else the one nearest to it / to MediaPipe's mouth) and nothing above the
        eyes-to-mouth midline."""
        seg = seg.astype(np.uint8)
        if meas_pts is not None:
            eye_y = meas_pts[[33, 133, 362, 263]][:, 1].mean()
            m_y = meas_pts[[13, 14]][:, 1].mean()
            cut = int(eye_y + 0.45 * (m_y - eye_y))
            if 0 < cut < seg.shape[0]:
                seg[:cut] = 0
        n, lab, st, cen = cv2.connectedComponentsWithStats(seg, connectivity=8)
        if n <= 2:
            return seg.astype(bool)
        if self.seg is not None:
            prev = cv2.dilate(self.seg.astype(np.uint8), np.ones((9, 9), np.uint8))
            ov = np.bincount(lab[prev > 0].ravel(), minlength=n)
            ov[0] = 0
            if ov.max() > 0:
                return lab == int(np.argmax(ov))
            ref = np.array(np.nonzero(self.seg)[::-1]).mean(1)
        elif meas_pts is not None:
            ref = meas_pts[[61, 291, 13, 14]].mean(0)
        else:
            return lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
        k = 1 + int(np.argmin(np.linalg.norm(cen[1:] - ref, axis=1)))
        return lab == k

    def update(self, frame, meas_pts):
        """-> (segment mask HxW bool or None, ok)."""
        if not self.started:
            if meas_pts is None:
                return None, False
            seg = self.prompt(frame, meas_pts)
        else:
            with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                _, logits = self.pred.track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            seg = (logits[0, 0] > 0).cpu().numpy()
        seg = self.clean(seg, meas_pts)
        n = int(seg.sum())
        ok = n > 30
        if ok and self.area is not None and not (0.4 < n / self.area < 2.5):
            ok = False                                   # sudden area jump: probably switched to something else
        if ok and meas_pts is not None:                  # far from MediaPipe's mouth for a while: re-prompt
            ys, xs = np.nonzero(seg)
            c = np.array([xs.mean(), ys.mean()])
            mc = meas_pts[[61, 291, 13, 14]].mean(0)
            mw = np.linalg.norm(meas_pts[291] - meas_pts[61])
            self.off = self.off + 1 if np.linalg.norm(c - mc) > 0.6 * mw else 0
            if self.off >= 8:
                seg, ok = self.prompt(frame, meas_pts), True
                self.stats["drift_reprompt"] += 1
                n = int(seg.sum())
        if ok:
            self.area = n if self.area is None else 0.9 * self.area + 0.1 * n
            self.seg, self.bad = seg, 0
            self.stats["ok"] += 1
        else:
            self.stats["lost"] += 1
            self.bad += 1
            if self.bad >= 5 and meas_pts is not None:   # lost for a while: re-prompt on the current frame
                seg = self.prompt(frame, meas_pts)
                if seg.sum() > 30:
                    self.seg, self.bad, ok = seg, 0, True
        return self.seg, ok


def paste_mask_from_seg(seg, feather=5.0, grow=0.15, grow_up=0.1, drop=0.4, w=None, h=None):
    """Paste mask from a mouth segment: grown by `grow` x mouth width around the lips, and down by `drop` x width
    for the jaw drop of the re-rendered mouth, feathered. Returns (x0, y0, x1, y1, mask) or None."""
    ys, xs = np.nonzero(seg)
    if len(xs) < 30:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mw = max(x1 - x0, 8)
    g, gy, dr = int(grow * mw), int(grow_up * mw), int(drop * mw)
    H, W = seg.shape
    X0, Y0 = max(x0 - g - int(2 * feather), 0), max(y0 - gy - int(2 * feather), 0)
    X1, Y1 = min(x1 + g + int(2 * feather) + 1, W), min(y1 + gy + dr + int(2 * feather) + 1, H)
    roi = seg[Y0:Y1, X0:X1].astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * gy + 1))   # wide, short: lips not cheeks
    D = cv2.dilate(roi, k)
    m = D.copy()
    for s in range(1, dr + 1, max(1, dr // 6)):          # sweep down: room for the opened jaw
        if s < m.shape[0]:
            m[s:] |= D[:-s]
    m = m.astype(np.float32)
    sh = max(1, int(1.5 * feather))
    m = cv2.erode(m, np.ones((sh, sh), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), feather / 2)
    m[m < 1e-3] = 0
    return X0, Y0, X1, Y1, m
