"""MediaPipe face / pose measurements: presence, face box (for framing), lip gap (for mouth sync).

Lip-gap logic follows cloud/client/mouth_sync.py: inner-lip gap / face height (landmarks 13/14, 10/152) and the
Pearson correlation of the input and output series (best over a small lag window).
"""

from pathlib import Path

import cv2
import numpy as np

from common import REPO, log

MODELS = REPO / "cloud" / "client" / "models"
FACE_MODEL = MODELS / "face_landmarker.task"
POSE_MODEL = MODELS / "pose_landmarker_lite.task"
URLS = {
    FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/"
                "face_landmarker.task",
    POSE_MODEL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/"
                "pose_landmarker_lite.task",
}


def _ensure(model):
    if not model.exists():
        import requests
        model.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", model.name)
        r = requests.get(URLS[model], timeout=120)
        r.raise_for_status()
        model.write_bytes(r.content)
    return str(model)


class Vision:
    def __init__(self, use_pose=True):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure(FACE_MODEL)), output_face_blendshapes=True,
            num_faces=1, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
        self.pose = None
        if use_pose:
            try:
                self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_ensure(POSE_MODEL)), num_poses=1,
                    min_pose_detection_confidence=0.4))
            except Exception as e:  # pose is optional
                log.warning("pose model unavailable (%s); person check uses face only", e)

    def _img(self, frame):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def landmarks(self, frame, roi=None):
        """Face landmarks in full-frame pixels (N x 2) + blendshapes, or (None, None).

        roi = (x0, y0, side) in pixels: detect on that square crop upscaled to 256 px (small faces in full-body
        shots are missed by the full-frame detector).
        """
        H, W = frame.shape[:2]
        img, ox, oy, sc = frame, 0, 0, 1.0
        if roi is not None:
            x0, y0, side = (int(round(v)) for v in roi)
            x0, y0 = max(0, min(W - 1, x0)), max(0, min(H - 1, y0))
            crop = frame[y0:y0 + side, x0:x0 + side]
            if crop.shape[0] < 16 or crop.shape[1] < 16:
                return None, None
            sc = 256.0 / max(crop.shape[:2])
            img, ox, oy = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC), x0, y0
        r = self.face.detect(self._img(img))
        if not r.face_landmarks:
            return None, None
        ih, iw = img.shape[:2]
        pts = np.array([(p.x * iw / sc + ox, p.y * ih / sc + oy) for p in r.face_landmarks[0]])
        return pts, (r.face_blendshapes[0] if r.face_blendshapes else [])

    def landmarks_any(self, frame, fallback_roi=None):
        """Full frame, then this frame's pose head ROI, then fallback_roi. -> (pts, blend, roi_used)."""
        pts, bs = self.landmarks(frame)
        if pts is not None:
            return pts, bs, None
        for roi in (self.head_roi(frame), fallback_roi):
            if roi is not None:
                pts, bs = self.landmarks(frame, roi)
                if pts is not None:
                    return pts, bs, roi
        return None, None, None

    def face_frame(self, frame, roi=None):
        """-> (gap, jaw, box) with box = (cx, cy, h) normalized to the full frame, or NaNs if no face.
        The lip-gap ratio (inner-lip gap / face height) is scale invariant."""
        pts, bs = self.landmarks(frame, roi)
        return self._measure(frame, pts, bs)

    @staticmethod
    def _measure(frame, pts, bs):
        if pts is None:
            return np.nan, np.nan, (np.nan, np.nan, np.nan)
        H, W = frame.shape[:2]
        face_h = abs(pts[152, 1] - pts[10, 1]) or 1e-6
        gap = abs(pts[14, 1] - pts[13, 1]) / face_h
        jaw = next((b.score for b in bs if b.category_name == "jawOpen"), np.nan)
        xs, ys = pts[:, 0] / W, pts[:, 1] / H
        return gap, jaw, (float((xs.min() + xs.max()) / 2), float((ys.min() + ys.max()) / 2),
                          float(ys.max() - ys.min()))

    def head_roi(self, frame):
        """Square head region (x0, y0, side) in pixels from pose landmarks, or None."""
        if self.pose is None:
            return None
        r = self.pose.detect(self._img(frame))
        if not r.pose_landmarks:
            return None
        H, W = frame.shape[:2]
        lm = r.pose_landmarks[0]
        pts = np.array([(lm[i].x * W, lm[i].y * H) for i in range(11)])  # nose, eyes, ears, mouth
        sh = np.array([(lm[i].x * W, lm[i].y * H) for i in (11, 12)])
        span = np.ptp(pts[:, 0])
        side = max(3.0 * span, 0.9 * float(np.linalg.norm(sh[0] - sh[1])), 64.0)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean() - 0.1 * side
        return (cx - side / 2, cy - side / 2, side)

    def pose_frame(self, frame):
        if self.pose is None:
            return False
        return bool(self.pose.detect(self._img(frame)).pose_landmarks)

    def faces(self, frames, fallback_rois=None):
        """Arrays for a frame list: gap[n], jaw[n], box[n, 3], rois (list).

        Per frame: full-frame detection; if it fails, the pose head ROI of this frame; if that fails, the
        caller-provided ROI (e.g. the aligned input frame's, for G-Star whose pose may not be detected).
        """
        res, rois = [], []
        for i, f in enumerate(frames):
            fb = fallback_rois[i] if fallback_rois is not None and i < len(fallback_rois) else None
            pts, bs, roi = self.landmarks_any(f, fb)
            r = self._measure(f, pts, bs)
            if roi is None and pts is not None:  # remember where the face was, as an ROI for the partner
                H, W = f.shape[:2]
                side = max(64.0, 2.2 * r[2][2] * H)
                roi = (r[2][0] * W - side / 2, r[2][1] * H - side / 2, side)
            res.append(r)
            rois.append(roi)
        return (np.array([r[0] for r in res], float), np.array([r[1] for r in res], float),
                np.array([r[2] for r in res], float), rois)

    def pose_rate(self, frames):
        if self.pose is None or not frames:
            return None
        return float(np.mean([self.pose_frame(f) for f in frames]))

    def close(self):
        for d in (self.face, self.pose):
            try:
                d and d.close()
            except Exception:
                pass


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() <= 10 or a[m].std() < 1e-6 or b[m].std() < 1e-6:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def best_lag(a, b, max_lag=8):
    """Best correlation of out[t] with in[t - lag] for lag in [-max_lag, max_lag] -> (corr, lag)."""
    best = (None, 0)
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], float), np.asarray(b[:n], float)
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[:n - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:n + lag])
        c = corr(x, y)
        if c is not None and (best[0] is None or c > best[0]):
            best = (round(c, 3), lag)
    return best


def framing(box_in, box_out, min_frames=10):
    """Median relative framing of output vs input from face boxes on aligned frames.

    Returns dict(scale, dx, dy, n): scale = output face height / input face height (>1: output zoomed in /
    cropped), dx, dy = output face centre - input face centre (fraction of frame). None if too few faces.
    """
    m = ~np.isnan(box_in[:, 2]) & ~np.isnan(box_out[:, 2])
    if m.sum() < min_frames:
        return None
    s = np.median(box_out[m, 2] / np.maximum(box_in[m, 2], 1e-6))
    return {"scale": round(float(s), 3), "dx": round(float(np.median(box_out[m, 0] - box_in[m, 0])), 3),
            "dy": round(float(np.median(box_out[m, 1] - box_in[m, 1])), 3), "n": int(m.sum())}
