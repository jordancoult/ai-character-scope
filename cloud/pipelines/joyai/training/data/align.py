"""Temporal alignment of an input/output pair.

Convention: offset_s = t_input - t_output for the same event (input time = output time + offset_s).

Audio path (default), mono 16 kHz:
  1. raw-waveform normalized FFT cross-correlation over the whole clip -> decisive when both files carry the same
     audio master (TikTok re-uploads / re-encodes keep it);
  2. otherwise onset-strength envelope (log-spectral flux, 100 Hz) cross-correlation, refined on the waveform
     within +-30 ms.
Confidence = peak / best competing peak outside +-0.15 s of the main one (repetitive music gives ~1, a clean
match gives >> 1); `peak` is the normalized correlation.

Fallback (no audio in one of the files): frame-difference motion envelope at the target fps. When both exist,
the motion estimate is kept as a cross-check (build_pairs flags disagreement).
"""

import numpy as np
from scipy import signal

SR = 16000
HOP = 160  # 10 ms -> 100 Hz envelope


def onset_envelope(x, sr=SR, hop=HOP):
    if len(x) < 4 * hop:
        return np.zeros(0)
    _, _, Z = signal.stft(x, fs=sr, nperseg=4 * hop, noverlap=3 * hop, boundary=None, padded=False)
    S = np.log1p(100 * np.abs(Z))
    flux = np.maximum(0, np.diff(S, axis=1)).sum(axis=0)
    flux = np.concatenate([[0], flux])
    flux -= signal.medfilt(flux, 31) if len(flux) > 31 else flux.mean()  # remove slow loudness trend
    return np.maximum(flux, 0)


def _xcorr_norm(a, b):
    """Normalized cross-correlation. Returns (lags, c) where c[lag] ~ corr(a[n + lag], b[n]); i.e. the feature at b's time n is at a's time n + lag.
    """
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    c = signal.correlate(a, b, mode="full", method="fft")
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    # normalize by overlap length so partial overlaps are comparable
    ov = (np.minimum(len(b), len(a) - lags) - np.maximum(0, -lags)).astype(float).clip(1, None)
    return lags, c / ov, ov


def _peak_conf(lags, c, ov, rate, min_overlap_s, excl_s=0.15):
    valid = ov >= min_overlap_s * rate
    if not valid.any():
        return None
    cc = np.where(valid, c, -np.inf)
    i = int(np.argmax(cc))
    excl = int(excl_s * rate)
    mask = valid.copy()
    mask[max(0, i - excl):i + excl + 1] = False
    second = float(np.max(c[mask])) if mask.any() else 0.0
    peak = float(c[i])
    ratio = peak / second if second > 1e-9 else 99.0
    return int(lags[i]), peak, min(ratio, 99.0)


def align_audio(a_in, a_out, sr=SR, min_overlap_s=2.0):
    """Return dict(offset_s, peak, ratio, wave_ncc, method) or None if either signal is missing/silent.

    1) raw-waveform cross-correlation over the whole clip (decisive when both carry the same audio master,
       even re-encoded); 2) if that is not convincing, onset-envelope cross-correlation (robust to different
       mixes / codecs), refined on the waveform within +-30 ms.
    """
    if len(a_in) < sr or len(a_out) < sr or a_in.std() < 1e-4 or a_out.std() < 1e-4:
        return None
    lags, c, ov = _xcorr_norm(a_in, a_out)
    wv = _peak_conf(lags, c, ov, sr, min_overlap_s, excl_s=0.15)
    if wv and wv[1] >= 0.3 and wv[2] >= 2.0:
        lag, peak, ratio = wv
        return {"offset_s": round(lag / sr, 4), "coarse_offset_s": round(lag / sr, 3), "peak": round(peak, 3),
                "ratio": round(ratio, 3), "wave_ncc": round(peak, 3), "method": "audio_wave"}
    rate = sr / HOP
    e_in, e_out = onset_envelope(a_in, sr), onset_envelope(a_out, sr)
    if e_in.std() < 1e-9 or e_out.std() < 1e-9:
        return None
    lags, c, ov = _xcorr_norm(e_in, e_out)
    pk = _peak_conf(lags, c, ov, rate, min_overlap_s)
    if pk is None:
        return None
    lag, peak, ratio = pk
    coarse = lag / rate  # input time = output time + coarse
    fine, wpeak = coarse, None
    try:
        s0 = int(round(coarse * sr))
        w = int(0.03 * sr)
        o0 = max(0, -s0) + w                       # overlapping region in output samples
        o1 = min(len(a_out), len(a_in) - s0) - w
        o1 = min(o1, o0 + 20 * sr)                 # 20 s is plenty
        if o1 - o0 > sr:
            seg_out = a_out[o0:o1]
            seg_in = a_in[o0 + s0 - w:o1 + s0 + w]
            cc = signal.correlate(seg_in, seg_out, mode="valid", method="fft")  # 2w+1 lags
            energy = signal.correlate(seg_in ** 2, np.ones(len(seg_out)), mode="valid", method="fft")
            ncc = cc / np.sqrt(np.sum(seg_out ** 2) * energy.clip(1e-12))
            k = int(np.argmax(ncc))
            wpeak = float(ncc[k])
            if wpeak > 0.2:  # waveforms agree locally; otherwise keep the envelope lag (10 ms resolution)
                fine = (s0 - w + k) / sr
    except Exception:
        pass
    return {"offset_s": round(fine, 4), "coarse_offset_s": round(coarse, 3), "peak": round(peak, 3),
            "ratio": round(ratio, 3), "wave_ncc": None if wpeak is None else round(wpeak, 3),
            "method": "audio_envelope"}


def motion_envelope(frames):
    """Mean absolute frame difference on small grayscale frames (one value per frame)."""
    import cv2
    prev, out = None, []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (64, 112)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        out.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
        prev = g
    e = np.array(out)
    # box-smooth over ~5 frames: fps conversion (e.g. 24 -> 30) duplicates frames, which zeroes single diffs
    e = np.convolve(e, np.ones(5) / 5, mode="same")
    if len(e) > 31:
        e = e - signal.medfilt(e, 31)
    # robust: cuts / fades from black make single huge spikes that would dominate the correlation
    s = np.median(np.abs(e)) * 1.4826 + 1e-6
    return np.clip(e, -4 * s, 4 * s)


def align_motion(frames_in, frames_out, fps=30.0, min_overlap_s=2.0):
    min_overlap_s = max(min_overlap_s, 0.4 * min(len(frames_in), len(frames_out)) / fps)
    e_in, e_out = motion_envelope(frames_in), motion_envelope(frames_out)
    if len(e_in) < fps or len(e_out) < fps or e_in.std() < 1e-6 or e_out.std() < 1e-6:
        return None
    lags, c, ov = _xcorr_norm(e_in, e_out)
    pk = _peak_conf(lags, c, ov, fps, min_overlap_s, excl_s=0.2)
    if pk is None:
        return None
    lag, peak, ratio = pk
    i = int(np.where(lags == lag)[0][0])
    if 0 < i < len(c) - 1:  # parabolic sub-frame refinement
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            lag = lag + float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return {"offset_s": round(lag / fps, 4), "coarse_offset_s": round(lag / fps, 3), "peak": round(peak, 3),
            "ratio": round(ratio, 3), "wave_ncc": None, "method": "motion"}


def overlap(offset_s, dur_in, dur_out):
    """Overlapping span: returns (in_start, out_start, duration)."""
    out_start = max(0.0, -offset_s)
    in_start = out_start + offset_s
    dur = min(dur_in - in_start, dur_out - out_start)
    return in_start, out_start, max(0.0, dur)
