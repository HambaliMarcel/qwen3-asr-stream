"""Lightweight speech vs non-speech hints (numpy only, no extra model).

This is NOT a sound-event classifier. It uses coarse audio features to guess
whether incoming audio is likely human speech or something else (clap, cough,
impact, hiss, etc.). Specific labels are low-confidence hints only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE


@dataclass(frozen=True)
class SoundHint:
    is_speech: bool
    category: str  # speech | impulse | burst | tonal | noise | silence
    label: str  # short UI label, may be empty
    confidence: float  # 0..1


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)))


def _spectral_features(x: np.ndarray) -> tuple[float, float, float]:
    """Return (centroid_hz, flatness, low_high_ratio)."""
    n = int(2 ** np.ceil(np.log2(max(512, x.size))))
    spec = np.abs(np.fft.rfft(x, n=n)) + 1e-12
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    p = spec / spec.sum()
    centroid = float((freqs * p).sum())
    geo = float(np.exp(np.mean(np.log(spec))))
    arith = float(spec.mean())
    flatness = geo / arith
    low = float(spec[freqs < 400].sum())
    high = float(spec[freqs >= 1200].sum())
    ratio = low / (high + 1e-9)
    return centroid, flatness, ratio


def _attack_ms(x: np.ndarray) -> float:
    if x.size < 64:
        return 999.0
    win = max(32, int(0.008 * SAMPLE_RATE))
    hop = win // 2
    peaks: list[float] = []
    for i in range(0, x.size - win, hop):
        peaks.append(_rms(x[i : i + win]))
    if not peaks:
        return 999.0
    peak = max(peaks)
    if peak <= 1e-6:
        return 999.0
    thr = peak * 0.35
    for i, v in enumerate(peaks):
        if v >= thr:
            return i * hop / SAMPLE_RATE * 1000.0
    return 999.0


def analyze(pcm16k: np.ndarray, min_rms: float = 0.008) -> SoundHint:
    x = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
    level = _rms(x)
    if level < min_rms:
        return SoundHint(False, "silence", "", 0.95)

    dur_ms = x.size / SAMPLE_RATE * 1000.0
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x))))) if x.size > 1 else 0.0
    centroid, flatness, low_high = _spectral_features(x)
    attack = _attack_ms(x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    crest = peak / (level + 1e-9)

    # Speech-ish: formant band energy, moderate flatness, not pure impulse.
    speech_score = 0.0
    if 350 <= centroid <= 3200:
        speech_score += 0.35
    if 0.05 <= flatness <= 0.55:
        speech_score += 0.25
    if 0.04 <= zcr <= 0.22:
        speech_score += 0.20
    if dur_ms >= 180:
        speech_score += 0.10
    if attack > 25:
        speech_score += 0.10
    if attack < 15:
        speech_score -= 0.30

    # Impulse: clap, knock, desk tap.
    impulse_score = 0.0
    if attack < 18 and crest > 6.0:
        impulse_score += 0.45
    if flatness > 0.35:
        impulse_score += 0.20
    if dur_ms < 220:
        impulse_score += 0.20
    if centroid > 900:
        impulse_score += 0.15

    # Burst: cough, sneeze, fart-ish pop, short exclamation.
    burst_score = 0.0
    if 120 <= dur_ms <= 900:
        burst_score += 0.20
    if flatness > 0.28:
        burst_score += 0.20
    if crest > 3.5:
        burst_score += 0.20
    if speech_score < 0.45:
        burst_score += 0.25
    if low_high > 1.4 and centroid < 900:
        burst_score += 0.15  # low rumble pop

    # Tonal / sustained non-speech: whistle, ring, some animal calls.
    tonal_score = 0.0
    if flatness < 0.12 and centroid > 250:
        tonal_score += 0.35
    if zcr < 0.06:
        tonal_score += 0.20
    if dur_ms > 250:
        tonal_score += 0.20
    if speech_score < 0.50:
        tonal_score += 0.25

    hard_impulse = crest >= 5.0 and attack < 22 and dur_ms < 280
    hard_burst = crest >= 4.0 and 140 <= dur_ms <= 950 and speech_score < 0.62
    hard_tonal = flatness < 0.10 and dur_ms > 220 and speech_score < 0.58
    if hard_impulse:
        impulse_score = max(impulse_score, 0.82)
    if hard_burst:
        burst_score = max(burst_score, 0.72)
    if hard_tonal:
        tonal_score = max(tonal_score, 0.70)

    scores = {
        "speech": speech_score,
        "impulse": impulse_score,
        "burst": burst_score,
        "tonal": tonal_score,
        "noise": max(0.0, 0.55 - speech_score),
    }
    category = max(scores, key=scores.get)
    best = scores[category]
    second = sorted(scores.values(), reverse=True)[1]
    confidence = max(0.0, min(1.0, best - second + 0.35))
    if hard_impulse and category == "impulse":
        confidence = max(confidence, 0.72)
    if hard_burst and category == "burst":
        confidence = max(confidence, 0.68)
    if hard_tonal and category == "tonal":
        confidence = max(confidence, 0.65)

    if category == "speech" and speech_score >= 0.58 and impulse_score < 0.70:
        return SoundHint(True, "speech", "", min(0.92, confidence))

    label_map = {
        "impulse": "tepuk / ketukan?",
        "burst": "batuk / bersin / ledakan?",
        "tonal": "siulan / nada / binatang?",
        "noise": "bukan suara bicara",
    }
    label = label_map.get(category, "bukan suara bicara")
    if confidence < 0.50:
        label = "suara non-bicara?"
        category = "noise"
    return SoundHint(False, category, label, confidence)


def should_skip_asr(hint: SoundHint, min_confidence: float = 0.62) -> bool:
    return (not hint.is_speech) and hint.category != "silence" and hint.confidence >= min_confidence
