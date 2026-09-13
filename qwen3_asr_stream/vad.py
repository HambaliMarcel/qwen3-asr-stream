"""Lightweight adaptive energy VAD — no extra native deps."""

from __future__ import annotations

import numpy as np


class EnergyVAD:
    def __init__(
        self,
        speech_ratio: float = 2.4,
        min_rms: float = 0.008,
        noise_adapt: float = 0.05,
    ):
        self.speech_ratio = speech_ratio
        self.min_rms = min_rms
        self.noise_adapt = noise_adapt
        self.noise_rms = 0.004
        self._primed = False

    def rms(self, pcm: np.ndarray) -> float:
        if pcm.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(pcm), dtype=np.float32)))

    def is_speech(self, pcm: np.ndarray) -> bool:
        level = self.rms(pcm)
        if not self._primed:
            self.noise_rms = max(level, 0.002)
            self._primed = True
            return False

        thresh = max(self.min_rms, self.noise_rms * self.speech_ratio)
        speaking = level >= thresh
        if not speaking:
            self.noise_rms = (1.0 - self.noise_adapt) * self.noise_rms + self.noise_adapt * level
        return speaking
