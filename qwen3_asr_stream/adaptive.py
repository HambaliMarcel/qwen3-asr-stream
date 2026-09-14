"""Runtime auto-tuning for hop, pause, refine, and sound tagging."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .client import DecodeResult
from .parse import has_lexical_speech

HOP_MIN = 0.80
HOP_MAX = 2.00
PAUSE_MIN = 1.35
PAUSE_MAX = 2.20


def _mean(values: deque, default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _snap(value: float, step: float = 0.05) -> float:
    return round(round(value / step) * step, 2)


@dataclass
class RuntimeTuner:
    """Adjust streaming knobs from live RTF, utterance shape, and false tags."""

    enabled: bool = True
    locked: set[str] = field(default_factory=set)
    hop_sec: float = 1.0
    silence_commit_sec: float = 1.50
    silence_hangover_sec: float = 0.45
    refine_on_commit: bool = True
    sound_gate: bool = True
    sound_model: str = "auto"
    pann_interval_sec: float = 1.5
    hint: str = "auto · warming up"

    _rtf: deque = field(default_factory=lambda: deque(maxlen=8))
    _latency_ms: deque = field(default_factory=lambda: deque(maxlen=8))
    _false_tags: int = 0
    _early_commits: int = 0
    _early_streak: int = 0
    _good_commits: int = 0
    _clean_since_false: int = 0
    _decode_count: int = 0

    @classmethod
    def from_config(cls, cfg) -> RuntimeTuner:
        enabled = bool(getattr(cfg, "auto_tune", True))
        return cls(
            enabled=enabled,
            locked=set(getattr(cfg, "manual_fields", set()) or ()),
            hop_sec=float(cfg.hop_sec),
            silence_commit_sec=float(cfg.silence_commit_sec),
            silence_hangover_sec=float(getattr(cfg, "silence_hangover_sec", 0.45)),
            refine_on_commit=bool(cfg.refine_on_commit),
            sound_gate=bool(cfg.sound_gate),
            sound_model=str(cfg.sound_model or "auto").strip().lower(),
            pann_interval_sec=float(cfg.pann_interval_sec),
            hint="auto · warming up" if enabled else "",
        )

    def _lock(self, name: str) -> bool:
        return name in self.locked

    def record_decode(self, result: DecodeResult | None) -> None:
        if not self.enabled or result is None:
            return
        self._decode_count += 1
        # First GPU pass is CUDA/graph warmup — do not steer hop from it.
        if self._decode_count == 1:
            self._refresh_hint()
            return
        self._rtf.append(float(result.rtf))
        self._latency_ms.append(float(result.latency_ms))
        if len(self._rtf) >= 2 and self._decode_count % 2 == 0:
            self._retune()
        else:
            self._refresh_hint()

    def record_commit(self, utterance_sec: float, silence_sec: float) -> None:
        if not self.enabled:
            return
        restored = False
        long_enough = utterance_sec >= 1.0
        real_pause = silence_sec >= self.silence_commit_sec + 0.25
        if long_enough and real_pause:
            self._good_commits += 1
            self._early_streak = 0
            self._clean_since_false += 1
            restored = self._maybe_restore_tags()
            if (
                not self._lock("silence_commit_sec")
                and self._early_commits == 0
                and self._good_commits >= 3
            ):
                self.silence_commit_sec = _snap(
                    max(PAUSE_MIN, self.silence_commit_sec - 0.05)
                )
        if self._rtf:
            self._retune(allow_tag_promote=not restored)
        else:
            self._refresh_hint()

    def record_early_resume(self) -> None:
        """Commit likely cut a sentence: silence was at threshold, speech resumed immediately."""
        if not self.enabled or self._lock("silence_commit_sec"):
            return
        self._early_streak += 1
        if self._early_streak < 2:
            self._refresh_hint()
            return
        self._early_commits += 1
        self._early_streak = 0
        self.silence_commit_sec = _snap(
            min(PAUSE_MAX, self.silence_commit_sec + 0.10)
        )
        self.silence_hangover_sec = _snap(
            min(0.65, self.silence_hangover_sec + 0.05)
        )
        self._refresh_hint()

    def record_false_tag(self) -> None:
        """LAST sealed as non-speech, then real speech resumed (true false lock)."""
        if not self.enabled:
            return
        self._false_tags += 1
        self._clean_since_false = 0
        if self._false_tags >= 2 and not self._lock("sound_model"):
            if self.sound_model in ("auto", "pann", "pann_cnn6", "cnn6"):
                self.sound_model = "heuristic"
            elif self.sound_model == "heuristic":
                self.sound_model = "off"
                if not self._lock("sound_gate"):
                    self.sound_gate = False
        if not self._lock("pann_interval_sec"):
            self.pann_interval_sec = min(4.0, self.pann_interval_sec + 0.5)
        self._refresh_hint()

    def _maybe_restore_tags(self) -> bool:
        if self._lock("sound_model") or self._clean_since_false < 4:
            return False
        if self._false_tags <= 0:
            return False
        if self.sound_model == "off":
            self.sound_model = "heuristic"
            if not self._lock("sound_gate"):
                self.sound_gate = True
        elif self.sound_model == "heuristic":
            self.sound_model = "auto"
        else:
            return False
        self._false_tags = max(0, self._false_tags - 1)
        self._clean_since_false = 0
        if not self._lock("pann_interval_sec"):
            self.pann_interval_sec = max(1.5, self.pann_interval_sec - 0.5)
        return True

    def should_refine(
        self,
        utterance_sec: float,
        draft: str,
        has_tail: bool = False,
    ) -> bool:
        """Whether to run a background LAST seal.

        The seal is prefixed with the live draft and streams tokens, so it is
        cheap: run it for every real utterance to catch the tail and stabilize
        the line. Skip only when the tuner disabled refine under load and
        there is no undecoded tail to rescue.
        """
        lexical = has_lexical_speech(draft)
        if not lexical and utterance_sec < 0.5:
            return False
        if has_tail:
            return True
        if not self.refine_on_commit:
            return False
        if not lexical:
            return utterance_sec >= 0.5
        if utterance_sec < 0.6:
            return False
        if not self.enabled or self._lock("refine_on_commit"):
            return True
        avg_rtf = _mean(self._rtf)
        if avg_rtf > 0.85:
            return False
        return True

    def _target_hop(self, avg_rtf: float, avg_lat: float) -> float:
        if avg_rtf >= 0.55 or avg_lat >= 1100:
            return HOP_MAX
        if avg_rtf >= 0.35 or avg_lat >= 750:
            return 1.40
        if avg_rtf <= 0.12 and avg_lat <= 350:
            return HOP_MIN
        return 1.00

    def _retune(self, allow_tag_promote: bool = True) -> None:
        avg_rtf = _mean(self._rtf)
        avg_lat = _mean(self._latency_ms)
        if not self._rtf:
            self._refresh_hint(avg_rtf, avg_lat)
            return

        if not self._lock("hop_sec"):
            target = self._target_hop(avg_rtf, avg_lat)
            blended = 0.65 * self.hop_sec + 0.35 * target
            self.hop_sec = _snap(_clamp(blended, HOP_MIN, HOP_MAX))

        if not self._lock("refine_on_commit"):
            if avg_rtf > 0.75:
                self.refine_on_commit = False
            elif avg_rtf < 0.28:
                self.refine_on_commit = True

        if not self._lock("sound_model"):
            if avg_rtf >= 0.70 and self.sound_model in (
                "auto",
                "pann",
                "pann_cnn6",
                "cnn6",
            ):
                self.sound_model = "heuristic"
            elif (
                allow_tag_promote
                and avg_rtf <= 0.18
                and self.sound_model == "heuristic"
                and self._false_tags == 0
            ):
                self.sound_model = "auto"
        if not self._lock("pann_interval_sec"):
            if avg_rtf >= 0.45:
                self.pann_interval_sec = min(3.5, max(self.pann_interval_sec, 2.5))
            elif avg_rtf <= 0.15:
                self.pann_interval_sec = max(1.5, min(self.pann_interval_sec, 2.0))

        self._refresh_hint(avg_rtf, avg_lat)

    def _refresh_hint(self, avg_rtf: float = 0.0, avg_lat: float = 0.0) -> None:
        if not self.enabled:
            self.hint = ""
            return
        if self._rtf:
            avg_rtf = _mean(self._rtf)
        if self._latency_ms:
            avg_lat = _mean(self._latency_ms)
        parts = [
            f"hop {self.hop_sec:.1f}s",
            f"pause {self.silence_commit_sec:.1f}s",
        ]
        if not self.refine_on_commit:
            parts.append("no-refine")
        if self.sound_model == "off":
            parts.append("asr-only")
        elif self.sound_model == "heuristic":
            parts.append("light-tags")
        if avg_rtf > 0:
            parts.append(f"rtf {avg_rtf:.2f}×")
        elif self._decode_count <= 1:
            parts.append("warming up")
        if avg_lat >= 500:
            parts.append(f"lat {avg_lat:.0f}ms")
        self.hint = "auto · " + " · ".join(parts)
