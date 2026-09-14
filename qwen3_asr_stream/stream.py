"""Qwen3-ASR streaming loop aligned with the official SDK + paper.

Official references
- SDK (`qwen_asr/inference/qwen3_asr.py`):
    chunk_size_sec=2.0, unfixed_chunk_num=2, unfixed_token_num=5
    force_language → assistant prefill `language X<asr_text>` (text-only)
- Paper (arXiv:2601.21337 §streaming): 2 s chunks, 5-token fallback,
    last 4 chunks unfixed
- LID is evaluated on full utterances, not 400 ms slices. Guessing on
    padded short hops is why language ID looked chaotic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .audio import SAMPLE_RATE
from .client import DecodeResult, LlamaAsrClient
from .parse import (
    classify_sound_event_text,
    combine_event_and_transcript,
    has_lexical_speech,
    merge_languages,
    parse_asr_output,
    strip_event_prefix,
    stitch_transcript,
)
from .sound_gate import SoundHint, analyze as analyze_sound, should_skip_asr
from .vad import EnergyVAD


OnUpdate = Callable[["StreamState"], None]


@dataclass
class StreamConfig:
    hop_sec: float = 0.40
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5
    max_audio_sec: float = 24.0
    min_audio_sec: float = 0.50
    overlap_sec: float = 0.80
    max_tokens: int = 128
    refine_max_tokens: int = 256
    temperature: float = 0.01
    language: Optional[str] = None
    context: str = ""
    vad: bool = True
    silence_commit_sec: float = 0.70
    stream_tokens: bool = True
    # MIX default: never force language. Official force_language is
    # text-only for ONE language and breaks campur Indo/English.
    # LIVE hops are drafts. LAST is a full official offline pass.
    lid_chunk_sec: float = 1.0
    lid_confirm_chunks: int = 1
    relid_silence_sec: float = 0.0
    lid_lock: bool = False
    unlock_on_utterance: bool = True
    refine_on_commit: bool = True
    sound_gate: bool = True
    sound_gate_min_conf: float = 0.62
    # auto = PANNs CNN6 if torch installed, else heuristic numpy gate
    sound_model: str = "auto"
    pann_interval_sec: float = 1.5
    pann_min_score: float = 0.45
    pann_block_score: float = 0.55
    pann_companion_score: float = 0.38
    pann_speech_score: float = 0.18


def profile_config(name: str) -> StreamConfig:
    name = (name or "ultralow").strip().lower()
    if name in ("official", "sdk", "vllm"):
        return StreamConfig(
            hop_sec=2.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            max_audio_sec=0.0,
            min_audio_sec=0.50,
            max_tokens=128,
            silence_commit_sec=0.70,
            lid_chunk_sec=2.0,
            lid_confirm_chunks=1,
            lid_lock=False,
            unlock_on_utterance=True,
            refine_on_commit=True,
        )
    if name in ("paper",):
        return StreamConfig(
            hop_sec=2.0,
            unfixed_chunk_num=4,
            unfixed_token_num=5,
            max_audio_sec=0.0,
            min_audio_sec=0.50,
            max_tokens=128,
            silence_commit_sec=0.70,
            lid_chunk_sec=2.0,
            lid_confirm_chunks=1,
            lid_lock=False,
            unlock_on_utterance=True,
            refine_on_commit=True,
        )
    if name in ("balanced", "mid"):
        return StreamConfig(
            hop_sec=0.70,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            max_audio_sec=24.0,
            min_audio_sec=0.50,
            max_tokens=128,
            silence_commit_sec=0.70,
            lid_chunk_sec=1.0,
            lid_confirm_chunks=1,
            lid_lock=False,
            unlock_on_utterance=True,
            refine_on_commit=True,
        )
    return StreamConfig(
        hop_sec=0.40,
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        max_audio_sec=24.0,
        min_audio_sec=0.50,
        max_tokens=128,
        silence_commit_sec=0.70,
        lid_chunk_sec=1.0,
        lid_confirm_chunks=1,
        lid_lock=False,
        unlock_on_utterance=True,
        refine_on_commit=True,
    )


@dataclass
class StreamState:
    cfg: StreamConfig
    chunk_id: int = 0
    buffer: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    raw_decoded: str = ""
    language: str = ""
    locked_language: str = ""
    language_status: str = "waiting"
    lid_votes: list[str] = field(default_factory=list)
    lid_rounds: int = 0
    text: str = ""
    committed: str = ""
    unfixed: str = ""
    finalized: str = ""
    last: Optional[DecodeResult] = None
    utterance_id: int = 0
    speech_seen: bool = False
    silence_sec: float = 0.0
    speaking: bool = False
    decoding: bool = False
    level: float = 0.0
    languages_seen: list[str] = field(default_factory=list)
    utterance_langs: list[str] = field(default_factory=list)
    refining: bool = False
    sound_hint: Optional[SoundHint] = None
    sound_label: str = ""
    non_speech_only: bool = False
    non_speech_hops: int = 0
    event_label: str = ""
    event_score: float = 0.0
    event_top: tuple[str, ...] = ()
    pann_last_sec: float = 0.0
    pann_cached: object = None

    @property
    def display_text(self) -> str:
        return ((self.committed + " " if self.committed else "") + self.unfixed).strip()

    @property
    def force_language(self) -> Optional[str]:
        if self.cfg.language:
            return self.cfg.language
        if self.cfg.lid_lock and self.locked_language:
            return self.locked_language
        return None


class StreamingAsr:
    def __init__(self, client: LlamaAsrClient, cfg: StreamConfig, on_update: Optional[OnUpdate] = None):
        self.client = client
        self.cfg = cfg
        self.on_update = on_update
        self.state = StreamState(cfg=cfg)
        self.vad = EnergyVAD() if cfg.vad else None
        self._pann = self._init_pann()
        if cfg.language:
            self.state.locked_language = cfg.language
            self.state.language = cfg.language
            self.state.language_status = "forced"

    def reset_utterance(self, keep_tail: bool = False, unlock: bool = False) -> None:
        st = self.state
        tail = np.zeros((0,), dtype=np.float32)
        if keep_tail and st.audio_accum.size:
            n = int(self.cfg.overlap_sec * SAMPLE_RATE)
            tail = st.audio_accum[-n:]
        st.buffer = np.zeros((0,), dtype=np.float32)
        st.audio_accum = tail
        st.raw_decoded = ""
        st.text = ""
        st.unfixed = ""
        st.committed = ""
        st.chunk_id = 0
        st.lid_rounds = 0
        st.utterance_langs = []
        st.refining = False
        st.sound_hint = None
        st.sound_label = ""
        st.non_speech_only = False
        st.non_speech_hops = 0
        st.event_label = ""
        st.event_score = 0.0
        st.event_top = ()
        st.pann_last_sec = 0.0
        st.pann_cached = None
        st.speech_seen = tail.size > 0
        st.silence_sec = 0.0
        st.speaking = False
        st.decoding = False
        st.utterance_id += 1
        if unlock and not self.cfg.language:
            st.locked_language = ""
            st.language = ""
            st.language_status = "waiting"
            st.lid_votes = []
        elif self.cfg.language:
            st.language = self.cfg.language
            st.language_status = "forced"
        elif st.locked_language and not unlock:
            st.language = st.locked_language
            st.language_status = "locked"
        else:
            st.language = ""
            st.language_status = "waiting"

    def push(self, pcm16k: np.ndarray) -> None:
        x = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return
        st = self.state
        st.level = (
            float(self.vad.rms(x))
            if self.vad is not None
            else float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        )
        speaking = True
        if self.vad is not None:
            speaking = self.vad.is_speech(x)
            dt = x.size / float(SAMPLE_RATE)
            if speaking:
                st.speech_seen = True
                st.silence_sec = 0.0
            else:
                st.silence_sec += dt
                if not st.speech_seen:
                    st.speaking = False
                    self._maybe_unlock_on_long_silence()
                    self._emit()
                    return
        st.speaking = bool(speaking)

        if speaking or st.speech_seen:
            st.buffer = np.concatenate([st.buffer, x], axis=0)

        hop = int(round(self.cfg.hop_sec * SAMPLE_RATE))
        if st.buffer.size >= hop:
            chunk = st.buffer
            st.buffer = np.zeros((0,), dtype=np.float32)
            self._consume_chunk(chunk)

        commit_after = self.cfg.silence_commit_sec
        if (
            self.vad is not None
            and st.speech_seen
            and st.silence_sec >= commit_after
        ):
            self.commit()
            unlock = bool(self.cfg.unlock_on_utterance) and not self.cfg.language
            self.reset_utterance(keep_tail=False, unlock=unlock)
            self._emit()

    def finish(self) -> StreamState:
        st = self.state
        if st.buffer.size:
            tail = st.buffer
            st.buffer = np.zeros((0,), dtype=np.float32)
            self._consume_chunk(tail, allow_short=True)
        return st

    def commit(self) -> str:
        self.finish()
        st = self.state
        sealed = self._refine_utterance()
        if not st.non_speech_only:
            st.language = merge_languages(st.utterance_langs) or st.language
        st.finalized = sealed
        self._emit()
        return sealed

    def _non_speech_label(
        self,
        audio: np.ndarray,
        text: str,
        hint: Optional[SoundHint],
    ) -> Optional[str]:
        if not self.cfg.sound_gate:
            return classify_sound_event_text(text)
        event = classify_sound_event_text(text)
        if event:
            return event
        if hint is None:
            hint = analyze_sound(audio)
        if should_skip_asr(hint, self.cfg.sound_gate_min_conf):
            return hint.label or "suara non-bicara"
        if (
            not hint.is_speech
            and hint.category in ("impulse", "burst", "tonal", "noise")
            and hint.confidence >= 0.42
        ):
            return hint.label or "suara non-bicara"
        st = self.state
        if st.non_speech_hops > 0 and hint and not hint.is_speech:
            return st.sound_label or hint.label or "suara non-bicara"
        return None

    def _seal_non_speech(self, label: str) -> str:
        st = self.state
        st.non_speech_only = True
        st.sound_label = label
        st.language_status = "non-speech"
        st.utterance_langs = []
        st.lid_votes = []
        st.unfixed = f"[{label}]"
        st.text = st.unfixed
        st.raw_decoded = ""
        return st.unfixed

    def _refine_utterance(self) -> str:
        """Official offline transcribe of the full utterance (no prefix, no force)."""
        st = self.state
        draft = stitch_transcript(st.committed, st.text)
        audio = st.audio_accum
        min_n = int(self.cfg.min_audio_sec * SAMPLE_RATE)
        hint = analyze_sound(audio) if self.cfg.sound_gate else None
        st.sound_hint = hint
        blocked = self._non_speech_label(audio, draft, hint)
        if blocked or (st.non_speech_only and st.sound_label):
            return self._seal_non_speech(blocked or st.sound_label)
        if not self.cfg.refine_on_commit or audio.size < min_n:
            return draft
        st.refining = True
        st.language_status = "refining"
        st.decoding = True
        self._emit()
        try:
            result = self.client.transcribe(
                audio,
                raw_prefix="",
                context=self.cfg.context,
                force_language=self.cfg.language,
                max_tokens=self.cfg.refine_max_tokens,
                temperature=0.01,
                on_partial=None,
            )
        finally:
            st.refining = False
            st.decoding = False
        blocked = self._non_speech_label(audio, result.text, hint)
        if blocked:
            return self._seal_non_speech(blocked)
        if result.language:
            self._remember_language(result.language)
            if result.language not in st.utterance_langs:
                st.utterance_langs.append(result.language)
        if result.text.strip():
            st.last = result
            st.text = result.text
            st.unfixed = result.text
            return result.text
        return draft

    def _maybe_unlock_on_long_silence(self) -> None:
        st = self.state
        if (
            self.cfg.relid_silence_sec > 0
            and st.silence_sec >= self.cfg.relid_silence_sec
            and not self.cfg.language
            and st.locked_language
        ):
            st.locked_language = ""
            st.language = ""
            st.language_status = "waiting"
            st.lid_votes = []
            st.lid_rounds = 0

    def _ready_for_decode(self, allow_short: bool) -> bool:
        st = self.state
        audio_sec = st.audio_accum.size / float(SAMPLE_RATE)
        if st.audio_accum.size == 0:
            return False
        min_s = self.cfg.min_audio_sec
        if allow_short:
            return audio_sec >= min_s * 0.6
        # Open LID and locked text both start as soon as the encoder minimum
        # is met, so LIVE is not blank while SPEAKING.
        return audio_sec >= min_s

    def _consume_chunk(self, chunk: np.ndarray, allow_short: bool = False) -> None:
        st = self.state
        if st.audio_accum.size == 0:
            st.audio_accum = chunk
        else:
            st.audio_accum = np.concatenate([st.audio_accum, chunk], axis=0)

        if self.cfg.max_audio_sec > 0:
            cap = int(self.cfg.max_audio_sec * SAMPLE_RATE)
            if st.audio_accum.size > cap:
                if st.text:
                    st.committed = stitch_transcript(st.committed, st.text)
                n = int(self.cfg.overlap_sec * SAMPLE_RATE)
                st.audio_accum = st.audio_accum[-n:]
                st.raw_decoded = ""
                st.chunk_id = 0
                st.unfixed = ""
                # Keep locked language across the rolling window (same session).

        if not self._ready_for_decode(allow_short):
            if not st.force_language:
                st.language_status = "waiting" if not st.lid_votes else "confirming"
            self._emit()
            return

        audio = st.audio_accum
        if self.cfg.sound_gate:
            hint = analyze_sound(audio)
            st.sound_hint = hint
            if should_skip_asr(hint, self.cfg.sound_gate_min_conf):
                st.non_speech_hops += 1
                st.non_speech_only = True
                st.sound_label = hint.label or "suara non-bicara"
                st.language_status = "non-speech"
                st.unfixed = f"[{st.sound_label}]"
                st.text = st.unfixed
                st.raw_decoded = ""
                st.decoding = False
                self._emit()
                return
            st.non_speech_only = False
            st.sound_label = ""
        min_n = int(self.cfg.min_audio_sec * SAMPLE_RATE)
        if audio.size < min_n:
            if not allow_short:
                self._emit()
                return
            audio = np.pad(audio, (0, min_n - int(audio.size)))

        force = st.force_language
        prefix = ""
        clean_prev = parse_asr_output(st.raw_decoded, user_language=force or None)[1] or st.text
        if st.chunk_id >= 1 and clean_prev:
            prefix = self.client.rollback_prefix(clean_prev, self.cfg.unfixed_token_num)

        def on_partial(lang: str, text: str) -> None:
            if classify_sound_event_text(text):
                st.unfixed = f"[{classify_sound_event_text(text)}]"
                self._emit()
                return
            if not force:
                st.language = lang or st.language
            else:
                st.language = force
            st.unfixed = text
            self._emit()

        st.decoding = True
        if not force:
            st.language_status = "confirming" if st.lid_votes else "guessing"
        self._emit()
        try:
            result = self.client.transcribe(
                audio,
                raw_prefix=prefix,
                context=self.cfg.context,
                force_language=force,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                on_partial=on_partial if self.cfg.stream_tokens else None,
            )
        finally:
            st.decoding = False

        blocked = self._non_speech_label(audio, result.text, st.sound_hint)
        if blocked:
            st.non_speech_hops += 1
            self._seal_non_speech(blocked)
            st.chunk_id += 1
            self._emit()
            return

        st.last = result
        st.raw_decoded = result.text
        st.text = result.text
        st.unfixed = result.text
        st.chunk_id += 1

        if force:
            st.language = force
            st.language_status = "forced" if self.cfg.language else "locked"
        else:
            guessed = (result.language or "").strip()
            if guessed and guessed.lower() != "none":
                st.lid_votes.append(guessed)
                if guessed not in st.utterance_langs:
                    st.utterance_langs.append(guessed)
                self._remember_language(guessed)
            st.lid_rounds += 1
            if self.cfg.lid_lock:
                self._maybe_lock_language()
            else:
                st.language = merge_languages(st.utterance_langs) or guessed
                st.language_status = "mix" if st.utterance_langs else "guessing"

        self._remember_language(st.language)
        self._emit()

    def _maybe_lock_language(self) -> None:
        st = self.state
        if not self.cfg.lid_lock or self.cfg.language:
            return
        if not st.lid_votes:
            st.language_status = "waiting"
            return
        # Official: after N open 2 s chunks, lock the language from the
        # longest-context decode (last vote). Confirm if the last two agree.
        audio_sec = st.audio_accum.size / float(SAMPLE_RATE)
        if st.lid_rounds < self.cfg.lid_confirm_chunks or audio_sec < self.cfg.lid_chunk_sec:
            st.language_status = "confirming" if st.lid_votes else "guessing"
            return
        last = st.lid_votes[-1]
        if len(st.lid_votes) >= 2 and st.lid_votes[-1] != st.lid_votes[-2]:
            # Longer audio wins (official later chunk is more reliable).
            last = st.lid_votes[-1]
        counts = Counter(st.lid_votes)
        majority, n = counts.most_common(1)[0]
        chosen = majority if n >= 2 else last
        st.locked_language = chosen
        st.language = chosen
        st.language_status = "locked"

    def _remember_language(self, lang: str) -> None:
        lang = (lang or "").strip()
        if lang and lang not in self.state.languages_seen:
            self.state.languages_seen.append(lang)

    def _emit(self) -> None:
        if self.on_update is not None:
            self.on_update(self.state)
