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

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Companion tags go stale fast (music stops, cough ends). Never prefix a
# transcript with a tag older than this.
EVENT_TTL_SEC = 4.0
# Tail seal budget: prefix the live draft so only the tail is generated.
SEAL_MAX_TOKENS = 96
# Minimum speech-like energy in the leftover buffer to count as an undecoded tail.
TAIL_MIN_SEC = 0.08
TAIL_MIN_RMS = 0.010

from .audio import SAMPLE_RATE
from .client import DecodeResult, LlamaAsrClient
from .parse import (
    classify_sound_event_text,
    combine_event_and_transcript,
    has_lexical_speech,
    infer_languages,
    merge_languages,
    parse_asr_output,
    strip_event_prefix,
    stitch_transcript,
)
from .adaptive import RuntimeTuner
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
    # 0.7 s was cutting utterances on normal word gaps. Official streaming
    # uses ~2 s chunks; end-of-utterance needs a longer trailing silence.
    silence_commit_sec: float = 1.50
    # Ignore brief energy dips between syllables when counting commit silence.
    silence_hangover_sec: float = 0.45
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
    pann_speech_score: float = 0.35
    auto_tune: bool = True
    manual_fields: set[str] = field(default_factory=set)


def profile_config(name: str) -> StreamConfig:
    name = (name or "auto").strip().lower()
    if name in ("auto", "adaptive", "smart"):
        return StreamConfig(
            hop_sec=1.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            max_audio_sec=24.0,
            min_audio_sec=0.50,
            max_tokens=128,
            silence_commit_sec=1.50,
            silence_hangover_sec=0.45,
            lid_chunk_sec=1.0,
            lid_confirm_chunks=1,
            lid_lock=False,
            unlock_on_utterance=True,
            refine_on_commit=True,
            auto_tune=True,
        )
    if name in ("official", "sdk", "vllm"):
        return StreamConfig(
            hop_sec=2.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            max_audio_sec=0.0,
            min_audio_sec=0.50,
            max_tokens=128,
            silence_commit_sec=1.50,
            lid_chunk_sec=2.0,
            lid_confirm_chunks=1,
            lid_lock=False,
            unlock_on_utterance=True,
            refine_on_commit=True,
            auto_tune=False,
        )
    if name in ("paper",):
        return StreamConfig(
            hop_sec=2.0,
            unfixed_chunk_num=4,
            unfixed_token_num=5,
            max_audio_sec=0.0,
            min_audio_sec=0.50,
            max_tokens=128,
            silence_commit_sec=1.50,
            lid_chunk_sec=2.0,
            lid_confirm_chunks=1,
            lid_lock=False,
            unlock_on_utterance=True,
            refine_on_commit=True,
            auto_tune=False,
        )
    if name in ("balanced", "mid"):
        return StreamConfig(
            hop_sec=1.0,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            max_audio_sec=24.0,
            min_audio_sec=0.50,
            max_tokens=128,
            silence_commit_sec=1.50,
            lid_chunk_sec=1.0,
            lid_confirm_chunks=1,
            lid_lock=False,
            unlock_on_utterance=True,
            refine_on_commit=True,
            auto_tune=False,
        )
    return StreamConfig(
        hop_sec=0.40,
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        max_audio_sec=24.0,
        min_audio_sec=0.50,
        max_tokens=128,
        silence_commit_sec=1.40,
        lid_chunk_sec=1.0,
        lid_confirm_chunks=1,
        lid_lock=False,
        unlock_on_utterance=True,
        refine_on_commit=True,
        auto_tune=False,
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
    silence_hangover_until: float = 0.0
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
    event_is_companion: bool = False
    event_updated_at: float = 0.0
    pann_last_sec: float = 0.0
    pann_cached: object = None
    adapt_hint: str = ""
    tune_hop_sec: float = 0.0
    tune_pause_sec: float = 0.0

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
        self._pann = None
        self._pann_ready = None
        self._refine_gen = 0
        self._refine_thread: Optional[threading.Thread] = None
        self._last_commit_at = 0.0
        self._last_commit_silence = 0.0
        self._tuner = RuntimeTuner.from_config(cfg)
        self._sound_model_live: object = None
        self._sync_pann_model()
        if cfg.language:
            self.state.locked_language = cfg.language
            self.state.language = cfg.language
            self.state.language_status = "forced"

    def _live_hop(self) -> float:
        return self._tuner.hop_sec if self._tuner.enabled else self.cfg.hop_sec

    def _live_pause(self) -> float:
        return (
            self._tuner.silence_commit_sec
            if self._tuner.enabled
            else self.cfg.silence_commit_sec
        )

    def _live_hangover(self) -> float:
        return (
            self._tuner.silence_hangover_sec
            if self._tuner.enabled
            else self.cfg.silence_hangover_sec
        )

    def _live_sound_gate(self) -> bool:
        if self._tuner.enabled and "sound_gate" not in self._tuner.locked:
            return self._tuner.sound_gate
        return self.cfg.sound_gate

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
        st.event_is_companion = False
        st.event_updated_at = 0.0
        st.pann_last_sec = 0.0
        st.pann_cached = None
        st.speech_seen = tail.size > 0
        st.silence_sec = 0.0
        st.silence_hangover_until = 0.0
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

    def _clear_non_speech_lock(self) -> None:
        """Speech resumed after a non-speech tag — do not stay locked."""
        st = self.state
        if not st.non_speech_only and not st.sound_label:
            return
        # Only a hard LAST lock was a real false tag. Live soft hints are normal
        # (cough then talk) and must not demote PANNs.
        if st.non_speech_only:
            self._tuner.record_false_tag()
            self._sync_pann_model()
        st.non_speech_only = False
        st.sound_label = ""
        if st.language_status == "non-speech":
            st.language_status = "mix" if st.utterance_langs else "guessing"
        tag_only = (
            (st.unfixed or st.text or "").strip().startswith("[")
            and "]" in (st.unfixed or st.text or "")
            and not has_lexical_speech(strip_event_prefix(st.unfixed or st.text or ""))
        )
        if tag_only:
            st.unfixed = ""
            st.text = ""

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
        now = time.monotonic()
        if self.vad is not None:
            speaking = self.vad.is_speech(x)
            dt = x.size / float(SAMPLE_RATE)
            if speaking:
                if (
                    not st.speech_seen
                    and self._last_commit_at > 0
                    and (now - self._last_commit_at) < 0.40
                    and self._live_pause() < 1.45
                    and self._last_commit_silence <= (self._live_pause() + 0.15)
                ):
                    self._tuner.record_early_resume()
                st.speech_seen = True
                st.silence_sec = 0.0
                st.silence_hangover_until = now + self._live_hangover()
                self._clear_non_speech_lock()
            else:
                if st.speech_seen and now >= st.silence_hangover_until:
                    st.silence_sec += dt
                if not st.speech_seen:
                    st.speaking = False
                    self._maybe_unlock_on_long_silence()
                    self._emit()
                    return
        else:
            if speaking:
                st.speech_seen = True
                self._clear_non_speech_lock()
        st.speaking = bool(speaking)

        if speaking or st.speech_seen:
            st.buffer = np.concatenate([st.buffer, x], axis=0)

        # LIVE must keep draining during pauses. Suppressing hop decodes while
        # silent is what froze LIVE for ~1s before every commit.
        hop_sec = self._live_hop()
        hop = int(round(hop_sec * SAMPLE_RATE))
        if st.buffer.size >= hop:
            chunk = st.buffer
            st.buffer = np.zeros((0,), dtype=np.float32)
            self._consume_chunk(chunk)

        commit_after = self._live_pause()
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
        self._absorb_buffer()
        return self.state

    def _absorb_buffer(self) -> bool:
        """Move leftover PCM into the utterance without a blocking GPU decode."""
        st = self.state
        if st.buffer.size == 0:
            return False
        if st.audio_accum.size:
            st.audio_accum = np.concatenate([st.audio_accum, st.buffer], axis=0)
        else:
            st.audio_accum = st.buffer
        st.buffer = np.zeros((0,), dtype=np.float32)
        return True

    def _buffer_has_speech(self) -> bool:
        """True when the leftover buffer holds real speech energy, not just silence."""
        st = self.state
        n = st.buffer.size
        if n < int(TAIL_MIN_SEC * SAMPLE_RATE):
            return False
        tail = st.buffer[-int(min(n, 0.6 * SAMPLE_RATE)) :]
        if tail.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(tail), dtype=np.float32)))
        if rms < TAIL_MIN_RMS:
            return False
        if self.vad is not None:
            try:
                if not self.vad.is_speech(tail):
                    # One VAD frame can miss soft endings; fall back to RMS.
                    return rms >= TAIL_MIN_RMS * 2.0
            except Exception:
                pass
        return True

    def _has_undecoded_tail(self) -> bool:
        return self._buffer_has_speech()

    def join_refine(self, timeout: float = 6.0) -> str:
        t = self._refine_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        return self.state.finalized or ""

    def commit(self, wait: bool = False, timeout: float = 6.0) -> str:
        """End utterance: LAST shows the live draft instantly, tail seals behind it."""
        st = self.state
        has_tail = self._has_undecoded_tail()
        self._absorb_buffer()
        draft = self._with_event(stitch_transcript(st.committed, st.text))
        if not st.non_speech_only:
            st.language = merge_languages(st.utterance_langs) or st.language
        # Instant LAST — the mic loop never blocks on the seal pass.
        st.finalized = draft
        st.refining = False
        self._emit()
        utterance_sec = st.audio_accum.size / float(SAMPLE_RATE)
        self._tuner.record_commit(utterance_sec, st.silence_sec)
        self._last_commit_at = time.monotonic()
        self._last_commit_silence = float(st.silence_sec)
        min_n = int(self.cfg.min_audio_sec * SAMPLE_RATE)
        utterance_id = st.utterance_id
        if (
            self._tuner.should_refine(utterance_sec, draft, has_tail=has_tail)
            and st.audio_accum.size >= min_n
            and not st.non_speech_only
        ):
            self._start_refine_background(
                st.audio_accum.copy(),
                draft,
                utterance_id,
            )
        if wait:
            self.join_refine(timeout)
            return self.state.finalized or draft
        return draft

    def _start_refine_background(
        self,
        audio: np.ndarray,
        draft: str,
        utterance_id: int,
    ) -> None:
        self._refine_gen += 1
        gen = self._refine_gen
        st = self.state
        event_label = (
            st.event_label
            if st.event_is_companion and st.event_label and self._event_fresh()
            else ""
        )
        st.refining = True
        self._emit()

        def run() -> None:
            try:
                merged = self._refine_audio_snapshot(audio, draft, gen, event_label)
            except Exception:
                merged = draft
            if gen != self._refine_gen:
                return
            if not (merged or "").strip():
                merged = draft
            st = self.state
            # LAST belongs to the committed utterance: always safe to paint it,
            # even after reset_utterance started the next LIVE line.
            st.finalized = merged
            if st.utterance_id == utterance_id:
                st.refining = False
            else:
                # New utterance already speaking; don't leave a stale SEAL flag.
                self.state.refining = False
            self._emit()

        self._refine_thread = threading.Thread(target=run, daemon=True)
        self._refine_thread.start()

    def _refine_audio_snapshot(
        self,
        audio: np.ndarray,
        draft: str,
        gen: int,
        event_label: str = "",
    ) -> str:
        """Fast LAST seal: prefix the live draft, stream tokens, skip extra PANNs."""
        body = strip_event_prefix(draft)
        lexical = has_lexical_speech(body)
        if not lexical:
            hint = analyze_sound(audio) if self._live_sound_gate() else None
            blocked = self._block_label_stateless(audio, draft, hint)
            if blocked:
                return f"[{blocked}]"
        prefix = ""
        max_tok = SEAL_MAX_TOKENS
        if lexical:
            # Token-exact rollback keeps continuity; heuristic fallback inside.
            prefix = self.client.rollback_prefix(body, self.cfg.unfixed_token_num)
        else:
            max_tok = min(self.cfg.refine_max_tokens, 128)

        def on_partial(_lang: str, text: str) -> None:
            if gen != self._refine_gen:
                return
            _, clean = parse_asr_output(text, user_language=self.cfg.language)
            clean = strip_event_prefix(clean)
            if not clean.strip():
                return
            st = self.state
            st.finalized = clean
            self._emit()

        result = self.client.transcribe(
            audio,
            raw_prefix=prefix,
            context=self.cfg.context,
            force_language=self.cfg.language,
            max_tokens=max_tok,
            temperature=0.01,
            on_partial=on_partial if self.cfg.stream_tokens else None,
        )
        _, clean = parse_asr_output(result.text, user_language=self.cfg.language)
        clean = strip_event_prefix(clean)
        if not lexical:
            hint = analyze_sound(audio) if self._live_sound_gate() else None
            blocked = self._block_label_stateless(audio, clean or draft, hint)
            if blocked:
                return f"[{blocked}]"
        if clean.strip():
            if event_label:
                return combine_event_and_transcript(event_label, clean)
            return clean
        return draft

    def _block_label_stateless(
        self,
        audio: np.ndarray,
        text: str,
        hint: Optional[SoundHint],
    ) -> Optional[str]:
        if has_lexical_speech(text):
            return None
        onom = classify_sound_event_text(text)
        if onom:
            return onom
        if self._pann is not None:
            tag = self._pann.tag(audio)
            if tag is not None and tag.blocks_asr:
                return tag.label
        if not self._live_sound_gate() or hint is None:
            return None
        if (
            should_skip_asr(hint, max(self.cfg.sound_gate_min_conf, 0.78))
            and not hint.is_speech
            and hint.category in ("impulse", "burst", "noise")
        ):
            return hint.label or "suara non-bicara"
        return None

    def _effective_sound_model(self) -> str:
        if self._tuner.enabled and "sound_model" not in self._tuner.locked:
            return self._tuner.sound_model
        return str(self.cfg.sound_model or "auto").strip().lower()

    def _sync_pann_model(self) -> None:
        mode = self._effective_sound_model()
        key = (mode, self._live_sound_gate())
        if key != self._sound_model_live:
            self._sound_model_live = key
            self._pann = self._init_pann(mode)
        self.state.adapt_hint = self._tuner.hint

    def _init_pann(self, mode: Optional[str] = None):
        mode = (mode or self._effective_sound_model()).strip().lower()
        if not self._live_sound_gate() or mode in ("off", "heuristic"):
            return None
        from .pann_cnn6 import get_tagger, pann_available

        if mode in ("auto", "pann", "pann_cnn6", "cnn6") and pann_available():
            if self._pann_ready is not None:
                return self._pann_ready
            tagger = get_tagger(
                min_score=self.cfg.pann_min_score,
                block_score=self.cfg.pann_block_score,
                companion_score=self.cfg.pann_companion_score,
                speech_score=self.cfg.pann_speech_score,
            )
            if tagger is not None:
                tagger.min_score = self.cfg.pann_min_score
                self._pann_ready = tagger
            return tagger
        return None

    def _pann_interval(self) -> float:
        if self._tuner.enabled and "pann_interval_sec" not in self._tuner.locked:
            return self._tuner.pann_interval_sec
        return self.cfg.pann_interval_sec

    def _event_fresh(self) -> bool:
        st = self.state
        return bool(st.event_label) and (time.monotonic() - st.event_updated_at) < EVENT_TTL_SEC

    def _maybe_pann_tag(self, audio: np.ndarray):
        if self._pann is None:
            return None
        st = self.state
        sec = audio.size / float(SAMPLE_RATE)
        if sec < 0.8:
            return None
        # Wall-clock throttle: the old audio-growth check re-ran inference on
        # every hop once the utterance grew, serialising a CNN forward pass
        # into the 400 ms hot path. PANNs is a slow clip tagger; 1.5 s cadence
        # is plenty for display tags.
        now = time.monotonic()
        if st.pann_cached is not None and (now - st.pann_last_sec) < self._pann_interval():
            return st.pann_cached
        tag = self._pann.tag(audio)
        st.pann_last_sec = now
        st.pann_cached = tag
        if tag is not None:
            st.event_label = tag.label
            st.event_score = tag.score
            st.event_top = tag.top_labels
            st.event_is_companion = bool(tag.is_companion)
            st.event_updated_at = now
        elif not self._event_fresh():
            # No event detected and the previous tag expired: clear it so a
            # stale "[musik]" doesn't stick to unrelated speech forever.
            st.event_label = ""
            st.event_score = 0.0
            st.event_top = ()
            st.event_is_companion = False
        return tag

    def _refresh_event_tag(self, audio: np.ndarray) -> None:
        tag = self._maybe_pann_tag(audio)
        if tag is None or not tag.label:
            return
        st = self.state
        st.event_label = tag.label
        st.event_score = tag.score
        st.event_top = tag.top_labels
        st.event_is_companion = bool(tag.is_companion)
        st.event_updated_at = time.monotonic()

    def _with_event(self, text: str) -> str:
        st = self.state
        body = strip_event_prefix(text)
        # Only fresh *companion* tags (music/singing) are prefixed onto real
        # speech. Blocking/informational labels must never rewrite transcript.
        if (
            has_lexical_speech(body)
            and st.event_is_companion
            and self._event_fresh()
            and st.event_label
        ):
            return combine_event_and_transcript(st.event_label, body)
        if body:
            return body
        if st.event_label and self._event_fresh():
            return f"[{st.event_label}]"
        return text

    def _should_block_refine_only(
        self,
        audio: np.ndarray,
        text: str,
        hint: Optional[SoundHint],
    ) -> Optional[str]:
        """LAST/refine only: block when evidence is strong, never on live hops."""
        if has_lexical_speech(text):
            return None
        onom = classify_sound_event_text(text)
        if onom:
            return onom
        tag = self._maybe_pann_tag(audio)
        if tag is not None and tag.blocks_asr:
            return tag.label
        if not self._live_sound_gate():
            return None
        if hint is None:
            hint = analyze_sound(audio)
        # Heuristic alone is too noisy on mic speech; require high confidence
        # and no speech-like energy before sealing LAST.
        if (
            should_skip_asr(hint, max(self.cfg.sound_gate_min_conf, 0.78))
            and not hint.is_speech
            and hint.category in ("impulse", "burst", "noise")
            and not has_lexical_speech(text)
        ):
            return hint.label or "suara non-bicara"
        return None

    def _seal_non_speech(self, label: str, *, live: bool = False) -> str:
        st = self.state
        if live:
            # LIVE: soft hint only — never wipe transcript or lock the session.
            st.sound_label = label
            if not has_lexical_speech(strip_event_prefix(st.unfixed or st.text or "")):
                st.unfixed = f"[{label}]"
            return st.unfixed
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
        self._refresh_event_tag(audio)
        hint = analyze_sound(audio) if self._live_sound_gate() else None
        st.sound_hint = hint
        blocked = self._should_block_refine_only(audio, draft, hint)
        if blocked:
            return self._seal_non_speech(blocked)
        if not self.cfg.refine_on_commit or audio.size < min_n:
            return self._with_event(draft)
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
        _, clean = parse_asr_output(result.text, user_language=st.force_language)
        blocked = self._should_block_refine_only(audio, clean, hint)
        if blocked:
            return self._seal_non_speech(blocked)
        if result.language:
            self._remember_language(result.language)
            if result.language not in st.utterance_langs:
                st.utterance_langs.append(result.language)
        if clean.strip():
            st.last = result
            merged = self._with_event(clean)
            st.text = merged
            st.unfixed = merged
            st.non_speech_only = False
            st.language_status = "mix" if st.utterance_langs else "guessing"
            return merged
        return self._with_event(draft)

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
        self._sync_pann_model()
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
        if self._live_sound_gate():
            st.sound_hint = analyze_sound(audio)
        min_n = int(self.cfg.min_audio_sec * SAMPLE_RATE)
        if audio.size < min_n:
            if not allow_short:
                self._emit()
                return
            audio = np.pad(audio, (0, min_n - int(audio.size)))

        force = st.force_language
        prefix = ""
        clean_prev = strip_event_prefix(
            parse_asr_output(st.raw_decoded, user_language=force or None)[1] or st.text
        )
        if st.chunk_id >= 1 and clean_prev:
            # Token-exact rollback keeps continuity from the second hop.
            # Language bias is handled in infer_languages, not by dropping
            # the prefix (dropping it made every hop rewrite from scratch).
            prefix = self.client.rollback_prefix(clean_prev, self.cfg.unfixed_token_num)

        def on_partial(lang: str, text: str) -> None:
            _, clean = parse_asr_output(text, user_language=force or None)
            if classify_sound_event_text(clean) and not has_lexical_speech(clean):
                st.sound_label = classify_sound_event_text(clean) or ""
                st.unfixed = f"[{st.sound_label}]"
                self._emit()
                return
            st.sound_label = ""
            if not force:
                st.language = lang or st.language
            else:
                st.language = force
            st.unfixed = self._with_event(clean)
            self._emit()

        st.decoding = True
        if not force and not has_lexical_speech(st.unfixed or st.text):
            st.language_status = "confirming" if st.lid_votes else "guessing"
        elif not force:
            st.language_status = "mix"
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

        _, clean = parse_asr_output(result.text, user_language=force or None)
        if has_lexical_speech(clean):
            st.sound_label = ""
            st.non_speech_only = False
        elif classify_sound_event_text(clean) and not has_lexical_speech(clean):
            # LIVE hop: show tag but keep listening — do not lock session.
            self._seal_non_speech(classify_sound_event_text(clean) or "suara non-bicara", live=True)
            st.chunk_id += 1
            self._emit()
            return

        st.last = result
        st.raw_decoded = result.text
        merged = self._with_event(clean)
        st.text = merged
        st.unfixed = merged
        st.non_speech_only = False
        st.chunk_id += 1
        self._tuner.record_decode(result)
        self._sync_pann_model()
        # PANNs after ASR so tagging never blocks the live decode path.
        if st.chunk_id % 2 == 0 and self._pann is not None:
            self._refresh_event_tag(audio)

        if force:
            st.language = force
            st.language_status = "forced" if self.cfg.language else "locked"
        else:
            guessed = (result.language or "").strip()
            inferred = infer_languages(clean, guessed)
            for name in inferred:
                if not name or name.lower() == "none":
                    continue
                st.lid_votes.append(name)
                if name not in st.utterance_langs:
                    st.utterance_langs.append(name)
                self._remember_language(name)
            st.lid_rounds += 1
            if self.cfg.lid_lock and len(inferred) <= 1:
                self._maybe_lock_language()
            else:
                st.language = merge_languages(st.utterance_langs) or guessed
                st.language_status = (
                    "mix"
                    if len(st.utterance_langs) > 1
                    or has_lexical_speech(st.unfixed or st.text)
                    else "guessing"
                )

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
        if last == "English" and audio_sec < 2.0:
            st.language_status = "confirming"
            return
        if len(st.lid_votes) >= 2 and st.lid_votes[-1] != st.lid_votes[-2]:
            st.language_status = "confirming"
            return
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
        st = self.state
        st.tune_hop_sec = self._live_hop()
        st.tune_pause_sec = self._live_pause()
        st.adapt_hint = self._tuner.hint if self._tuner.enabled else ""
        if self.on_update is not None:
            self.on_update(st)
