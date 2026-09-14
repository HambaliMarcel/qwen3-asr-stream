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
# LAST seal: one full re-decode of the finished window (no prefix), in the
# background, so early frozen-prefix mistakes get a second look.
SEAL_MAX_TOKENS = 128
# Minimum speech-like energy in the leftover buffer to count as an undecoded tail.
TAIL_MIN_SEC = 0.08
TAIL_MIN_RMS = 0.006
# LIVE → LAST batching. A window this long is sealed into LAST at the next
# breath gap, and LIVE starts a fresh window — no overlap, so no echo.
CUT_MIN_SEC = 6.0
CUT_GAP_SEC = 0.35
# No breath gap at all (fast rap): cut anyway so the window never grows past
# what the model decodes in ~200 ms.
HARD_CUT_SEC = 16.0
# Between the two: a long window may cut on any short energy dip (word
# boundary) instead of waiting for a real breath or splitting a word at 16 s.
MID_CUT_SEC = 11.0
MID_CUT_GAP_SEC = 0.15
# A window that has produced no words yet (beat intro, hush the VAD let
# through) is trimmed to this much audio so `language None` can never lock it.
IDLE_TRIM_SEC = 1.0

from .audio import SAMPLE_RATE
from .client import DecodeResult, LlamaAsrClient
from .parse import (
    classify_sound_event_text,
    combine_event_and_transcript,
    has_lexical_speech,
    infer_languages,
    join_segments,
    merge_languages,
    parse_asr_output,
    prefer_transcript,
    strip_event_prefix,
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
            hop_sec=0.6,
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
    silence_flushed: bool = False
    last_speech_at: float = 0.0
    # Audio-time since the last speech frame (unlike silence_sec, not gated by
    # the syllable hangover) — drives LIVE→LAST batching at breath gaps.
    gap_sec: float = 0.0
    # Language tag the model itself emitted for the current window; used to
    # keep the continuation prefill in the model's own format.
    window_lang: str = ""
    segments: int = 0
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
        # `refining` is owned by the seal thread (see _seal_alive), so the
        # LAST · sealing label survives the reset into the next LIVE line.
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
        st.silence_flushed = False
        st.last_speech_at = 0.0
        st.gap_sec = 0.0
        st.window_lang = ""
        st.segments = 0
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
                st.silence_flushed = False
                st.last_speech_at = now
                st.gap_sec = 0.0
                self._clear_non_speech_lock()
            else:
                # Audio-time, not wall-clock: file mode and a busy decode
                # thread must not stretch or shrink the pause.
                st.gap_sec += dt
                if st.speech_seen and st.gap_sec >= self._live_hangover():
                    st.silence_sec += dt
                if not st.speech_seen:
                    st.speaking = False
                    self._maybe_unlock_on_long_silence()
                    self._emit()
                    return
        else:
            if speaking:
                st.speech_seen = True
                st.last_speech_at = now
                self._clear_non_speech_lock()
        st.speaking = bool(speaking)

        hangover = st.speech_seen and not speaking and st.gap_sec < self._live_hangover()
        if speaking or hangover:
            st.buffer = np.concatenate([st.buffer, x], axis=0)
        elif st.speech_seen and not st.silence_flushed:
            # Breath gap confirmed. Decode the leftover once (it holds the end
            # of the last word) and then stop feeding silence to the GPU —
            # silence hops only re-emit the line and stall the mic loop.
            st.silence_flushed = True
            self._flush_buffer_decode()

        hop_sec = self._live_hop()
        hop = int(round(hop_sec * SAMPLE_RATE))
        if st.chunk_id == 0 and st.audio_accum.size == 0:
            # First words appear after ~0.5 s, not a full hop.
            hop = min(hop, int(0.5 * SAMPLE_RATE))
        if (speaking or hangover) and st.buffer.size >= hop:
            chunk = st.buffer
            st.buffer = np.zeros((0,), dtype=np.float32)
            self._consume_chunk(chunk)

        # Batching: a long window is moved to LAST at the first breath gap and
        # LIVE keeps going in a fresh window. LAST grows while the speaker
        # never really stops (long rap), and each hop stays ~200 ms.
        if self.vad is not None and st.speech_seen and not speaking:
            win = st.audio_accum.size + st.buffer.size
            breath = st.silence_flushed and st.gap_sec >= max(CUT_GAP_SEC, self._live_hangover())
            dip = st.gap_sec >= MID_CUT_GAP_SEC and win >= int(MID_CUT_SEC * SAMPLE_RATE)
            if (
                (breath and win >= int(CUT_MIN_SEC * SAMPLE_RATE)) or dip
            ) and has_lexical_speech(strip_event_prefix(st.text)):
                self._cut_window()

        commit_after = self._live_pause()
        if (
            self.vad is not None
            and st.speech_seen
            and st.silence_sec >= commit_after
        ):
            live = strip_event_prefix(st.unfixed or st.text or st.committed or "")
            if has_lexical_speech(live) or st.non_speech_only or st.committed:
                self.commit()
                unlock = bool(self.cfg.unlock_on_utterance) and not self.cfg.language
                self.reset_utterance(keep_tail=False, unlock=unlock)
            elif st.silence_sec >= commit_after + 0.6:
                # False VAD start (hush decoded as language none) — drop it.
                unlock = bool(self.cfg.unlock_on_utterance) and not self.cfg.language
                self.reset_utterance(keep_tail=False, unlock=unlock)
            self._emit()

    def _cut_window(self) -> None:
        """Seal the current LIVE window into LAST and start a fresh one.

        No audio is carried over, so the next window can never re-transcribe
        (echo) words that are already in LAST.
        """
        st = self.state
        self._flush_buffer_decode()
        seg = strip_event_prefix(st.text or "")
        if has_lexical_speech(seg):
            if self._seal_alive():
                # An older seal must not paint its paragraph over this one.
                self._refine_gen += 1
            st.committed = join_segments(st.committed, seg)
            st.finalized = self._with_event(st.committed)
            st.segments += 1
        st.audio_accum = np.zeros((0,), dtype=np.float32)
        st.raw_decoded = ""
        st.text = ""
        st.unfixed = ""
        st.chunk_id = 0
        st.window_lang = ""
        st.pann_cached = None
        st.pann_last_sec = 0.0
        self._emit()

    def _flush_buffer_decode(self) -> None:
        """Decode whatever is left in the hop buffer (end of the last word)."""
        st = self.state
        if st.buffer.size == 0:
            return
        chunk = st.buffer
        st.buffer = np.zeros((0,), dtype=np.float32)
        if st.audio_accum.size == 0 and not self._buffer_has_speech_in(chunk):
            return
        self._consume_chunk(chunk, allow_short=True)

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
        return self._buffer_has_speech_in(self.state.buffer)

    def _buffer_has_speech_in(self, buf: np.ndarray) -> bool:
        n = buf.size
        if n < int(TAIL_MIN_SEC * SAMPLE_RATE):
            return False
        tail = buf[-int(min(n, 0.6 * SAMPLE_RATE)) :]
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

    def join_refine(self, timeout: float = 6.0) -> str:
        t = self._refine_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        return self.state.finalized or ""

    def _seal_alive(self) -> bool:
        t = self._refine_thread
        alive = t is not None and t.is_alive()
        if not alive and self.state.refining:
            self.state.refining = False
        return alive

    def commit(self, wait: bool = False, timeout: float = 6.0) -> str:
        """End utterance: LAST shows the live draft instantly, seal re-checks it behind."""
        st = self.state
        # Last word may still be in the buffer: one short decode (~150 ms).
        self._flush_buffer_decode()
        head = st.committed
        seg = strip_event_prefix(st.text or "")
        draft = self._with_event(join_segments(head, seg))
        if not st.non_speech_only:
            st.language = merge_languages(st.utterance_langs) or st.language
        lexical = has_lexical_speech(strip_event_prefix(draft))
        if not lexical and not st.non_speech_only:
            # Hush / `language None` is not a line — keep LAST as it was.
            st.refining = False
            self._emit()
            return ""
        st.finalized = draft
        st.refining = False
        self._emit()
        utterance_sec = st.audio_accum.size / float(SAMPLE_RATE)
        self._tuner.record_commit(utterance_sec, st.silence_sec)
        self._last_commit_at = time.monotonic()
        self._last_commit_silence = float(st.silence_sec)
        min_n = int(self.cfg.min_audio_sec * SAMPLE_RATE)
        if self._seal_alive():
            # Older seal still in flight — it must not paint over this LAST.
            self._refine_gen += 1
        if (
            has_lexical_speech(seg)
            and self._tuner.should_refine(utterance_sec, seg, has_tail=False)
            and st.audio_accum.size >= min_n
            and not st.non_speech_only
        ):
            self._start_refine_background(st.audio_accum.copy(), head, seg)
        if wait:
            self.join_refine(timeout)
            return self.state.finalized or draft
        return draft

    def _start_refine_background(self, audio: np.ndarray, head: str, seg: str) -> None:
        """Full re-decode of the last window in a thread.

        Live hops are never held back for it: the slot is idle anyway during
        the pause that triggered the commit, and the result only replaces the
        last segment of LAST when it is at least as complete as the draft.
        """
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

        def paint(seg_text: str) -> None:
            body = strip_event_prefix(seg_text)
            merged = join_segments(head, prefer_transcript(seg, body))
            if event_label:
                merged = combine_event_and_transcript(event_label, merged)
            self.state.finalized = merged

        def run() -> None:
            try:
                sealed = self._refine_audio_snapshot(audio, seg, gen, paint)
            except Exception:
                sealed = seg
            if gen != self._refine_gen:
                return
            paint(sealed or seg)
            self.state.refining = False
            self._emit()

        self._refine_thread = threading.Thread(target=run, daemon=True)
        self._refine_thread.start()

    def _refine_audio_snapshot(
        self,
        audio: np.ndarray,
        seg: str,
        gen: int,
        paint: Callable[[str], None],
    ) -> str:
        """LAST seal: one full-window decode (no prefix) of the finished segment."""
        body = strip_event_prefix(seg)
        max_tok = self._token_budget(audio, SEAL_MAX_TOKENS, 512)

        def on_partial(_lang: str, text: str) -> None:
            if gen != self._refine_gen:
                return
            _, clean = parse_asr_output(text, user_language=self.cfg.language)
            clean = strip_event_prefix(clean)
            if not clean.strip():
                return
            dw, cw = body.split(), clean.split()
            if len(cw) < max(4, int(len(dw) * 0.72)):
                return
            paint(clean)
            self._emit()

        result = self.client.transcribe(
            audio,
            raw_prefix="",
            context=self.cfg.context,
            force_language=self.cfg.language,
            max_tokens=max_tok,
            temperature=0.01,
            on_partial=on_partial if self.cfg.stream_tokens else None,
        )
        _, clean = parse_asr_output(result.text, user_language=self.cfg.language)
        clean = strip_event_prefix(clean)
        if has_lexical_speech(clean):
            return clean
        return seg

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

    def _token_budget(self, audio: np.ndarray, base: int, cap: int) -> int:
        """Grow max_tokens with clip length so long rap is not cut off."""
        sec = float(getattr(audio, "size", 0) or 0) / float(SAMPLE_RATE)
        return min(cap, max(int(base), 48 + int(sec * 16)))

    def _consume_chunk(self, chunk: np.ndarray, allow_short: bool = False) -> None:
        st = self.state
        self._sync_pann_model()
        if st.audio_accum.size == 0:
            st.audio_accum = chunk
        else:
            st.audio_accum = np.concatenate([st.audio_accum, chunk], axis=0)
        self._seal_alive()

        if not self._ready_for_decode(allow_short):
            if not st.force_language:
                st.language_status = "waiting" if not st.lid_votes else "guessing"
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
        # Official streaming: the window's fixed text (minus the last K
        # tokens) is prefilled and the model only writes the new tail. That
        # is what makes a hop ~150 ms instead of a full re-transcription.
        prefix = ""
        prev_clean = strip_event_prefix(st.text or "")
        if st.chunk_id >= 1 and has_lexical_speech(prev_clean):
            prefix = self.client.rollback_prefix(prev_clean, self.cfg.unfixed_token_num)

        def on_partial(lang: str, text: str) -> None:
            _, clean = parse_asr_output(text, user_language=force or None)
            if classify_sound_event_text(clean) and not has_lexical_speech(clean):
                st.sound_label = classify_sound_event_text(clean) or ""
                st.unfixed = f"[{st.sound_label}]"
                self._emit()
                return
            if not clean.strip() or not has_lexical_speech(clean):
                return
            if prefix and len(clean) < len(prev_clean) and prev_clean.startswith(clean):
                # First streamed delta is just the rolled-back prefill —
                # repainting it would make the last words blink every hop.
                return
            st.sound_label = ""
            if not force:
                st.language = lang or st.language
            else:
                st.language = force
            st.unfixed = self._with_event(clean)
            self._emit()

        st.decoding = True
        if not force:
            st.language_status = (
                "mix" if has_lexical_speech(st.unfixed or st.text) else "guessing"
            )
        self._emit()
        try:
            result = self.client.transcribe(
                audio,
                raw_prefix=prefix,
                context=self.cfg.context,
                force_language=force,
                max_tokens=self._token_budget(audio, self.cfg.max_tokens, 384),
                temperature=self.cfg.temperature,
                on_partial=on_partial if self.cfg.stream_tokens else None,
                prefill_language=st.window_lang or None,
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
        else:
            # `language None` / empty. If this window has no words yet, keep
            # only a short tail so a beat intro or hush never becomes a 10 s
            # window the model keeps calling non-speech.
            st.last = result
            self._tuner.record_decode(result)
            if not has_lexical_speech(strip_event_prefix(st.text or "")):
                keep = int(IDLE_TRIM_SEC * SAMPLE_RATE)
                if st.audio_accum.size > keep:
                    st.audio_accum = st.audio_accum[-keep:]
                st.chunk_id = 0
                st.raw_decoded = ""
                if self._pann is not None:
                    self._refresh_event_tag(audio)
                    if st.event_label and self._event_fresh():
                        st.unfixed = f"[{st.event_label}]"
            else:
                st.chunk_id += 1
            self._emit()
            return

        st.last = result
        st.raw_decoded = result.text
        if (result.language or "").strip().lower() not in {"", "none"}:
            st.window_lang = result.language.strip()
        merged = self._with_event(clean)
        if st.text and prefix:
            # Continuation must never come back shorter than what LIVE
            # already showed (token cap / early EOS).
            merged = prefer_transcript(st.text, merged)
        st.text = merged
        st.unfixed = merged
        st.non_speech_only = False
        st.chunk_id += 1
        self._tuner.record_decode(result)
        self._sync_pann_model()
        # PANNs after ASR so tagging never blocks the live decode path.
        if st.chunk_id % 2 == 0 and self._pann is not None:
            self._refresh_event_tag(audio)
        hard_cut = self.cfg.max_audio_sec if self.cfg.max_audio_sec > 0 else HARD_CUT_SEC
        if st.audio_accum.size >= int(hard_cut * SAMPLE_RATE):
            self._cut_window()
            return

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
        if lang and lang.lower() != "none" and lang not in self.state.languages_seen:
            self.state.languages_seen.append(lang)

    def _emit(self) -> None:
        st = self.state
        st.tune_hop_sec = self._live_hop()
        st.tune_pause_sec = self._live_pause()
        st.adapt_hint = self._tuner.hint if self._tuner.enabled else ""
        if self.on_update is not None:
            self.on_update(st)
