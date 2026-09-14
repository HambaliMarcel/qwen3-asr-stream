"""Publish live STT to a localhost JSONL bus without changing the ASR engine.

Run beside the existing mic dashboard:

    python -m qwen3_asr_stream.integrator

The core stream/client/ui modules are imported as-is. Brain clients connect to
127.0.0.1:18765 (one JSON object per line).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from typing import Optional

from .__main__ import _apply_cli_overrides, _connect, _url, build_parser
from .audio import MicStream
from .parse import has_lexical_speech, strip_event_prefix
from .stream import StreamState, StreamingAsr, profile_config
from .ui import LiveTranscript


DEFAULT_BUS_PORT = 18765
COMMIT_SETTLE_SEC = 0.08


class JsonlHub:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_BUS_PORT):
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._alive = False

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(16)
        sock.settimeout(0.5)
        self._sock = sock
        self._alive = True
        threading.Thread(target=self._accept, name="asr-bus-accept", daemon=True).start()

    def close(self) -> None:
        self._alive = False
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for c in clients:
            try:
                c.close()
            except OSError:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def publish(self, payload: dict) -> None:
        data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            clients = list(self._clients)
        dead: list[socket.socket] = []
        for c in clients:
            try:
                c.sendall(data)
            except OSError:
                dead.append(c)
        if dead:
            with self._lock:
                self._clients = [c for c in self._clients if c not in dead]
            for c in dead:
                try:
                    c.close()
                except OSError:
                    pass

    def _accept(self) -> None:
        assert self._sock is not None
        while self._alive:
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            with self._lock:
                self._clients.append(conn)


def _event_from_raw(raw: str) -> str:
    t = (raw or "").strip()
    if t.startswith("[") and "]" in t:
        inner = t[1 : t.index("]")].strip()
        if inner:
            return inner
    return ""


def _event_label(st: StreamState) -> str:
    tagged = str(getattr(st, "event_label", "") or getattr(st, "sound_label", "") or "").strip()
    if tagged:
        return tagged
    return _event_from_raw(st.unfixed or st.text or "")


def _event_payload(st: StreamState) -> dict:
    event = _event_label(st)
    top = getattr(st, "event_top", ()) or ()
    return {
        "event": event,
        "event_score": float(getattr(st, "event_score", 0.0) or 0.0),
        "event_top": [str(x) for x in top[:4]],
        "companion": bool(getattr(st, "event_is_companion", False)),
        "non_speech_only": bool(getattr(st, "non_speech_only", False)),
        "sound_label": str(getattr(st, "sound_label", "") or ""),
    }


class SttPublisher:
    """Map StreamingAsr on_update → live/commit/sound JSONL events."""

    def __init__(self, hub: JsonlHub, ui: Optional[LiveTranscript] = None):
        self.hub = hub
        self.ui = ui
        self._last_live = ""
        self._last_commit = ""
        self._last_commit_uid = -1
        self._last_event = ""
        self._last_emit = 0.0
        self._commit_lock = threading.Lock()
        self._pending_commit: Optional[dict] = None
        self._commit_seq = 0
        self._refining_uid = -1

    def _queue_commit(self, payload: dict, *, refining: bool) -> None:
        """Publish only the settled LAST, not the pre-refine draft seal."""
        uid = int(payload.get("utterance_id") or 0)
        with self._commit_lock:
            self._pending_commit = payload
            self._commit_seq += 1
            seq = self._commit_seq
            if refining:
                self._refining_uid = uid
                return
            refined = self._refining_uid == uid
            if refined:
                self._refining_uid = -1
        if refined:
            self._flush_commit(seq)
            return
        timer = threading.Timer(COMMIT_SETTLE_SEC, self._flush_commit, args=(seq,))
        timer.daemon = True
        timer.start()

    def _flush_commit(self, seq: int) -> None:
        with self._commit_lock:
            if seq != self._commit_seq or self._pending_commit is None:
                return
            payload = self._pending_commit
            self._pending_commit = None
        text = str(payload.get("text") or "").strip()
        uid = int(payload.get("utterance_id") or 0)
        if not text or (uid == self._last_commit_uid and text == self._last_commit):
            return
        self._last_commit = text
        self._last_commit_uid = uid
        self.hub.publish(payload)

    def on_update(self, st: StreamState) -> None:
        finalized = st.finalized or ""
        raw = (st.unfixed or st.text or "").strip()
        live = strip_event_prefix(raw)
        event = _event_label(st)
        if not live and event:
            live = f"[{event}]"
        display = raw if raw else (f"[{event}]" if event else "")
        now = time.monotonic()
        extras = _event_payload(st)
        if live != self._last_live or event != self._last_event or (now - self._last_emit) >= 0.04:
            self._last_live = live
            self._last_emit = now
            self.hub.publish(
                {
                    "v": 1,
                    "type": "live",
                    "text": live,
                    "display": display,
                    "language": st.language or "",
                    "speaking": bool(st.speaking),
                    "decoding": bool(st.decoding),
                    "utterance_id": int(st.utterance_id),
                    "gap_sec": float(st.gap_sec),
                    "silence_sec": float(st.silence_sec),
                    "ts": time.time(),
                    **extras,
                }
            )
        if event and event != self._last_event:
            self._last_event = event
            self.hub.publish(
                {
                    "v": 1,
                    "type": "sound",
                    "text": f"[{event}]",
                    "display": display or f"[{event}]",
                    "language": st.language or "",
                    "speaking": bool(st.speaking),
                    "decoding": bool(st.decoding),
                    "utterance_id": int(st.utterance_id),
                    "gap_sec": float(st.gap_sec),
                    "silence_sec": float(st.silence_sec),
                    "ts": time.time(),
                    **extras,
                }
            )
        elif not event:
            self._last_event = ""
        if finalized:
            text = strip_event_prefix(finalized)
            if has_lexical_speech(text):
                self._queue_commit(
                    {
                        "v": 1,
                        "type": "commit",
                        "text": text,
                        "display": finalized.strip() or text,
                        "language": st.language or "",
                        "speaking": False,
                        "decoding": False,
                        "utterance_id": int(st.utterance_id),
                        "ts": time.time(),
                        **extras,
                    },
                    refining=bool(getattr(st, "refining", False)),
                )
        if self.ui is not None:
            self.ui.render(st)
        else:
            st.finalized = ""


def main(argv: list[str] | None = None) -> int:
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--bus-host", default="127.0.0.1")
    own.add_argument("--bus-port", type=int, default=DEFAULT_BUS_PORT)
    own.add_argument("--no-ui", action="store_true", help="Publish only; no ASR dashboard")
    own.add_argument("-h", "--help", action="store_true")
    extra, rest = own.parse_known_args(argv)
    if extra.help:
        print("Publish live Qwen3-ASR to a localhost JSONL bus.")
        print("  python -m qwen3_asr_stream.integrator [--bus-port 18765] [--no-ui] [mic flags...]")
        print()
        print("All `python -m qwen3_asr_stream mic` flags are accepted.")
        return 0

    mic_args = build_parser().parse_args(["mic", *rest])
    cfg = _apply_cli_overrides(profile_config(mic_args.profile), mic_args)
    client = _connect(mic_args)
    hub = JsonlHub(extra.bus_host, extra.bus_port)
    hub.start()
    ui = None if extra.no_ui else LiveTranscript()
    pub = SttPublisher(hub, ui)
    engine = StreamingAsr(client, cfg, on_update=pub.on_update)
    lang_label = cfg.language or (
        "MIX · multilingual" if not cfg.lid_lock else "LOCK · one language per sentence"
    )
    sealed = ""
    try:
        mode = "auto-tune" if cfg.auto_tune else "fixed"
        banner = (
            f"{_url(mic_args.host, mic_args.port)}   bus {extra.bus_host}:{extra.bus_port}   "
            f"{mic_args.profile} · {mode}   hop {cfg.hop_sec:.2f}s   {lang_label}"
        )
        if ui is not None:
            ui.banner("Qwen3-ASR  ·  live bus → brain", banner)
        else:
            print(banner, flush=True)
        with MicStream(device=mic_args.device, block_ms=20) as mic:
            while True:
                pcm = mic.read(timeout=0.15)
                if pcm.size:
                    engine.push(pcm)
    except KeyboardInterrupt:
        sealed = engine.commit(wait=True) or engine.state.text or engine.state.unfixed
    finally:
        if ui is not None:
            ui.close(sealed)
        hub.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
