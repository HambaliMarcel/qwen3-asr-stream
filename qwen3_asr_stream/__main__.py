"""CLI: python -m qwen3_asr_stream <mic|file|serve|devices>"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from .audio import SAMPLE_RATE, MicStream, list_input_devices, load_wav_file
from .client import LlamaAsrClient, LlamaServerError
from .parse import SUPPORTED_LANGUAGES, canonicalize_language
from .server import DEFAULT_PORT, build_server_cmd, ensure_server, start_server
from .stream import StreamingAsr, StreamConfig, profile_config
from .ui import LiveTranscript


def _url(host: str, port: int) -> str:
    if host.startswith("http://") or host.startswith("https://"):
        return host
    return f"http://{host}:{port}"


def _lock(cfg: StreamConfig, field: str) -> None:
    cfg.manual_fields.add(field)


def _apply_cli_overrides(cfg: StreamConfig, args: argparse.Namespace) -> StreamConfig:
    if args.hop is not None:
        cfg.hop_sec = float(args.hop)
        _lock(cfg, "hop_sec")
    if args.unfixed_chunks is not None:
        cfg.unfixed_chunk_num = int(args.unfixed_chunks)
    if args.unfixed_tokens is not None:
        cfg.unfixed_token_num = int(args.unfixed_tokens)
    if args.max_audio is not None:
        cfg.max_audio_sec = float(args.max_audio)
    if args.max_tokens is not None:
        cfg.max_tokens = int(args.max_tokens)
    if getattr(args, "language", None) is not None:
        raw_lang = str(args.language).strip().lower()
        cfg.language = canonicalize_language(args.language)
        if raw_lang in {"mix", "mixed", "campuran", "multi", "auto", "detect"}:
            # Auto/mix = multilingual. Never force `language English<asr_text>`.
            # Official one-lang lock is opt-in via --lid-lock.
            cfg.lid_lock = False
    if getattr(args, "lid_lock_on", False):
        cfg.lid_lock = True
        if getattr(args, "lid_wait", None) is None:
            cfg.lid_chunk_sec = max(cfg.lid_chunk_sec, 2.0)
        if getattr(args, "lid_confirm", None) is None:
            cfg.lid_confirm_chunks = max(cfg.lid_confirm_chunks, 2)
    if args.context:
        cfg.context = args.context
    if args.no_vad:
        cfg.vad = False
    if getattr(args, "silence_commit", None) is not None:
        cfg.silence_commit_sec = float(args.silence_commit)
        _lock(cfg, "silence_commit_sec")
    if getattr(args, "silence_hangover", None) is not None:
        cfg.silence_hangover_sec = float(args.silence_hangover)
        _lock(cfg, "silence_hangover_sec")
    if getattr(args, "no_refine", False):
        cfg.refine_on_commit = False
        _lock(cfg, "refine_on_commit")
    if getattr(args, "no_auto_tune", False):
        cfg.auto_tune = False
    if args.no_token_stream:
        cfg.stream_tokens = False
    if getattr(args, "lid_wait", None) is not None:
        cfg.lid_chunk_sec = float(args.lid_wait)
    if getattr(args, "lid_confirm", None) is not None:
        cfg.lid_confirm_chunks = int(args.lid_confirm)
    if getattr(args, "relid_after", None) is not None:
        cfg.relid_silence_sec = float(args.relid_after)
    if getattr(args, "no_lid_lock", False):
        cfg.lid_lock = False
    if getattr(args, "keep_lock", False):
        cfg.unlock_on_utterance = False
    if getattr(args, "no_sound_gate", False):
        cfg.sound_gate = False
        _lock(cfg, "sound_gate")
    if getattr(args, "sound_gate_conf", None) is not None:
        cfg.sound_gate_min_conf = float(args.sound_gate_conf)
    if getattr(args, "sound_model", None) is not None:
        cfg.sound_model = str(args.sound_model).strip().lower()
        _lock(cfg, "sound_model")
    if getattr(args, "pann_interval", None) is not None:
        cfg.pann_interval_sec = float(args.pann_interval)
        _lock(cfg, "pann_interval_sec")
    if getattr(args, "pann_min_score", None) is not None:
        cfg.pann_min_score = float(args.pann_min_score)
    if getattr(args, "pann_block_score", None) is not None:
        cfg.pann_block_score = float(args.pann_block_score)
    if getattr(args, "pann_companion_score", None) is not None:
        cfg.pann_companion_score = float(args.pann_companion_score)
    if getattr(args, "pann_speech_score", None) is not None:
        cfg.pann_speech_score = float(args.pann_speech_score)
    return cfg


def cmd_devices(_: argparse.Namespace) -> int:
    print("Input devices:")
    for idx, name, sr in list_input_devices():
        print(f"  [{idx:3d}] {name}  ({sr} Hz)")
    return 0


def cmd_languages(_: argparse.Namespace) -> int:
    print("Default is multilingual auto-detect (no language is forced).")
    print("Force one language with --language Indonesian (or English, Chinese, ...).")
    print("Official one-language-per-sentence lock: --lid-lock")
    print("Supported:")
    for name in SUPPORTED_LANGUAGES:
        print(f"  {name}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    cmd = build_server_cmd(
        port=args.port,
        llama_server=args.llama_server,
        model=args.model,
        mmproj=args.mmproj,
        ngl=args.ngl,
        ctx=args.ctx,
    )
    print("Starting llama-server:")
    print(" ", " ".join(cmd))
    proc = start_server(
        port=args.port,
        llama_server=args.llama_server,
        model=args.model,
        mmproj=args.mmproj,
        ngl=args.ngl,
        ctx=args.ctx,
        new_console=not args.same_console,
    )
    if args.same_console:
        return int(proc.wait() or 0)
    url = _url(args.host, args.port)
    LlamaAsrClient(url).wait_until_ready(timeout=240.0)
    print(f"Ready at {url}  (PID {proc.pid})")
    print("Leave this process running, or close the server console to stop.")
    try:
        while proc.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        proc.terminate()
    return 0


def _connect(args: argparse.Namespace) -> LlamaAsrClient:
    url = _url(args.host, args.port)
    if args.start_server or args.serve:
        parsed = urlparse(url)
        ensure_server(
            url,
            start=True,
            port=parsed.port or args.port,
            llama_server=args.llama_server,
            model=args.model,
            mmproj=args.mmproj,
            ngl=args.ngl,
            ctx=args.ctx,
        )
    else:
        client = LlamaAsrClient(url)
        if not client.health():
            raise LlamaServerError(
                f"No llama-server at {url}. Run `python -m qwen3_asr_stream serve` "
                "in another terminal, or add --start-server."
            )
        return client
    return LlamaAsrClient(url)


def cmd_mic(args: argparse.Namespace) -> int:
    cfg = _apply_cli_overrides(profile_config(args.profile), args)
    client = _connect(args)
    ui = LiveTranscript()
    engine = StreamingAsr(client, cfg, on_update=ui.render)
    lang_label = cfg.language or (
        "MIX · multilingual" if not cfg.lid_lock else "LOCK · one language per sentence"
    )
    sealed = ""
    try:
        mode = "auto-tune" if cfg.auto_tune else "fixed"
        ui.banner(
            "Qwen3-ASR  ·  live multilingual stream",
            f"{_url(args.host, args.port)}   {args.profile} · {mode}   hop {cfg.hop_sec:.2f}s   {lang_label}",
        )
        with MicStream(device=args.device, block_ms=20) as mic:
            while True:
                pcm = mic.read(timeout=0.15)
                if pcm.size:
                    engine.push(pcm)
    except KeyboardInterrupt:
        sealed = engine.commit(wait=True) or engine.state.text or engine.state.unfixed
    finally:
        ui.close(sealed)
    return 0


def cmd_file(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 2
    pcm = load_wav_file(str(path))
    cfg = _apply_cli_overrides(profile_config(args.profile), args)
    if args.no_vad:
        cfg.vad = False
    else:
        # File tests should not drop quiet speech; default VAD off unless asked.
        if not args.vad:
            cfg.vad = False
    client = _connect(args)
    ui = LiveTranscript()
    engine = StreamingAsr(client, cfg, on_update=ui.render)
    mode = "auto-tune" if cfg.auto_tune else "fixed"
    ui.banner(
        "Qwen3-ASR  ·  file stream",
        f"{path.name}  {pcm.size / SAMPLE_RATE:.1f}s  {args.profile} · {mode}  hop={cfg.hop_sec:.2f}s",
    )
    hop = int(round(cfg.hop_sec * SAMPLE_RATE))
    pos = 0
    t0 = time.perf_counter()
    sealed = ""
    try:
        while pos < pcm.size:
            end = min(pcm.size, pos + hop)
            chunk = pcm[pos:end]
            pos = end
            if args.realtime:
                target = t0 + (pos / SAMPLE_RATE)
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            engine.push(chunk)
        sealed = engine.commit(wait=True)
    except KeyboardInterrupt:
        sealed = engine.commit(wait=True)
    finally:
        ui.close(sealed)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwen3_asr_stream",
        description="Realtime streaming STT for Qwen3-ASR using llama.cpp (no vLLM).",
    )
    p.add_argument("--host", default=os.environ.get("QWEN_ASR_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("QWEN_ASR_PORT", DEFAULT_PORT)))
    p.add_argument("--llama-server", default=os.environ.get("LLAMA_SERVER"))
    p.add_argument("--model", default=os.environ.get("QWEN_ASR_MODEL"))
    p.add_argument("--mmproj", default=os.environ.get("QWEN_ASR_MMPROJ"))
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--start-server", "--serve-with", dest="start_server", action="store_true")
    p.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)

    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Start llama-server with low-latency ASR flags")
    s.add_argument("--same-console", action="store_true", help="Do not open a second console window")
    s.set_defaults(func=cmd_serve)

    d = sub.add_parser("devices", help="List microphone devices")
    d.set_defaults(func=cmd_devices)

    lg = sub.add_parser("languages", help="List the 30 languages Qwen3-ASR can detect")
    lg.set_defaults(func=cmd_languages)

    def add_stream_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--profile",
            choices=("auto", "ultralow", "balanced", "official", "paper"),
            default="auto",
            help="auto (default) adapts hop/pause/refine/tags while running",
        )
        sp.add_argument("--hop", type=float, default=None, help="Audio hop seconds (default from profile)")
        sp.add_argument("--unfixed-chunks", type=int, default=None)
        sp.add_argument("--unfixed-tokens", type=int, default=None)
        sp.add_argument("--max-audio", type=float, default=None, help="Rolling window seconds, 0 = grow forever")
        sp.add_argument("--max-tokens", type=int, default=None)
        sp.add_argument(
            "--language",
            default="mix",
            help="mix/auto (default): multilingual, no forced language. Or force Indonesian/English/...",
        )
        sp.add_argument("--lid-wait", type=float, default=None, help="Seconds of speech before first LID (official 2.0)")
        sp.add_argument("--lid-confirm", type=int, default=None, help="Open 2s chunks before lock (SDK 2, paper 4)")
        sp.add_argument("--relid-after", type=float, default=None, help="Re-open LID after this many silent seconds (default 8)")
        sp.add_argument(
            "--lid-lock",
            dest="lid_lock_on",
            action="store_true",
            help="Official mode: lock ONE language per sentence (can bias to English on short hops)",
        )
        sp.add_argument("--no-lid-lock", action="store_true", help="Never force a language tag (same as default mix/auto)")
        sp.add_argument("--keep-lock", action="store_true", help="Keep language lock across sentences (only with --lid-lock)")
        sp.add_argument("--context", default="", help="Optional biasing context (system prompt)")
        sp.add_argument("--no-vad", action="store_true")
        sp.add_argument("--vad", action="store_true")
        sp.add_argument(
            "--silence-commit",
            type=float,
            default=None,
            help="Seconds of trailing silence before LAST refine (default 1.5)",
        )
        sp.add_argument(
            "--silence-hangover",
            type=float,
            default=None,
            help="Ignore brief energy dips shorter than this before counting silence (default 0.45)",
        )
        sp.add_argument(
            "--no-refine",
            action="store_true",
            help="Disable background LAST refine (disables auto-tune for refine)",
        )
        sp.add_argument(
            "--no-auto-tune",
            action="store_true",
            help="Disable runtime auto-tuning (advanced)",
        )
        sp.add_argument("--no-token-stream", action="store_true")
        sp.add_argument(
            "--no-sound-gate",
            action="store_true",
            help="Disable speech vs non-speech heuristic (send everything to ASR)",
        )
        sp.add_argument(
            "--sound-gate-conf",
            type=float,
            default=None,
            help="Min confidence 0..1 to skip ASR on pure non-speech (default 0.62)",
        )
        sp.add_argument(
            "--sound-model",
            choices=("auto", "pann", "heuristic", "off"),
            default=None,
            help="Force sound tagging mode (default: runtime auto picks best)",
        )
        sp.add_argument(
            "--pann-interval",
            type=float,
            default=None,
            help="Seconds between PANNs CNN6 event scans (default 1.5)",
        )
        sp.add_argument(
            "--pann-min-score",
            type=float,
            default=None,
            help="Min PANNs score to show an event tag (default 0.45)",
        )
        sp.add_argument(
            "--pann-block-score",
            type=float,
            default=None,
            help="Min score to block ASR for pure events like cough (default 0.55)",
        )
        sp.add_argument(
            "--pann-companion-score",
            type=float,
            default=None,
            help="Min score to show companion tags like music (default 0.38)",
        )
        sp.add_argument(
            "--pann-speech-score",
            type=float,
            default=None,
            help="Speech presence threshold inside PANNs (default 0.35)",
        )

    m = sub.add_parser("mic", help="Stream from the Windows microphone")
    add_stream_flags(m)
    m.add_argument("--device", type=int, default=None, help="sounddevice input index")
    m.set_defaults(func=cmd_mic)

    f = sub.add_parser("file", help="Replay a WAV as if it were a live stream")
    add_stream_flags(f)
    f.add_argument("file", help="16-bit WAV (any rate; resampled to 16 kHz)")
    f.add_argument("--realtime", action="store_true", help="Feed chunks at wall-clock speed")
    f.set_defaults(func=cmd_file)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LlamaServerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
