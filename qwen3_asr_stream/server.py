"""Launch llama-server with low-latency flags for Qwen3-ASR."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .client import LlamaAsrClient, LlamaServerError

DEFAULT_LLAMA_DIR = Path(r"C:\AI\llama.cpp")
DEFAULT_MODELS_DIR = Path(r"C:\AI\models")
# Q4_K_M: same VRAM class as Q4_0, far fewer decoder loops on sung input.
DEFAULT_MODEL = "Qwen3-ASR-1.7B-Q4_K_M.gguf"
FALLBACK_MODEL = "Qwen3-ASR-1.7B-Q4_0.gguf"
DEFAULT_MMPROJ = "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"
DEFAULT_PORT = 9999


def _env_path(name: str, fallback: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else fallback


def resolve_paths(
    llama_server: Optional[str] = None,
    model: Optional[str] = None,
    mmproj: Optional[str] = None,
) -> tuple[Path, Path, Path]:
    exe = Path(llama_server) if llama_server else _env_path("LLAMA_SERVER", DEFAULT_LLAMA_DIR / "llama-server.exe")
    if exe.is_dir():
        exe = exe / "llama-server.exe"
    models = _env_path("QWEN_ASR_MODELS_DIR", DEFAULT_MODELS_DIR)
    default_p = models / DEFAULT_MODEL
    if not default_p.is_file() and (models / FALLBACK_MODEL).is_file():
        default_p = models / FALLBACK_MODEL
    model_p = Path(model) if model else Path(os.environ.get("QWEN_ASR_MODEL") or default_p)
    mmproj_p = Path(mmproj) if mmproj else Path(os.environ.get("QWEN_ASR_MMPROJ") or (models / DEFAULT_MMPROJ))
    if not model_p.is_file() and (models / model_p.name).is_file():
        model_p = models / model_p.name
    if model_p.name == "qwen3-asr-1.7b-q4_0.gguf" and default_p.is_file():
        # cstr's GGUF is a CrispASR arch (`qwen3asr`) llama.cpp cannot load.
        model_p = default_p
    if not mmproj_p.is_file() and (models / mmproj_p.name).is_file():
        mmproj_p = models / mmproj_p.name
    return exe, model_p, mmproj_p


def build_server_cmd(
    port: int = DEFAULT_PORT,
    llama_server: Optional[str] = None,
    model: Optional[str] = None,
    mmproj: Optional[str] = None,
    ngl: int = 99,
    ctx: int = 1536,
    extra: Optional[list[str]] = None,
) -> list[str]:
    exe, model_p, mmproj_p = resolve_paths(llama_server, model, mmproj)
    if not exe.is_file():
        raise FileNotFoundError(f"llama-server not found: {exe}")
    if not model_p.is_file():
        raise FileNotFoundError(f"ASR model not found: {model_p}")
    if not mmproj_p.is_file():
        raise FileNotFoundError(f"mmproj not found: {mmproj_p}")

    cmd = [
        str(exe),
        "-m",
        str(model_p),
        "--mmproj",
        str(mmproj_p),
        "-ngl",
        str(ngl),
        "-c",
        str(ctx),
        # 20 s seal ≈ 250 audio + 256 text tokens, so 1536 is plenty; q8 KV
        # and a smaller batch trim ~150 MB VRAM at identical hop latency.
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "-b",
        "512",
        "-ub",
        "256",
        "-np",
        "1",
        "-n",
        "32",
        "--temp",
        "0.01",
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "-fa",
        "on",
        "--jinja",
        "--prefill-assistant",
        "--cache-prompt",
        # Every hop is a new audio prompt, so the host-RAM prompt cache only
        # adds a KV copy per hop; in-slot reuse is what actually helps.
        "--cache-ram",
        "0",
        # No mmap: keeps the weights out of the file page cache so Windows
        # does not evict and re-read them from SSD mid-session.
        "--load-mode",
        "none",
        "--mmproj-offload",
        "--no-webui",
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def start_server(
    port: int = DEFAULT_PORT,
    host: str = "127.0.0.1",
    llama_server: Optional[str] = None,
    model: Optional[str] = None,
    mmproj: Optional[str] = None,
    ngl: int = 99,
    ctx: int = 4096,
    extra: Optional[list[str]] = None,
    new_console: bool = True,
) -> subprocess.Popen:
    cmd = build_server_cmd(
        port=port,
        llama_server=llama_server,
        model=model,
        mmproj=mmproj,
        ngl=ngl,
        ctx=ctx,
        extra=extra,
    )
    kwargs: dict = {}
    if sys.platform == "win32" and new_console:
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    cwd = str(Path(cmd[0]).parent)
    return subprocess.Popen(cmd, cwd=cwd, **kwargs)


def ensure_server(
    url: str,
    start: bool,
    **launch_kw,
) -> Optional[subprocess.Popen]:
    client = LlamaAsrClient(url)
    if client.health():
        return None
    if not start:
        raise LlamaServerError(
            f"llama-server is not running at {url}. Start it with: python -m qwen3_asr_stream serve"
        )
    proc = start_server(**launch_kw)
    client.wait_until_ready(timeout=240.0)
    return proc


def which_or(path: str) -> Optional[str]:
    return shutil.which(path)
