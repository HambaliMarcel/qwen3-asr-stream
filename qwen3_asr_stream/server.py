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
DEFAULT_MODEL = "Qwen3-ASR-1.7B-Q8_0.gguf"
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
    model_p = Path(model) if model else Path(os.environ.get("QWEN_ASR_MODEL") or (models / DEFAULT_MODEL))
    mmproj_p = Path(mmproj) if mmproj else Path(os.environ.get("QWEN_ASR_MMPROJ") or (models / DEFAULT_MMPROJ))
    if not model_p.is_file() and (models / model_p.name).is_file():
        model_p = models / model_p.name
    if not mmproj_p.is_file() and (models / mmproj_p.name).is_file():
        mmproj_p = models / mmproj_p.name
    return exe, model_p, mmproj_p


def build_server_cmd(
    port: int = DEFAULT_PORT,
    llama_server: Optional[str] = None,
    model: Optional[str] = None,
    mmproj: Optional[str] = None,
    ngl: int = 99,
    ctx: int = 4096,
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
