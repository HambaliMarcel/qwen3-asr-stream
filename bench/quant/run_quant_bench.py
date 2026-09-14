"""Compare Qwen3-ASR 1.7B Q8_0 vs bf16 without touching the live stack.

Uses a throwaway llama-server on the bench port from config.json (default 19999).
Never binds 9999 or 8080. Never edits production launchers.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASR_ROOT = ROOT.parents[1]
if str(ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(ASR_ROOT))

from qwen3_asr_stream.audio import load_wav_file  # noqa: E402
from qwen3_asr_stream.client import LlamaAsrClient  # noqa: E402
from qwen3_asr_stream.server import build_server_cmd  # noqa: E402

CLIPS = ROOT / "clips"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
CFG_PATH = ROOT / "config.json"
MANIFEST_PATH = CLIPS / "manifest.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def download(url: str, dest: Path, timeout: float = 60.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "qwen-asr-quant-bench"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1000:
            return False
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"  skip {url} ({exc})")
        return False


def prepare_clips(manifest: dict) -> list[dict]:
    clips: list[dict] = []
    for item in manifest.get("clips") or []:
        dest = CLIPS / item["file"]
        ok = True
        if item.get("url"):
            print(f"clip  {item['id']}  {item['file']}")
            ok = download(item["url"], dest)
        if ok and dest.is_file():
            clips.append({**item, "path": str(dest)})
    for url in manifest.get("probe_urls") or []:
        name = url.rsplit("/", 1)[-1]
        dest = CLIPS / name
        print(f"probe {name}")
        if download(url, dest):
            lang_guess = name.replace("asr_", "").replace(".wav", "")
            lang_map = {
                "ja": "Japanese",
                "yue": "Cantonese",
                "ko": "Korean",
                "id": "Indonesian",
                "ar": "Arabic",
                "es": "Spanish",
                "de": "German",
            }
            clips.append(
                {
                    "id": f"probe_{dest.stem}",
                    "file": name,
                    "path": str(dest),
                    "expected_lang": lang_map.get(lang_guess, ""),
                    "reference": "",
                }
            )
    return clips


def normalize_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff']+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def units(text: str) -> list[str]:
    t = normalize_text(text)
    if not t:
        return []
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", t):
        return [ch for ch in re.sub(r"\s+", "", t)]
    return t.split()


def levenshtein(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def error_rate(ref: str, hyp: str) -> float | None:
    ru = units(ref)
    if not ru:
        return None
    return levenshtein(ru, units(hyp)) / len(ru)


def port_open(host: str, port: int) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def start_bench_server(cfg: dict, model_path: str) -> subprocess.Popen:
    logs = LOGS
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"server-{Path(model_path).stem}.log"
    cmd = build_server_cmd(
        port=int(cfg["bench_port"]),
        llama_server=cfg.get("llama_server"),
        model=model_path,
        mmproj=cfg.get("mmproj"),
        ngl=int(cfg.get("ngl", 99)),
        ctx=int(cfg.get("ctx", 4096)),
    )
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(cmd[0]).parent),
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    proc._bench_log = handle  # type: ignore[attr-defined]
    return proc


def stop_bench_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    handle = getattr(proc, "_bench_log", None)
    try:
        proc.terminate()
        proc.wait(timeout=8)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    if handle:
        try:
            handle.close()
        except Exception:
            pass


def wait_ready(client: LlamaAsrClient, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.health():
            return
        time.sleep(0.5)
    raise RuntimeError(f"bench llama-server not ready at {client.base_url}")


def load_clip_pcm(path: str):
    """Bench-only loader. Supports 24-bit official Qwen wavs without changing production audio.py."""
    import numpy as np
    import wave

    from qwen3_asr_stream.audio import float_pcm, load_wav_file, resample_16k

    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 3:
        return load_wav_file(path)
    packed = np.frombuffer(raw, dtype=np.uint8)
    n = packed.size // 3
    triples = packed[: n * 3].reshape(n, 3).astype(np.int32)
    vals = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    vals = np.where(vals >= 0x800000, vals - 0x1000000, vals)
    x = vals.astype(np.float32) / 8388608.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return resample_16k(float_pcm(x), sr)


def transcribe_clip(client: LlamaAsrClient, clip: dict, cfg: dict, force: str | None) -> dict:
    pcm = load_clip_pcm(clip["path"])
    result = client.transcribe(
        pcm,
        force_language=force or None,
        max_tokens=int(cfg.get("max_tokens", 128)),
        temperature=float(cfg.get("temperature", 0.01)),
    )
    ref = clip.get("reference") or ""
    return {
        "clip_id": clip["id"],
        "file": clip["file"],
        "mode": "forced" if force else "mix",
        "force_language": force or "",
        "expected_lang": clip.get("expected_lang") or "",
        "language": result.language,
        "text": result.text,
        "latency_ms": round(result.latency_ms, 1),
        "audio_sec": round(result.audio_sec, 3),
        "rtf": round(result.rtf, 3),
        "error_rate": error_rate(ref, result.text),
        "lid_ok": (
            bool(force)
            or not clip.get("expected_lang")
            or (result.language or "").lower() == clip["expected_lang"].lower()
        ),
    }


def run_model(cfg: dict, model: dict, clips: list[dict]) -> dict:
    host = cfg.get("bench_host", "127.0.0.1")
    port = int(cfg["bench_port"])
    if port_open(host, port):
        raise RuntimeError(f"bench port {port} already in use — not touching it")
    print(f"\n== {model['label']}  :{port}")
    proc = start_bench_server(cfg, model["path"])
    client = LlamaAsrClient(f"http://{host}:{port}", timeout=180.0)
    rows: list[dict] = []
    t_load = time.perf_counter()
    try:
        wait_ready(client, float(cfg.get("load_timeout_sec", 240)))
        load_ms = (time.perf_counter() - t_load) * 1000.0
        print(f"  ready in {load_ms:.0f}ms")
        for clip in clips:
            modes: list[str | None] = [None]
            if clip.get("expected_lang"):
                modes.append(str(clip["expected_lang"]))
            for force in modes:
                mode = "forced" if force else "mix"
                print(f"  {clip['id']:24s} {mode:7s}", end=" ", flush=True)
                try:
                    row = transcribe_clip(client, clip, cfg, force)
                except Exception as exc:
                    shown = str(exc).encode("ascii", "replace").decode("ascii")
                    print(f"FAIL {shown[:120]}")
                    rows.append(
                        {
                            "clip_id": clip["id"],
                            "file": clip["file"],
                            "mode": mode,
                            "force_language": force or "",
                            "expected_lang": clip.get("expected_lang") or "",
                            "language": "",
                            "text": "",
                            "latency_ms": 0.0,
                            "audio_sec": 0.0,
                            "rtf": 0.0,
                            "error_rate": None,
                            "lid_ok": False,
                            "error": str(exc),
                        }
                    )
                    continue
                shown = (row["text"] or "").encode("ascii", "replace").decode("ascii")
                print(f"{row['latency_ms']:.0f}ms  [{row['language']}] {shown[:80]}")
                rows.append(row)
        return {
            "id": model["id"],
            "label": model["label"],
            "path": model["path"],
            "bytes": Path(model["path"]).stat().st_size,
            "load_ms": round(load_ms, 1),
            "rows": rows,
        }
    finally:
        stop_bench_server(proc)
        time.sleep(1.2)


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compare(runs: list[dict]) -> dict:
    by_id = {run["id"]: run for run in runs}
    q8 = by_id["q8_0"]
    bf = by_id["bf16"]
    pairs = []
    q8_rows = {(r["clip_id"], r["mode"]): r for r in q8["rows"]}
    bf_rows = {(r["clip_id"], r["mode"]): r for r in bf["rows"]}
    keys = sorted(set(q8_rows) & set(bf_rows))
    agree = 0
    for key in keys:
        a, b = q8_rows[key], bf_rows[key]
        same_text = normalize_text(a["text"]) == normalize_text(b["text"])
        same_lang = (a["language"] or "").lower() == (b["language"] or "").lower()
        if same_text and same_lang:
            agree += 1
        pairs.append(
            {
                "clip_id": key[0],
                "mode": key[1],
                "q8_lang": a["language"],
                "bf16_lang": b["language"],
                "q8_text": a["text"],
                "bf16_text": b["text"],
                "q8_ms": a["latency_ms"],
                "bf16_ms": b["latency_ms"],
                "q8_er": a["error_rate"],
                "bf16_er": b["error_rate"],
                "same_text": same_text,
                "same_lang": same_lang,
                "lid_q8": a["lid_ok"],
                "lid_bf16": b["lid_ok"],
            }
        )

    def mix_rows(run: dict) -> list[dict]:
        return [r for r in run["rows"] if r["mode"] == "mix"]

    def summary(run: dict) -> dict:
        mix = mix_rows(run)
        ers = [r["error_rate"] for r in mix if r["error_rate"] is not None]
        return {
            "id": run["id"],
            "label": run["label"],
            "bytes_gb": round(run["bytes"] / (1024**3), 3),
            "load_s": round(run["load_ms"] / 1000.0, 1),
            "avg_latency_ms": round(mean([r["latency_ms"] for r in mix]) or 0.0, 1),
            "avg_rtf": round(mean([r["rtf"] for r in mix]) or 0.0, 3),
            "avg_error_rate": None if not ers else round(mean(ers), 4),
            "lid_correct": sum(1 for r in mix if r["lid_ok"]),
            "lid_n": sum(1 for r in mix if r["expected_lang"]),
        }

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pair_n": len(keys),
        "agree_n": agree,
        "agree_pct": round(100.0 * agree / len(keys), 1) if keys else 0.0,
        "summaries": [summary(q8), summary(bf)],
        "pairs": pairs,
        "runs": runs,
    }


def write_markdown(cmp: dict) -> str:
    lines = [
        "# Qwen3-ASR 1.7B  Q8_0 vs bf16",
        "",
        f"Independent bench on port 19999. {cmp['ts']}",
        "",
        "| quant | size | load | avg mix latency | RTF | error vs gold | mix LID |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in cmp["summaries"]:
        er = "—" if s["avg_error_rate"] is None else f"{100 * s['avg_error_rate']:.1f}%"
        lid = f"{s['lid_correct']}/{s['lid_n']}" if s["lid_n"] else "—"
        lines.append(
            f"| {s['label']} | {s['bytes_gb']:.2f} GB | {s['load_s']:.1f}s | "
            f"{s['avg_latency_ms']:.0f} ms | {s['avg_rtf']:.2f} | {er} | {lid} |"
        )
    lines += [
        "",
        f"Transcript+LID agreement: **{cmp['agree_n']}/{cmp['pair_n']}** ({cmp['agree_pct']}%).",
        "",
        "| clip | mode | Q8_0 | bf16 | same |",
        "|---|---|---|---|---|",
    ]
    for p in cmp["pairs"]:
        same = "yes" if p["same_text"] and p["same_lang"] else "no"
        q = f"[{p['q8_lang']}] {p['q8_text']}".replace("|", "/")
        b = f"[{p['bf16_lang']}] {p['bf16_text']}".replace("|", "/")
        lines.append(f"| {p['clip_id']} | {p['mode']} | {q} | {b} | {same} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    cfg = load_json(CFG_PATH)
    port = int(cfg["bench_port"])
    if port in {9999, 8080}:
        print("refusing to use production ports", file=sys.stderr)
        return 2
    for key in ("llama_server", "mmproj"):
        if not Path(cfg[key]).is_file():
            print(f"missing {key}: {cfg[key]}", file=sys.stderr)
            return 2
    for model in cfg["models"]:
        if not Path(model["path"]).is_file():
            print(f"missing model: {model['path']}", file=sys.stderr)
            return 2

    CLIPS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    clips = prepare_clips(load_json(MANIFEST_PATH))
    if not clips:
        print("no clips downloaded", file=sys.stderr)
        return 2
    print(f"\n{len(clips)} clips ready")

    runs = []
    for model in cfg["models"]:
        runs.append(run_model(cfg, model, clips))

    cmp = compare(runs)
    (RESULTS / "compare.json").write_text(json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8")
    md = write_markdown(cmp)
    (RESULTS / "compare.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"wrote {RESULTS / 'compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
