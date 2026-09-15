# Qwen3-ASR stream (llama.cpp)

Realtime microphone STT in a Windows terminal using **Qwen3-ASR GGUF + llama-server**. No vLLM.

Official Qwen streaming is documented for vLLM. llama-server exposes one-shot `/v1/chat/completions` with audio. This project implements the same **prefix-rollback** algorithm Qwen uses in `streaming_transcribe()` as a client loop:

1. Capture **16 kHz mono** PCM from the mic (WASAPI shared mode).
2. Every **hop** (default **~0.6 s**, auto-tuned down to **0.5 s** when the GPU keeps up), send the **current audio window** to llama-server.
3. Prefill the assistant with the previous transcript minus the last **5 tokens** (`/tokenize` + `/detokenize` on the live GGUF). The model's own `language X<asr_text>` tag from the window is reused so it **continues** the line instead of restarting it.
4. Show a yellow **LIVE** tail (words can still revise). On pause, **LAST** shows the line instantly; a background **seal** re-decodes the finished window without blocking the mic.

Long speech (rap, monologue) is **batched**: at a breath gap (~0.35 s after ≥6 s of speech), the finished window moves into **LAST** and **LIVE** starts a fresh window with **no audio overlap** — so words are never transcribed twice.

## Requirements

- Windows 10/11, Python 3.10+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server.exe` with `--prefill-assistant` and `--cache-prompt`
- Qwen3-ASR GGUF + matching `mmproj` (e.g. `Qwen3-ASR-1.7B-Q4_K_M.gguf` + `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf`; prefer K-quants over legacy Q4_0 — the Q4_0 decoder spirals into repeats on sung input). The stream has a loop guard: a repeated n-gram in the partial text aborts that decode, the window is reset, and DRY sampling is sent for the next ~8 s.

```powershell
python -m pip install -r requirements.txt
# Optional: anti-aliased mic resample + PANNs event tags
python -m pip install scipy
python -m pip install -r requirements-pann.txt
```

## Quick start

**Terminal 1 — server**

```powershell
cd path\to\qwen3-asr-stream
python -m qwen3_asr_stream serve
```

Or the launcher script (same flags as `serve`):

```powershell
.\scripts\start-server.ps1
```

**Terminal 2 — live mic**

```powershell
cd path\to\qwen3-asr-stream
python -m qwen3_asr_stream devices
python -m qwen3_asr_stream mic
```

Or:

```powershell
.\scripts\stream-mic.ps1
.\scripts\start-mic.bat
```

Auto-start server from the mic command:

```powershell
python -m qwen3_asr_stream mic --start-server
```

## Dashboard (LIVE vs LAST)

The terminal is a **fixed panel** — not a scrolling log.

| Box | Meaning |
|-----|---------|
| **LIVE** | Current window only. White = stable words, yellow + `▌` = word still being written. |
| **LAST** | Committed text: batched windows during long speech, plus the final line after ~1.5 s pause. |

During instrumental / beat-only stretches you may see `[musik] hearing sound, waiting for words` — the mic is hot but ASR has no lyrics yet. After a batch cut, LIVE shows `… next line` until the next words land.

Status line shows hop, decode latency, RTF, and an **auto · hop … pause …** hint when runtime tuning is active.

## Language

Default is **multilingual mix** — Indonesian, English, and others can appear in one session. The model is **not** forced to `language English<asr_text>` unless you ask.

```powershell
python -m qwen3_asr_stream mic                          # mix (default)
python -m qwen3_asr_stream mic --language Indonesian    # force one language
python -m qwen3_asr_stream mic --lid-lock               # official: one language per sentence
python -m qwen3_asr_stream languages                    # list supported names
```

Avoid `--language English` unless you want English-only; it skips open LID entirely.

## Profiles

| Profile | Hop | Auto-tune | Notes |
|---------|-----|-----------|-------|
| **`auto`** (default) | 0.6 s → 0.5–2.0 s | yes | Adapts hop, pause, refine, and PANNs from measured RTF. Best for daily use. |
| `ultralow` | 0.40 s | no | Fixed fastest hop; same multilingual defaults as `auto`. |
| `balanced` | 1.0 s | no | Middle ground. |
| `official` | 2.0 s | no | SDK-style 2 s chunks, no hard window cap (`--max-audio 0`). |
| `paper` | 2.0 s | no | Paper table: last four chunks unfixed. |

Replay a WAV at wall-clock speed (good for latency tests):

```powershell
python -m qwen3_asr_stream file C:\path\to\clip.wav --realtime
python -m qwen3_asr_stream file clip.wav --realtime --profile ultralow
```

## Useful flags

```powershell
python -m qwen3_asr_stream mic --hop 0.5
python -m qwen3_asr_stream mic --max-audio 12          # hard-cut LIVE window at 12 s (default cap 16 s when 0)
python -m qwen3_asr_stream mic --silence-commit 1.2    # pause before LAST commit
python -m qwen3_asr_stream mic --device 1
python -m qwen3_asr_stream mic --context "standup meeting"
python -m qwen3_asr_stream mic --no-auto-tune          # freeze profile knobs
python -m qwen3_asr_stream mic --no-refine             # skip background LAST seal
python -m qwen3_asr_stream mic --sound-model off       # ASR only, no event tags
python -m qwen3_asr_stream mic --no-vad                # file-style: no energy gate
```

## Latency (realistic)

llama.cpp **re-encodes the audio window on every hop**. There is no incremental encoder cache like vLLM's streaming backend. This client keeps that cost bounded:

- **Continuation prefills** (~150 ms per hop on a warm GPU) instead of full re-transcription every time.
- **No silence hops** after a breath — one flush decode for the last syllable, then wait for speech or commit.
- **Window batching** — long utterances split into ≤16 s (or `--max-audio`) segments so decode time stays flat.
- **Single server slot** (`-np 1`), flash-attn, `cache_prompt`, low temperature.

First decode after server start is slower (CUDA warmup). Trust the second utterance for real RTF.

## Layout (typical)

```
path\to\qwen3-asr-stream\   this repo
C:\AI\models\               Qwen3-ASR GGUF + mmproj
C:\AI\llama.cpp\            llama-server.exe
```

Override with environment variables: `LLAMA_SERVER`, `QWEN_ASR_MODEL`, `QWEN_ASR_MMPROJ`, `QWEN_ASR_MODELS_DIR`, `QWEN_ASR_PORT`, `QWEN_ASR_HOST`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/start-server.ps1` | Launch llama-server with ASR flags |
| `scripts/stream-mic.ps1` | UTF-8 console + `python -m qwen3_asr_stream mic` |
| `scripts/start-mic.bat` | Same as above from cmd.exe |

## Development branch

Active work lives on **`development`**. `main` is the stable pointer.
