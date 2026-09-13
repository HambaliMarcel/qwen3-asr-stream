# Qwen3-ASR stream (llama.cpp)

Realtime microphone STT in a Windows terminal using **your existing Qwen3-ASR GGUF + llama-server**. No vLLM.

Official Qwen streaming is documented as vLLM-only. llama-server only does one-shot `/v1/audio/transcriptions`. This project implements the same streaming algorithm Qwen uses in `streaming_transcribe()` as a client loop:

1. Capture 16 kHz mono PCM from the mic (low-latency WASAPI).
2. Every hop (default **400 ms**), send **all audio in the current window** to llama-server.
3. Prefill the assistant with the previous transcript minus the last **5 tokens** (`/tokenize` + `/detokenize` on the live GGUF).
4. Show a yellow **unfixed** tail that can still be revised; silence commits a green line.

That prefix-rollback trick is what makes chunked llama.cpp output feel like a real stream instead of disconnected clips.

## Run

Terminal 1 — start the server (your models, extra low-latency flags):

```powershell
cd C:\AI\qwen3-asr-stream
python -m qwen3_asr_stream serve
```

Or keep using your command. Prefill is already on by default in current llama.cpp:

```powershell
cd C:\AI\models
..\llama.cpp\llama-server.exe -m "Qwen3-ASR-1.7B-Q8_0.gguf" --mmproj "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf" -ngl 99 -c 4096 -np 1 --port 9999
```

Terminal 2 — live mic (auto language, in-place dashboard — not a log):

```powershell
cd C:\AI\qwen3-asr-stream
python -m pip install -r requirements.txt
python -m qwen3_asr_stream devices
python -m qwen3_asr_stream languages
python -m qwen3_asr_stream mic
```

Or:

```powershell
.\scripts\stream-mic.ps1
```

Language is **auto** by default (30 languages, switches every utterance). Do **not** pass `--language English` unless you want to lock English.

```powershell
python -m qwen3_asr_stream mic --language auto
python -m qwen3_asr_stream mic --language Indonesian
python -m qwen3_asr_stream mic --language Chinese
python -m qwen3_asr_stream mic --start-server
```

Replay a WAV as a stream (good for latency tests without talking):

```powershell
python -m qwen3_asr_stream file C:\path\to\clip.wav --realtime --profile ultralow
```

## Profiles

| Profile     | Text hop | LID          | Notes                                              |
|-------------|----------|--------------|----------------------------------------------------|
| `ultralow`  | 0.40s    | 2s × 2 lock  | Fast words **after** official LID lock.            |
| `balanced`  | 0.70s    | 2s × 2 lock  | Middle ground.                                     |
| `official`  | 2.00s    | SDK defaults | `chunk=2s`, `unfixed=2`, `rollback=5`.             |
| `paper`     | 2.00s    | 2s × 4       | Table 8: last four chunks unfixed.                 |

Default is **mix**: Indonesian + English (and others) in **one sentence**, no language lock.

LIVE is a fast draft. After you pause, LAST is a full official pass of that whole utterance (no prefix, no forced language, 256 tokens) — that is what fixes WER.

```powershell
python -m qwen3_asr_stream mic
python -m qwen3_asr_stream mic --language mix
python -m qwen3_asr_stream mic --language auto
python -m qwen3_asr_stream mic --language Indonesian
```

Do not pass `--language English` unless you want English-only. That skips LID entirely.

```powershell
python -m qwen3_asr_stream mic
python -m qwen3_asr_stream mic --language Indonesian
python -m qwen3_asr_stream mic --profile official
python -m qwen3_asr_stream mic --lid-confirm 4
python -m qwen3_asr_stream languages
```

The PowerShell window is a **fixed live panel**: white = stable words, yellow + `▌` = the word still being written, LAST box = the last committed sentence. It does not print a growing history.

More flags: `--hop 0.35`, `--max-audio 6`, `--unfixed-tokens 5`, `--device N`, `--context "standup meeting"`.

## What “ultra-low latency” can actually be

llama.cpp must **re-encode the audio window on every hop**. There is no incremental encoder cache like vLLM’s streaming backend. This client keeps that cost bounded:

- Short hops so the first partial appears after ~0.4s + one GPU decode.
- Latest-wins: if a decode is slower than the hop, pending audio is merged into the next request instead of queued.
- Rolling 8s window so encode time does not grow for long speech.
- Energy VAD: silence is not sent; a short pause commits the utterance.
- `max_tokens=24`, `temperature=0.01`, one server slot, flash-attn on.

First decode after server start is slower (CUDA graphs / kernel warmup). The second utterance is the real number.

## Layout

```
C:\AI\qwen3-asr-stream\     this project
C:\AI\models\               Qwen3-ASR-1.7B-Q8_0.gguf + mmproj
C:\AI\llama.cpp\            llama-server.exe
```

Override with `LLAMA_SERVER`, `QWEN_ASR_MODEL`, `QWEN_ASR_MMPROJ`, `QWEN_ASR_PORT`.
