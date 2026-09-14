# Independent Q8_0 vs bf16 ASR bench

Compares `Qwen3-ASR-1.7B-Q8_0.gguf` and `Qwen3-ASR-1.7B-bf16.gguf` on the same clips.

Does **not** change production `start-all.ps1`, port 9999, port 8080, or live mix flags. It spins a throwaway llama-server on **19999**, one quant at a time, then kills that process only.

```powershell
cd C:\Users\marce\Projects\qwen3-asr-stream\bench\quant
python run_quant_bench.py
```

Outputs `results/compare.json` and `results/compare.md`.
