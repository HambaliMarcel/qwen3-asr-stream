$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$asr = "C:\Users\marce\Projects\qwen3-asr-stream"
Set-Location $asr
python -m qwen3_asr_stream.integrator @args
