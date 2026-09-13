$ErrorActionPreference = "Stop"
$llama = if ($env:LLAMA_SERVER) { $env:LLAMA_SERVER } else { "C:\AI\llama.cpp\llama-server.exe" }
$models = if ($env:QWEN_ASR_MODELS_DIR) { $env:QWEN_ASR_MODELS_DIR } else { "C:\AI\models" }
$model = if ($env:QWEN_ASR_MODEL) { $env:QWEN_ASR_MODEL } else { Join-Path $models "Qwen3-ASR-1.7B-Q8_0.gguf" }
$mmproj = if ($env:QWEN_ASR_MMPROJ) { $env:QWEN_ASR_MMPROJ } else { Join-Path $models "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf" }
$port = if ($env:QWEN_ASR_PORT) { $env:QWEN_ASR_PORT } else { "9999" }

Write-Host "llama-server  $model"
Write-Host "mmproj       $mmproj"
Write-Host "port         $port"

& $llama `
  -m $model `
  --mmproj $mmproj `
  -ngl 99 `
  -c 4096 `
  -np 1 `
  -n 32 `
  --temp 0.01 `
  --port $port `
  --host 127.0.0.1 `
  -fa on `
  --jinja `
  --prefill-assistant `
  --cache-prompt `
  --mmproj-offload `
  --no-webui
