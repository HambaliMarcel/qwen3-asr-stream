$ErrorActionPreference = "Stop"
$llama = if ($env:LLAMA_SERVER) { $env:LLAMA_SERVER } else { "C:\AI\llama.cpp\llama-server.exe" }
$models = if ($env:QWEN_ASR_MODELS_DIR) { $env:QWEN_ASR_MODELS_DIR } else { "C:\AI\models" }
$default = Join-Path $models "Qwen3-ASR-1.7B-Q4_K_M.gguf"
if (-not (Test-Path -LiteralPath $default)) { $default = Join-Path $models "Qwen3-ASR-1.7B-Q4_0.gguf" }
$model = if ($env:QWEN_ASR_MODEL) { $env:QWEN_ASR_MODEL } else { $default }
if ([System.IO.Path]::GetFileName($model) -ceq "qwen3-asr-1.7b-q4_0.gguf") {
  Write-Host "cstr qwen3asr GGUF is CrispASR-only; using llama.cpp $default"
  $model = $default
}
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
