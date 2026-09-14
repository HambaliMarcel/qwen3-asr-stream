$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

try { chcp 65001 | Out-Null } catch {}
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$Host.UI.RawUI.WindowTitle = "Qwen3-ASR  live"

# Default: multilingual auto-detect (do not lock English).
python -m qwen3_asr_stream mic @args
