$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

try { chcp 65001 | Out-Null } catch {}
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$Host.UI.RawUI.WindowTitle = "Qwen3-ASR  live"

# Default: auto language (not English). Pass extra args to override.
python -m qwen3_asr_stream mic --language auto @args
