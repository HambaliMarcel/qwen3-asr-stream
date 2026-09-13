@echo off
cd /d "%~dp0.."
chcp 65001 >nul
title Qwen3-ASR live
python -m qwen3_asr_stream mic --language auto %*
