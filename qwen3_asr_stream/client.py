"""llama-server client: chat-completions ASR + official tokenize rollback."""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .audio import SAMPLE_RATE, pcm_to_wav_bytes
from .parse import assistant_prefill, parse_asr_output, stitch_transcript


class LlamaServerError(RuntimeError):
    pass


@dataclass
class DecodeResult:
    raw: str
    language: str
    text: str
    latency_ms: float
    audio_sec: float
    tokens: int = 0

    @property
    def rtf(self) -> float:
        if self.audio_sec <= 0:
            return 0.0
        return (self.latency_ms / 1000.0) / self.audio_sec


class LlamaAsrClient:
    def __init__(self, base_url: str = "http://127.0.0.1:9999", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._audio_style = "input_audio"

    def _request(
        self,
        path: str,
        payload: Optional[dict] = None,
        method: str = "POST",
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> tuple[int, bytes]:
        data = None
        hdrs = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        if headers:
            hdrs.update(headers)
        req = Request(self.base_url + path, data=data, headers=hdrs, method=method)
        try:
            with urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.status, resp.read()
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise LlamaServerError(f"{path} HTTP {e.code}: {body[:800]}") from e
        except URLError as e:
            raise LlamaServerError(f"Cannot reach llama-server at {self.base_url}: {e.reason}") from e

    def health(self) -> bool:
        try:
            code, _ = self._request("/health", method="GET", timeout=3.0)
            return 200 <= code < 300
        except Exception:
            return False

    def wait_until_ready(self, timeout: float = 180.0) -> None:
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            try:
                if self.health():
                    return
                last = "health not ready"
            except Exception as e:
                last = str(e)
            time.sleep(0.4)
        raise LlamaServerError(f"llama-server did not become ready: {last}")

    def tokenize(self, text: str) -> list[int]:
        if not text:
            return []
        _, body = self._request("/tokenize", {"content": text, "add_special": False})
        data = json.loads(body.decode("utf-8"))
        tokens = data.get("tokens") or []
        return [int(t) for t in tokens]

    def detokenize(self, tokens: list[int]) -> str:
        if not tokens:
            return ""
        _, body = self._request("/detokenize", {"tokens": tokens})
        data = json.loads(body.decode("utf-8"))
        return str(data.get("content") or "")

    def rollback_prefix(self, raw: str, unfixed_token_num: int, *, exact: bool = True) -> str:
        """Drop the last K tokens using the live GGUF tokenizer (Qwen official strategy)."""
        if not raw or unfixed_token_num <= 0:
            return raw
        if not exact:
            return _heuristic_rollback(raw, unfixed_token_num)
        try:
            ids = self.tokenize(raw)
        except LlamaServerError:
            return _heuristic_rollback(raw, unfixed_token_num)
        k = int(unfixed_token_num)
        while True:
            end = max(0, len(ids) - k)
            prefix = self.detokenize(ids[:end]) if end else ""
            if "\ufffd" not in prefix:
                return prefix
            if end == 0:
                return ""
            k += 1

    def _messages(self, wav_b64: str, prefill: str, context: str, style: str) -> list[dict]:
        if style == "input_audio":
            audio_part = {
                "type": "input_audio",
                "input_audio": {"data": wav_b64, "format": "wav"},
            }
        else:
            audio_part = {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"},
            }
        messages = [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [audio_part]},
        ]
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        return messages

    def _chat_once(
        self,
        wav_b64: str,
        prefill: str,
        context: str,
        max_tokens: int,
        temperature: float,
        style: str,
    ) -> str:
        payload = {
            "messages": self._messages(wav_b64, prefill, context, style),
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": max_tokens,
            "stream": False,
            # Server is launched with --cache-prompt; reuse the cached prompt
            # prefix across the many rolling-window decodes of one utterance.
            "cache_prompt": True,
        }
        _, body = self._request("/v1/chat/completions", payload)
        data = json.loads(body.decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return str(msg.get("content") or "")

    def transcribe(
        self,
        pcm,
        raw_prefix: str = "",
        context: str = "",
        force_language: Optional[str] = None,
        max_tokens: int = 32,
        temperature: float = 0.01,
        on_partial: Optional[Callable[[str, str], None]] = None,
        prefill_language: Optional[str] = None,
    ) -> DecodeResult:
        """One decode of `pcm`.

        `raw_prefix` is the fixed text the model must continue (official
        streaming). `prefill_language` is the tag the model itself emitted for
        this window — it keeps the prefill in the model's own output format
        (`language X<asr_text>…`) so it continues instead of restarting the
        line. It is not a user-forced language: `force_language` is.
        """
        wav = pcm_to_wav_bytes(pcm, SAMPLE_RATE)
        wav_b64 = base64.b64encode(wav).decode("ascii")
        tag_lang = force_language or (prefill_language if raw_prefix else None)
        prefill = assistant_prefill(raw_prefix, tag_lang)
        audio_sec = float(getattr(pcm, "size", 0)) / float(SAMPLE_RATE)
        t0 = time.perf_counter()

        last_err: Optional[Exception] = None
        gen = ""
        used_fallback = False
        styles = [self._audio_style]
        if self._audio_style == "input_audio":
            styles.append("audio_url")
        else:
            styles.append("input_audio")

        for style in styles:
            try:
                if on_partial is not None:
                    try:
                        gen = self._chat_stream(
                            wav_b64,
                            prefill,
                            context,
                            max_tokens,
                            temperature,
                            style,
                            on_partial=lambda delta_raw: _emit_partial(
                                on_partial, raw_prefix, delta_raw, force_language
                            ),
                        )
                    except LlamaServerError:
                        gen = self._chat_once(
                            wav_b64, prefill, context, max_tokens, temperature, style
                        )
                else:
                    gen = self._chat_once(wav_b64, prefill, context, max_tokens, temperature, style)
                self._audio_style = style
                last_err = None
                break
            except LlamaServerError as e:
                last_err = e
                continue
        if last_err is not None:
            try:
                gen = self._transcriptions(wav, temperature)
                last_err = None
                used_fallback = True
            except LlamaServerError as e:
                raise LlamaServerError(f"{last_err}\nfallback /v1/audio/transcriptions also failed: {e}") from e

        latency_ms = (time.perf_counter() - t0) * 1000.0
        lang, gen_text = parse_asr_output(gen, user_language=force_language)
        _, prev_text = parse_asr_output(raw_prefix, user_language=force_language)
        text = stitch_transcript(prev_text, gen_text)
        return DecodeResult(
            raw=text,
            language=lang or force_language or "",
            text=text,
            latency_ms=latency_ms,
            audio_sec=audio_sec,
        )

    def _chat_stream(
        self,
        wav_b64: str,
        prefill: str,
        context: str,
        max_tokens: int,
        temperature: float,
        style: str,
        on_partial: Callable[[str], None],
    ) -> str:
        payload = {
            "messages": self._messages(wav_b64, prefill, context, style),
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": max_tokens,
            "stream": True,
            "cache_prompt": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            self.base_url + "/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        pieces: list[str] = []
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                buf = ""
                while True:
                    chunk = resp.read(256)
                    if not chunk:
                        break
                    buf += chunk.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload_s = line[5:].strip()
                        if payload_s == "[DONE]":
                            return "".join(pieces)
                        try:
                            evt = json.loads(payload_s)
                        except json.JSONDecodeError:
                            continue
                        choices = evt.get("choices") or []
                        if not choices:
                            continue
                        delta = (choices[0].get("delta") or {}).get("content") or ""
                        if delta:
                            pieces.append(delta)
                            on_partial("".join(pieces))
        except HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise LlamaServerError(f"/v1/chat/completions HTTP {e.code}: {body[:800]}") from e
        except URLError as e:
            raise LlamaServerError(f"Cannot reach llama-server at {self.base_url}: {e.reason}") from e
        return "".join(pieces)

    def _transcriptions(self, wav: bytes, temperature: float) -> str:
        boundary = "----qwen3asrstream"
        chunks = [
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; filename="chunk.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode("utf-8")
            + wav
            + b"\r\n",
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="temperature"\r\n\r\n'
                f"{temperature}\r\n"
            ).encode("utf-8"),
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="response_format"\r\n\r\n'
                "json\r\n"
            ).encode("utf-8"),
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        body = b"".join(chunks)
        req = Request(
            self.base_url + "/v1/audio/transcriptions",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise LlamaServerError(f"/v1/audio/transcriptions HTTP {e.code}: {err[:800]}") from e
        except URLError as e:
            raise LlamaServerError(f"Cannot reach llama-server at {self.base_url}: {e.reason}") from e
        return str(data.get("text") or "")


def _emit_partial(
    on_partial: Callable[[str, str], None],
    prev_text: str,
    gen: str,
    force_language: Optional[str],
) -> None:
    lang, gen_text = parse_asr_output(gen, user_language=force_language)
    _, prev = parse_asr_output(prev_text, user_language=force_language)
    on_partial(lang, stitch_transcript(prev, gen_text))


def _heuristic_rollback(text: str, k: int) -> str:
    # Fallback if /tokenize is unavailable: drop last K whitespace/CJK units.
    parts = [p for p in _TOKEN_SPLIT.split(text) if p != ""]
    if len(parts) <= k:
        return ""
    return "".join(parts[:-k])


_TOKEN_SPLIT = re.compile(r"(\s+|[\u4e00-\u9fff]|[^\w\s])")
