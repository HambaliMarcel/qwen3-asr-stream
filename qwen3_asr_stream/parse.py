"""Qwen3-ASR output parsing (language tag + transcript)."""

from __future__ import annotations

import re
from typing import Optional

ASR_TEXT_TAG = "<asr_text>"
LANG_PREFIX = "language "

SUPPORTED_LANGUAGES = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]


def normalize_language_name(language: str) -> str:
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()


def canonicalize_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    s = str(language).strip()
    if not s:
        return None
    if s.lower() in {"auto", "multi", "any", "all", "none", "detect", "mix", "mixed", "campuran"}:
        return None
    aliases = {
        "en": "English",
        "eng": "English",
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "cn": "Chinese",
        "yue": "Cantonese",
        "ja": "Japanese",
        "jp": "Japanese",
        "ko": "Korean",
        "kr": "Korean",
        "id": "Indonesian",
        "ms": "Malay",
        "vi": "Vietnamese",
        "th": "Thai",
        "ar": "Arabic",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "pt": "Portuguese",
        "it": "Italian",
        "ru": "Russian",
        "tr": "Turkish",
        "hi": "Hindi",
        "nl": "Dutch",
        "sv": "Swedish",
        "da": "Danish",
        "fi": "Finnish",
        "pl": "Polish",
        "cs": "Czech",
        "fil": "Filipino",
        "fa": "Persian",
        "el": "Greek",
        "ro": "Romanian",
        "hu": "Hungarian",
        "mk": "Macedonian",
    }
    key = s.lower()
    if key in aliases:
        return aliases[key]
    name = normalize_language_name(s)
    if name not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")
    return name


_LANG_TAG_RE = re.compile(
    r"(?is)language\s+([A-Za-z]+)\s*<asr_text>"
)
_BARE_TAG_RE = re.compile(r"(?i)<asr_text>")
_LANG_ONLY_RE = re.compile(
    r"(?i)(?:^|\s)language\s+(" + "|".join(re.escape(x) for x in SUPPORTED_LANGUAGES) + r")\b"
)


def strip_asr_markup(text: str) -> tuple[str, str]:
    """Remove every `language X<asr_text>` leak. Return (last_language, clean_text)."""
    if not text:
        return "", ""
    s = str(text)
    langs = [m.group(1) for m in _LANG_TAG_RE.finditer(s)]
    lang = ""
    if langs:
        try:
            lang = normalize_language_name(langs[-1])
        except ValueError:
            lang = langs[-1]
    s = _LANG_TAG_RE.sub("", s)
    s = _BARE_TAG_RE.sub("", s)
    s = _LANG_ONLY_RE.sub(" ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+,", ",", s)
    return lang, s.strip()


def detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    def fix_char_repeats(s: str, thresh: int) -> str:
        res: list[str] = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i : i + count])
                i += count
        return "".join(res)

    text = fix_char_repeats(text, threshold)
    # Collapse only *glued* decoder echoes (no space between copies):
    #   "HelloHelloHello" / "Oke, coba kitaOke, coba kita"
    # Never collapse spoken repeats: "pesawat pesawat pesawat pesawat"
    glued_word = re.compile(r"(?<![A-Za-zÀ-ÿ\u4e00-\u9fff])([A-Za-zÀ-ÿ\u4e00-\u9fff]{5,24})\1+")
    glued_phrase = re.compile(r"(?<![A-Za-zÀ-ÿ\u4e00-\u9fff])(\S(?:.{6,46}?\S))\1+")
    prev = None
    while prev != text:
        prev = text
        text = glued_word.sub(r"\1", text)
        text = glued_phrase.sub(r"\1", text)
    return text


def stitch_transcript(prev: str, new: str) -> str:
    """Merge prefix + new decode without echoing.

    llama.cpp usually re-transcribes the whole window (full hypothesis).
    vLLM-style official concat is only safe when `new` is a continuation.
    """
    prev = detect_and_fix_repetitions((prev or "").strip())
    new = detect_and_fix_repetitions((new or "").strip())
    if not new:
        return prev
    if not prev:
        return new
    if new.startswith(prev):
        return new
    if prev.startswith(new):
        return prev
    if prev in new:
        return new
    if new in prev:
        return prev

    max_k = min(len(prev), len(new))
    for k in range(max_k, 3, -1):
        if prev[-k:] == new[:k]:
            return detect_and_fix_repetitions(prev + new[k:])

    pw, nw = prev.split(), new.split()
    for i in range(min(len(pw), len(nw)), 0, -1):
        if pw[-i:] == nw[:i]:
            return detect_and_fix_repetitions(" ".join(pw + nw[i:]))

    return new


def parse_asr_output(raw: str, user_language: Optional[str] = None) -> tuple[str, str]:
    if raw is None:
        return "", ""
    tagged_lang, stripped = strip_asr_markup(str(raw))
    s = detect_and_fix_repetitions(stripped)
    if not s and not tagged_lang:
        return "", ""

    if "language none" in str(raw).lower() and not s:
        return "", ""

    lang = tagged_lang
    if user_language:
        lang = user_language
    return lang, s.strip()


# ASR often maps coughs / bursts to onomatopoeia (especially Chinese 咳咳咳).
_EVENT_CJK = re.compile(r"^[\s。．，、！？咳嗯呵啊呃哈]+(?:[\s。．，、！？]+)?$")
_EVENT_EN = re.compile(
    r"^[\s\*]*(?:cough|ahem|achoo|hachoo|sneeze|sniff)(?:[\s\*.,!?]+(?:cough|ahem|achoo|hachoo|sneeze|sniff))*[\s\*.,!?]*$",
    re.I,
)


def classify_sound_event_text(text: str) -> Optional[str]:
    """Return a short event label when ASR output is non-lexical sound, not speech."""
    t = (text or "").strip()
    if not t:
        return None
    if t.count("咳") >= 1:
        return "batuk?"
    if _EVENT_EN.fullmatch(t):
        return "batuk / bersin?"
    core = re.sub(r"[\s。．，、！？\.,!?\-\*]+", "", t)
    if not core:
        return None
    if len(core) <= 10 and all(c == "咳" for c in core):
        return "batuk?"
    if _EVENT_CJK.fullmatch(t):
        return "suara non-bicara?"
    if len(core) <= 8 and len(set(core)) <= 2 and all("\u4e00" <= c <= "\u9fff" for c in core):
        return "suara non-bicara?"
    return None


def merge_languages(langs: list[str]) -> str:
    out: list[str] = []
    prev = None
    for x in langs:
        x = (x or "").strip()
        if not x or x == prev:
            continue
        if x not in out:
            out.append(x)
        prev = x
    return "+".join(out)


def assistant_prefill(raw_prefix: str, force_language: Optional[str]) -> str:
    """Build the assistant-side prefill that llama-server should continue."""
    _, clean = strip_asr_markup(raw_prefix or "")
    if force_language:
        return f"{LANG_PREFIX}{force_language}{ASR_TEXT_TAG}{clean}"
    return clean
