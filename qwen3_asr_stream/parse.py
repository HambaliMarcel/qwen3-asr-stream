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


_BARE_TAG_RE = re.compile(r"(?i)<asr_text>")
_LANG_ANY_RE = re.compile(
    r"(?i)(?:^|\s)language\s+([A-Za-z]+)\b(?:\s*<asr_text>)?"
)
_LANG_NONE_RE = re.compile(r"(?i)\blanguage\s+none\b")


def resolve_language_name(raw: str) -> str:
    """Map a model LID token, including truncated leaks like `Canton`."""
    s = (raw or "").strip()
    if not s or s.lower() == "none":
        return ""
    try:
        return canonicalize_language(s) or ""
    except ValueError:
        pass
    key = s.lower()
    matches = [name for name in SUPPORTED_LANGUAGES if name.lower().startswith(key)]
    if len(matches) == 1 and len(key) >= 3:
        return matches[0]
    return ""


def strip_asr_markup(text: str) -> tuple[str, str]:
    """Remove every `language X<asr_text>` leak. Return (last_language, clean_text)."""
    if not text:
        return "", ""
    s = str(text)
    langs: list[str] = []

    def _take(match: re.Match[str]) -> str:
        token = match.group(1)
        named = resolve_language_name(token)
        tagged = "<asr_text>" in match.group(0).lower()
        if token.lower() == "none" or named or tagged:
            if named:
                langs.append(named)
            return " "
        return match.group(0)

    s = _LANG_ANY_RE.sub(_take, s)
    s = _BARE_TAG_RE.sub("", s)
    s = _LANG_NONE_RE.sub(" ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\s+,", ",", s)
    lang = langs[-1] if langs else ""
    leftover = s.strip()
    if leftover.lower() in {"none", "language none"}:
        leftover = ""
    return lang, leftover


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


_LOOP_MAX_NGRAM = 8
_LOOP_MIN_REPEATS = 4
# "na na na na na" is a normal hook; only a longer run of one word is a spiral.
_LOOP_MIN_REPEATS_1GRAM = 6
_LOOP_NORM_RE = re.compile(r"[\s,.!?;:…\-\"'`]+")


def _loop_norm(word: str) -> str:
    return _LOOP_NORM_RE.sub("", word.lower())


def collapse_loops(text: str, keep: int = 2, min_repeats: int = _LOOP_MIN_REPEATS) -> tuple[str, bool]:
    """Cut a runaway decoder loop ("black on black on black on …").

    Spoken hooks repeated 2–3 times are kept. Only a word n-gram that repeats
    `min_repeats`+ times back-to-back is collapsed to `keep` copies. Returns
    (text, looped).
    """
    words = (text or "").split()
    n = len(words)
    if n < min_repeats:
        return text or "", False
    norm = [_loop_norm(w) for w in words]
    out: list[str] = []
    looped = False
    i = 0
    while i < n:
        best_len = 0
        best_reps = 0
        max_len = min(_LOOP_MAX_NGRAM, (n - i) // min_repeats)
        for length in range(1, max_len + 1):
            unit = norm[i : i + length]
            if not any(unit):
                continue
            reps = 1
            j = i + length
            while j + length <= n and norm[j : j + length] == unit:
                reps += 1
                j += length
            need = max(min_repeats, _LOOP_MIN_REPEATS_1GRAM) if length == 1 else min_repeats
            if reps >= need and reps * length > best_reps * best_len:
                best_len, best_reps = length, reps
        if best_len:
            out.extend(words[i : i + best_len * keep])
            i += best_len * best_reps
            looped = True
        else:
            out.append(words[i])
            i += 1
    return " ".join(out), looped


def _in_order_word_matches(prev_words: list[str], new_words: list[str]) -> int:
    """Count prev words that appear in order in new (greedy subsequence)."""
    matched = 0
    j = 0
    n = len(new_words)
    for w in prev_words:
        while j < n and new_words[j] != w:
            j += 1
        if j >= n:
            break
        matched += 1
        j += 1
    return matched


def _join_word_overlap(prev_words: list[str], new_words: list[str], min_words: int) -> int:
    limit = min(len(prev_words), len(new_words))
    for i in range(limit, min_words - 1, -1):
        if prev_words[-i:] == new_words[:i]:
            return i
    return 0


def stitch_transcript(prev: str, new: str) -> str:
    """Merge prefix + new decode without echoing.

    llama.cpp usually re-transcribes the whole window (full hypothesis).
    Concat is only safe when `new` is a short continuation of `prev`.
    Short word-overlap concat is what doubled rap lines ("no love" / "no love").
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

    pw, nw = prev.split(), new.split()
    if len(pw) >= 4:
        matched = _in_order_word_matches(pw, nw)
        if matched >= max(4, int(0.55 * len(pw))):
            return new if len(nw) >= len(pw) else prev

    # True continuation: new is short and shares a long tail overlap.
    min_join = 4 if min(len(pw), len(nw)) >= 6 else 3
    ov = _join_word_overlap(pw, nw, min_join)
    if ov and len(nw) <= max(8, len(pw) // 2):
        return detect_and_fix_repetitions(" ".join(pw + nw[ov:]))

    # Latest full-window hypothesis wins. Never glue on a 1–2 word overlap
    # (rap repeats those hooks constantly).
    return new


def join_segments(head: str, tail: str) -> str:
    """Append a finished LIVE window onto the LAST paragraph.

    Windows never share audio, so this is a plain join: a repeated hook line
    ("no love… no love…") is real speech, not an echo to dedupe.
    """
    head = (head or "").strip()
    tail = (tail or "").strip()
    if not tail:
        return head
    if not head:
        return tail
    return head + " " + tail


def prefer_transcript(prev: str, new: str) -> str:
    """Pick the better of two full-window hypotheses — never concatenate.

    Keep `prev` only when `new` looks truncated (seal/max_tokens cut the line).
    """
    prev = detect_and_fix_repetitions((prev or "").strip())
    new = detect_and_fix_repetitions((new or "").strip())
    if not new:
        return prev
    if not prev:
        return new
    if not has_lexical_speech(new) and has_lexical_speech(prev):
        return prev
    pw, nw = prev.split(), new.split()
    if prev.startswith(new) and len(pw) > len(nw):
        return prev
    if len(nw) < max(3, int(len(pw) * 0.72)) and _in_order_word_matches(nw, pw) >= max(
        2, int(0.6 * len(nw))
    ):
        return prev
    return new


def parse_asr_output(raw: str, user_language: Optional[str] = None) -> tuple[str, str]:
    if raw is None:
        return "", ""
    tagged_lang, stripped = strip_asr_markup(str(raw))
    s = detect_and_fix_repetitions(stripped)
    if (tagged_lang or "").strip().lower() == "none":
        tagged_lang = ""
    if not s and not tagged_lang:
        return "", ""

    if _LANG_NONE_RE.search(str(raw)) and not s:
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


_LEXICAL_RE = re.compile(
    r"[A-Za-zÀ-ÿ]{2,}|[\u4e00-\u9fff]+|[\u3040-\u30ff]+|[\uac00-\ud7af]+|\d{2,}"
)
_ONOMATOPOEIA_CJK = frozenset("咳嗯呵啊呃哈")


def _is_onomatopoeia_only(text: str) -> bool:
    core = re.sub(r"[\s。．，、！？\.,!?\-\*]+", "", text or "")
    if not core:
        return True
    if _EVENT_EN.fullmatch((text or "").strip()):
        return True
    return all(c in _ONOMATOPOEIA_CJK for c in core)


def has_lexical_speech(text: str) -> bool:
    """True when ASR output looks like real words, not only onomatopoeia."""
    t = strip_event_prefix((text or "").strip())
    t = _LANG_NONE_RE.sub(" ", t).strip()
    if t.lower() in {"", "none", "language", "language none"}:
        return False
    if not t:
        return False
    if _is_onomatopoeia_only(t):
        return False
    event = classify_sound_event_text(t)
    if event:
        for run in re.findall(r"[\u4e00-\u9fff]+", t):
            meaningful = [c for c in run if c not in _ONOMATOPOEIA_CJK]
            if len(meaningful) >= 2:
                return True
        lowered = re.sub(
            r"(?i)\b(?:cough|ahem|achoo|hachoo|sneeze|sniff)\b",
            " ",
            t,
        )
        if re.search(r"[A-Za-zÀ-ÿ]{2,}", lowered.strip()):
            return True
        return False
    if re.search(r"[A-Za-zÀ-ÿ]{2,}", t):
        return True
    if re.search(r"[\u3040-\u30ff]{2,}|[\uac00-\ud7af]{2,}|\d{2,}", t):
        return True
    return bool(_LEXICAL_RE.search(t))


def combine_event_and_transcript(event: str, text: str) -> str:
    """Prefix companion event tag, keep full ASR text (e.g. music + lyrics)."""
    ev = (event or "").strip()
    body = (text or "").strip()
    if not ev:
        return body
    if not body:
        return f"[{ev}]"
    if body.startswith("[") and "]" in body:
        return body
    return f"[{ev}] {body}"


def strip_event_prefix(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("[") and "]" in t:
        _, _, rest = t.partition("]")
        return rest.strip()
    return t


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
    return None


# Short-hop LID in Qwen3-ASR is English-heavy. Use function words in the
# transcript to correct a clearly wrong tag (not to force a language).
_ID_WORDS = frozenset(
    {
        "yang",
        "dan",
        "saya",
        "aku",
        "kamu",
        "anda",
        "tidak",
        "bukan",
        "sudah",
        "bisa",
        "untuk",
        "dengan",
        "ini",
        "itu",
        "ada",
        "dari",
        "kita",
        "kami",
        "mereka",
        "kalau",
        "kalo",
        "jadi",
        "tapi",
        "atau",
        "karena",
        "mau",
        "nggak",
        "ngga",
        "gak",
        "dong",
        "sih",
        "lah",
        "kok",
        "deh",
        "udah",
        "aja",
        "banget",
        "buat",
        "biar",
        "apa",
        "siapa",
        "mana",
        "kenapa",
        "gimana",
        "bagaimana",
        "terus",
        "lalu",
        "sekarang",
        "nanti",
        "tadi",
        "besok",
        "kemarin",
        "tolong",
        "coba",
        "ngomong",
        "iya",
        "yaudah",
        "pak",
        "bu",
        "mas",
        "mbak",
        "kabar",
        "berapa",
        "jam",
        "heran",
        "desa",
        "tak",
        "gue",
        "lu",
        "lo",
        "emang",
        "ngakak",
        "wkwk",
        "anjir",
        "buset",
    }
)
_EN_WORDS = frozenset(
    {
        "the",
        "and",
        "you",
        "are",
        "this",
        "that",
        "with",
        "from",
        "have",
        "what",
        "when",
        "will",
        "just",
        "like",
        "they",
        "them",
        "your",
        "about",
        "would",
        "could",
        "should",
        "there",
        "their",
        "been",
        "were",
        "going",
        "gonna",
        "don't",
        "isn't",
    }
)
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ']+")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_CJK_LANGS = {"Japanese", "Chinese", "Cantonese", "Korean"}
_ARABIC_LANGS = {"Arabic", "Persian"}
_ROMANCE_LANGS = {"Spanish", "Portuguese", "French", "Italian", "Romanian"}


def _lang_word_scores(text: str) -> tuple[int, int]:
    words = [w.lower().replace("'", "") for w in _WORD_RE.findall(text or "")]
    id_n = sum(1 for w in words if w in _ID_WORDS)
    en_n = sum(1 for w in words if w in _EN_WORDS)
    return id_n, en_n


def language_plausible(tag: str, text: str) -> bool:
    """Reject hop LID that cannot match the transcript script or lexicon."""
    name = (tag or "").strip()
    if not name or name.lower() == "none":
        return False
    try:
        name = canonicalize_language(name) or name
    except ValueError:
        return False
    body = strip_event_prefix(text or "")
    if not body.strip():
        return True
    id_n, en_n = _lang_word_scores(body)
    if name in _CJK_LANGS:
        return bool(_CJK_RE.search(body))
    if name in _ARABIC_LANGS:
        return bool(_ARABIC_RE.search(body))
    if _CJK_RE.search(body) and name in {"Indonesian", "Malay", "English"}:
        return False
    if name in _ROMANCE_LANGS:
        if id_n >= 1 and id_n >= en_n:
            return False
        return True
    if name == "Malay" and id_n >= 1:
        return True
    if name == "English" and id_n >= 1 and en_n == 0:
        return False
    return True


_MIX_FAMILY = frozenset({"English", "Indonesian", "Malay"})


def continuation_language(tagged: str, text: str) -> str:
    """Reuse the model's own LID tag for the next hop prefill.

    Official mix streaming does not force a language and does not replace the
    model's tag with a lexicon guess. Empty means the next hop runs LID again.
    """
    name = resolve_language_name(tagged)
    if not name:
        return ""
    body = strip_event_prefix(text or "")
    if not body.strip():
        return name
    if language_plausible(name, body):
        return name
    if name in _CJK_LANGS and _CJK_RE.search(body):
        return name
    if name in _ARABIC_LANGS and _ARABIC_RE.search(body):
        return name
    return ""


def infer_languages(text: str, tagged: str = "") -> list[str]:
    """Prefer the model's LID tag; only override English when Indonesian dominates.

    Official mix (`language=None`) identifies language via `language X<asr_text>`.
    A Latin lexicon must not replace Cantonese/Japanese/Arabic/Spanish tags.
    """
    tagged = resolve_language_name(tagged)
    body = strip_event_prefix(text)
    id_n, en_n = _lang_word_scores(body)
    out: list[str] = []

    if tagged == "English" and id_n >= 2 and id_n > en_n:
        out.append("Indonesian")
        if en_n >= 1:
            out.append("English")
    elif tagged in {"Indonesian", "Malay"}:
        out.append(tagged)
        if en_n >= 1 and "English" not in out:
            out.append("English")
    elif tagged and language_plausible(tagged, body):
        out.append(tagged)
    elif tagged in _CJK_LANGS and _CJK_RE.search(body):
        out.append(tagged)
    elif tagged in _ARABIC_LANGS and _ARABIC_RE.search(body):
        out.append(tagged)

    if not out:
        if id_n >= 2 and id_n > en_n:
            out.append("Indonesian")
        elif id_n >= 1 and en_n == 0:
            out.append("Indonesian")
        if en_n >= 2 and en_n > id_n:
            out.append("English")
        elif en_n >= 1 and id_n == 0 and "English" not in out:
            out.append("English")
        if id_n >= 1 and en_n >= 1:
            for name in ("Indonesian", "English"):
                if name not in out:
                    out.append(name)
        if not out and _ARABIC_RE.search(body):
            out.append("Arabic")
        if not out and _CJK_RE.search(body):
            out.append("Japanese" if re.search(r"[\u3040-\u30ff]", body) else "Chinese")
    elif tagged in _MIX_FAMILY and id_n >= 1 and en_n >= 1:
        for name in ("Indonesian", "English"):
            if name not in out:
                out.append(name)
    return out


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
