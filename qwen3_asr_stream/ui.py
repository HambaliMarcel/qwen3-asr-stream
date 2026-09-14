"""Fixed-screen live dashboard for Windows PowerShell (no scrolling log)."""

from __future__ import annotations

import ctypes
import re
import shutil
import sys
import time

from .parse import merge_languages
from .stream import StreamState

CSI = "\x1b["
RESET = f"{CSI}0m"
BOLD = f"{CSI}1m"
DIM = f"{CSI}2m"
ITALIC = f"{CSI}3m"
UNDER = f"{CSI}4m"
INVERSE = f"{CSI}7m"

WHITE = f"{CSI}97m"
GRAY = f"{CSI}90m"
CYAN = f"{CSI}36m"
CYAN_B = f"{CSI}96m"
GREEN = f"{CSI}32m"
GREEN_B = f"{CSI}92m"
YELLOW = f"{CSI}33m"
YELLOW_B = f"{CSI}93m"
MAGENTA = f"{CSI}35m"
BLUE = f"{CSI}34m"
RED = f"{CSI}31m"

ALT_ON = f"{CSI}?1049h"
ALT_OFF = f"{CSI}?1049l"
HIDE = f"{CSI}?25l"
SHOW = f"{CSI}?25h"
HOME = f"{CSI}H"
CLEAR = f"{CSI}2J"
CLEAR_LINE = f"{CSI}2K"

WORD_RE = re.compile(
    r"[\u4e00-\u9fff]|[\u3040-\u30ff]|[\uac00-\ud7af]"
    r"|[A-Za-zÀ-ÿ0-9']+|\S"
)
BARS = "▁▂▃▄▅▆▇█"


def enable_windows_vt() -> None:
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleOutputCP(65001)
    kernel32.SetConsoleCP(65001)
    handle = kernel32.GetStdHandle(-11)
    mode = ctypes.c_uint32()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _width() -> int:
    return max(72, min(110, shutil.get_terminal_size((100, 28)).columns - 1))


def _visible_len(s: str) -> int:
    return len(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s))


def _pad_row(inner: str, width: int) -> str:
    vis = _visible_len(inner)
    if vis < width:
        inner = inner + (" " * (width - vis))
    elif vis > width:
        # Keep ANSI, trim by visible chars.
        out = []
        n = 0
        i = 0
        while i < len(inner) and n < width - 1:
            if inner[i] == "\x1b":
                m = re.match(r"\x1b\[[0-9;]*[A-Za-z]", inner[i:])
                if m:
                    out.append(m.group(0))
                    i += len(m.group(0))
                    continue
            out.append(inner[i])
            n += 1
            i += 1
        inner = "".join(out) + "…"
        inner += " " * max(0, width - _visible_len(inner))
    return inner


def _wrap(text: str, width: int, limit: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    buf = ""
    for ch in text:
        if ch == "\n":
            lines.append(buf)
            buf = ""
            continue
        buf += ch
        if len(buf) >= width:
            lines.append(buf)
            buf = ""
        if len(lines) >= limit:
            break
    if buf and len(lines) < limit:
        lines.append(buf)
    if not lines:
        lines = [""]
    while len(lines) < limit:
        lines.append("")
    return lines[:limit]


def _meter(level: float, width: int = 16) -> str:
    # Typical speech RMS is ~0.02–0.2
    norm = max(0.0, min(1.0, (level - 0.004) / 0.12))
    filled = int(round(norm * (width - 1)))
    cells = []
    for i in range(width):
        idx = min(7, int((i + 1) / width * 8))
        ch = BARS[idx]
        if i <= filled:
            color = GREEN_B if i > width * 0.66 else (YELLOW_B if i > width * 0.33 else CYAN)
            cells.append(f"{color}{ch}{RESET}")
        else:
            cells.append(f"{GRAY}{BARS[0]}{RESET}")
    return "".join(cells)


def _paint_words(stable: str, live: str) -> str:
    """White stable words + bright live tail + block caret."""
    if not stable and not live:
        return f"{DIM}{ITALIC}identifying… keep talking{RESET}"
    parts: list[str] = []
    if stable:
        parts.append(f"{WHITE}{stable}{RESET}")
        if live and not stable.endswith((" ", "\n")) and not live.startswith(" "):
            if re.search(r"[A-Za-z0-9]$", stable) and re.search(r"^[A-Za-z0-9]", live):
                parts.append(" ")
    if live:
        units = WORD_RE.findall(live)
        tail = units[-1] if units else live
        head = live[: -len(tail)] if units and live.endswith(tail) else ""
        if head:
            parts.append(f"{YELLOW}{head}{RESET}")
        parts.append(f"{YELLOW_B}{BOLD}{tail}{RESET}")
        parts.append(f"{CYAN_B}{BOLD}▌{RESET}")
    else:
        parts.append(f"{GREEN_B} ▌{RESET}")
    return "".join(parts)


def _split_stable_live(committed: str, unfixed: str, prev_unfixed: str) -> tuple[str, str]:
    text = unfixed or committed or ""
    committed = committed or ""
    if committed and text.startswith(committed):
        return committed, text[len(committed) :].lstrip()
    # Common prefix with the previous paint — already-shown words go stable.
    n = 0
    limit = min(len(prev_unfixed), len(text))
    while n < limit and prev_unfixed[n] == text[n]:
        n += 1
    # Snap back to a word / CJK boundary so we don't freeze a half-word as stable.
    while n > 0 and text[n - 1].isalnum() and n < len(text) and text[n].isalnum():
        n -= 1
    if n < max(0, len(text) - 24):
        # keep a short revising tail even if the model rewrote a lot
        n = max(0, len(text) - 24)
        while n > 0 and text[n - 1].isalnum() and n < len(text) and text[n].isalnum():
            n -= 1
    return text[:n], text[n:]


class LiveTranscript:
    """In-place dashboard. History never scrolls; only the live line changes."""

    def __init__(self) -> None:
        enable_windows_vt()
        self._started = False
        self._prev_unfixed = ""
        self._last_result = ""
        self._last_lang = ""
        self._last_draw = 0.0
        self._force = True
        self._title = ""
        self._detail = ""
        self._rows = 16

    def banner(self, title: str, detail: str) -> None:
        self._title = title
        self._detail = detail
        if not self._started:
            sys.stdout.write(ALT_ON + HIDE + CLEAR + HOME)
            self._started = True
            self._force = True
            sys.stdout.flush()

    def render(self, st: StreamState) -> None:
        if st.finalized:
            from .parse import strip_asr_markup

            _, clean = strip_asr_markup(st.finalized)
            self._last_result = clean
            if clean.startswith("[") and clean.endswith("]"):
                self._last_lang = ""
            else:
                self._last_lang = st.language or self._last_lang
            st.finalized = ""
            self._prev_unfixed = ""
            self._force = True

        now = time.perf_counter()
        text_changed = (
            st.unfixed,
            st.committed,
            st.language,
            st.language_status,
            st.decoding,
            st.speaking,
            st.lid_rounds,
            st.refining,
            getattr(st, "event_label", ""),
            getattr(st, "non_speech_only", False),
            tuple(st.utterance_langs),
        ) != getattr(self, "_sig", None)
        self._sig = (
            st.unfixed,
            st.committed,
            st.language,
            st.language_status,
            st.decoding,
            st.speaking,
            st.lid_rounds,
            st.refining,
            getattr(st, "event_label", ""),
            getattr(st, "non_speech_only", False),
            tuple(st.utterance_langs),
        )
        if not self._force and not text_changed and (now - self._last_draw) < 0.07:
            return
        self._last_draw = now
        self._force = False
        try:
            self._draw(st)
        except Exception as exc:
            sys.stderr.write(f"\nui render skipped: {exc}\n")

    def _draw(self, st: StreamState) -> None:
        w = _width()
        inner = w - 2
        live_text = st.unfixed or st.text or ""
        committed = st.committed
        stable, live = _split_stable_live(committed, live_text, self._prev_unfixed)
        if live_text:
            self._prev_unfixed = live_text

        if st.decoding:
            status = f"{YELLOW_B}{BOLD} DECODING {RESET}"
            hint = "model is writing words"
        elif getattr(st, "non_speech_only", False) and getattr(st, "sound_label", ""):
            status = f"{MAGENTA}{BOLD} NON-SPEECH {RESET}"
            hint = f"{st.sound_label} · no speech detected"
        elif st.speaking or st.speech_seen:
            status = f"{GREEN_B}{BOLD} SPEAKING {RESET}"
            ev = getattr(st, "event_label", "")
            hint = f"live words can still revise · [{ev}]" if ev else "live words can still revise"
        else:
            status = f"{CYAN}{BOLD} LISTENING {RESET}"
            hint = "LIVE is draft · LAST is a full mixed-language pass"

        lid = st.language_status
        mix = merge_languages(st.utterance_langs) if st.utterance_langs else (st.language or "")
        shown = mix or st.locked_language or st.language or "—"
        if getattr(st, "refining", False) or lid == "refining":
            lang = f"{YELLOW_B}REFINE{RESET} {WHITE}{shown}{RESET}"
        elif lid == "forced":
            lang = f"{GREEN_B}LOCKED{RESET} {WHITE}{shown}{RESET} {GRAY}(forced){RESET}"
        elif lid == "locked":
            lang = f"{GREEN_B}LOCKED{RESET} {WHITE}{shown}{RESET}"
        elif lid == "mix":
            lang = f"{CYAN_B}MIX{RESET} {WHITE}{shown}{RESET}"
        elif lid == "confirming":
            votes = " → ".join(st.lid_votes[-3:]) or shown
            lang = f"{YELLOW_B}CONFIRM{RESET} {WHITE}{votes}{RESET}"
        elif lid == "guessing":
            lang = f"{YELLOW}LID{RESET} {WHITE}{shown}{RESET}"
        elif lid == "non-speech":
            ev = getattr(st, "event_label", "") or getattr(st, "sound_label", "") or "non-bicara"
            lang = f"{MAGENTA}SOUND{RESET} {WHITE}{ev}{RESET}"
        elif getattr(st, "event_label", ""):
            ev = st.event_label
            lang = f"{MAGENTA}EVENT{RESET} {WHITE}{ev}{RESET}  {CYAN}MIX{RESET} {WHITE}{shown}{RESET}"
        else:
            have = st.audio_accum.size / 16000.0
            lang = f"{CYAN}MIX{RESET} {WHITE}{shown}{RESET} {GRAY}{have:.1f}s{RESET}"
        seen = ", ".join(st.languages_seen[-5:]) if st.languages_seen else "Indo+English+… in one line"
        hop = f"{int(st.cfg.hop_sec * 1000)}ms"
        lat = f"{st.last.latency_ms:.0f}ms" if st.last else "—"
        rtf = f"{st.last.rtf:.2f}×" if st.last else "—"
        win = f"{st.last.audio_sec:.1f}s" if st.last else "0.0s"

        top = (
            f" {status}  {lang}   {GRAY}hop{RESET} {WHITE}{hop}{RESET}  "
            f"{GRAY}decode{RESET} {WHITE}{lat}{RESET}  "
            f"{GRAY}rtf{RESET} {WHITE}{rtf}{RESET}  "
            f"{GRAY}win{RESET} {WHITE}{win}{RESET}"
        )
        meter = _meter(st.level, 18)
        live_painted = _paint_words(stable, live)
        last_body = self._last_result or f"{DIM}committed line appears here after a short pause{RESET}"
        last_lang = (self._last_lang or shown) if self._last_result else ""

        def box_top(title: str) -> str:
            label = f" {title} "
            fill = max(0, inner - len(label) - 1)
            return f"{CYAN}┌{label}{'─' * fill}┐{RESET}"

        def box_bot() -> str:
            return f"{CYAN}└{'─' * inner}┘{RESET}"

        def row(content: str) -> str:
            return f"{CYAN}│{RESET}{_pad_row(content, inner)}{CYAN}│{RESET}"

        live_lines = _wrap_ansi(live_painted, inner - 2, 4)
        last_lines = _wrap(self._last_result or "", inner - 2, 3)
        if not self._last_result:
            last_lines = _wrap_ansi(last_body, inner - 2, 3)

        out: list[str] = [
            f"{BOLD}{CYAN_B}  {self._title}{RESET}",
            box_top("SESSION"),
            row(top),
            row(f" {meter}  {GRAY}{hint}{RESET}   {GRAY}heard:{RESET} {WHITE}{seen}{RESET}"),
            box_bot(),
            box_top("LIVE  ·  draft mix"),
            *[row(" " + line) for line in live_lines],
            box_bot(),
            box_top(f"LAST  ·  refined  {last_lang or 'mix'}"),
            *[row(" " + (f"{GREEN}{line}{RESET}" if self._last_result else line)) for line in last_lines],
            box_bot(),
            f" {GRAY}Ctrl+C stop{RESET}   {GRAY}speak campur Indo/English in one sentence · pause to refine LAST{RESET}",
        ]
        self._rows = len(out)
        sys.stdout.write(HOME + CLEAR)
        sys.stdout.write("\n".join(out))
        sys.stdout.write("\n")
        sys.stdout.flush()

    def newline(self) -> None:
        self.close()

    def close(self, final: str = "") -> None:
        if self._started:
            sys.stdout.write(SHOW + ALT_OFF)
            self._started = False
            sys.stdout.flush()
        text = final or self._last_result
        if text:
            print()
            print(f"{BOLD}{GREEN}result{RESET}  {text}")
            print()


def _wrap_ansi(text: str, width: int, limit: int) -> list[str]:
    if not text:
        return [""] * limit
    # wrap on visible characters, keep escape sequences attached
    lines: list[str] = []
    cur = ""
    vis = 0
    i = 0
    while i < len(text) and len(lines) < limit:
        if text[i] == "\x1b":
            m = re.match(r"\x1b\[[0-9;]*[A-Za-z]", text[i:])
            if m:
                cur += m.group(0)
                i += len(m.group(0))
                continue
        cur += text[i]
        vis += 1
        i += 1
        if vis >= width:
            lines.append(cur + RESET)
            cur = ""
            vis = 0
    if cur and len(lines) < limit:
        lines.append(cur)
    if not lines:
        lines = [""]
    while len(lines) < limit:
        lines.append("")
    return lines[:limit]
