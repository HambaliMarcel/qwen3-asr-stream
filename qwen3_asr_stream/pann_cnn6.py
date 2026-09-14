"""PANNs CNN6 AudioSet event tagger — CPU-only, lazy-loaded."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import SAMPLE_RATE

CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth?download=1"
)
CHECKPOINT_NAME = "Cnn6_mAP=0.343.pth"
MODEL_SR = 32000
MIN_SEC = 0.8
MAX_SEC = 10.0
# PANNs is a clip-level tagger (trained on ~10 s clips), not a frame VAD.
# Tag a fixed recent window so cost is constant regardless of utterance length.
TAG_WINDOW_SEC = 6.0

# Never show or block on ambient/noise tags.
IGNORE_LABELS = frozenset(
    {
        "Silence",
        "Static",
        "White noise",
        "Pink noise",
        "Noise",
        "Environmental noise",
        "Mains hum",
        "Distortion",
        "Sound effect",
        "Inside, small room",
        "Inside, large room or hall",
        "Inside, public space",
        "Outside, urban or manmade",
        "Outside, rural or natural",
        "Reverberation",
        "Echo",
        "Wind noise (microphone)",
        "Hubbub, speech noise, speech babble",
    }
)

# Show alongside transcript; never block ASR.
COMPANION_LABELS = frozenset(
    {
        "Music",
        "Song",
        "Musical instrument",
        "Singing",
        "Male singing",
        "Female singing",
        "Child singing",
        "Synthetic singing",
        "Rapping",
        "Humming",
        "Choir",
        "Background music",
        "Theme music",
        "Soundtrack music",
        "Pop music",
        "Electronic music",
        "Speech",
        "Conversation",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Narration, monologue",
        "Whispering",
    }
)

# May block ASR only when there is no lexical transcript.
BLOCKING_LABELS = frozenset(
    {
        "Cough",
        "Sneeze",
        "Throat clearing",
        "Clapping",
        "Applause",
        "Hands",
        "Finger snapping",
        "Fart",
        "Burping, eructation",
        "Hiccup",
        "Sniff",
        "Knock",
        "Tap",
        "Bang",
        "Burst, pop",
        "Laughter",
        "Baby laughter",
        "Giggle",
    }
)


@dataclass(frozen=True)
class EventTag:
    label: str
    score: float
    top_labels: tuple[str, ...] = ()
    blocks_asr: bool = False
    is_companion: bool = False


def pann_available() -> bool:
    try:
        import torch  # noqa: F401
        from torchlibrosa.stft import Spectrogram  # noqa: F401

        return True
    except ImportError:
        return False


def _cache_dir() -> Path:
    base = os.environ.get(
        "QWEN_ASR_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "qwen3_asr_stream"),
    )
    path = Path(base) / "panns"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resample_16k_to_32k(pcm16k: np.ndarray) -> np.ndarray:
    """Upsample 16 kHz -> 32 kHz with an anti-aliased polyphase filter.

    Official PANNs inference resamples with librosa/soxr. Naive linear
    interpolation folds imaging artefacts into the speech band (fmax 14 kHz)
    and measurably degrades clipwise scores, so prefer scipy's polyphase
    resampler and only fall back to linear when scipy is unavailable.
    """
    x = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    try:
        from scipy.signal import resample_poly

        return resample_poly(x, 2, 1).astype(np.float32)
    except ImportError:
        n = x.size * 2
        src = np.arange(x.size, dtype=np.float32)
        dst = np.linspace(0, x.size - 1, n, dtype=np.float32)
        return np.interp(dst, src, x).astype(np.float32)


def _short_label(name: str) -> str:
    mapping = {
        "Cough": "batuk",
        "Sneeze": "bersin",
        "Clapping": "tepuk tangan",
        "Applause": "tepuk tangan",
        "Hands": "tepuk / ketukan",
        "Fart": "kentut",
        "Burping, eructation": "sendawa",
        "Hiccup": "cegukan",
        "Dog": "anjing",
        "Bark": "gonggongan anjing",
        "Cat": "kucing",
        "Meow": "meong",
        "Bird": "burung",
        "Music": "musik",
        "Song": "lagu",
        "Singing": "nyanyian",
        "Laughter": "tawa",
        "Knock": "ketukan",
        "Tap": "ketukan",
        "Bang": "ledakan",
        "Burst, pop": "ledakan",
    }
    if name in mapping:
        return mapping[name]
    return name.split(",")[0].lower()


class PannCnn6Tagger:
    """Lazy PANNs CNN6 tagger. Runs on CPU only."""

    def __init__(
        self,
        min_score: float = 0.45,
        block_score: float = 0.55,
        companion_score: float = 0.38,
        # PANNs speech posteriors for clear speech are ~0.8-0.9; anything
        # below ~0.3 is babble/music bleed. 0.18 vetoed blocking constantly.
        speech_score: float = 0.35,
        device: str = "cpu",
    ) -> None:
        self.min_score = min_score
        self.block_score = block_score
        self.companion_score = companion_score
        self.speech_score = speech_score
        self.device = device
        self._model = None
        self._labels: tuple[str, ...] = ()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        from .panns.cnn6_model import Cnn6
        from .panns.labels import AUDIOSET_LABELS, SPEECH_LABELS

        self._speech_labels = SPEECH_LABELS
        ckpt_path = _cache_dir() / CHECKPOINT_NAME
        if not ckpt_path.is_file():
            print(f"Downloading PANNs CNN6 (~24 MB) to {ckpt_path} ...")
            urllib.request.urlretrieve(CHECKPOINT_URL, ckpt_path)

        self._labels = AUDIOSET_LABELS
        model = Cnn6(
            sample_rate=MODEL_SR,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=len(self._labels),
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self._model = model

    def tag(self, pcm16k: np.ndarray) -> Optional[EventTag]:
        if not pann_available():
            return None
        x16 = np.asarray(pcm16k, dtype=np.float32).reshape(-1)
        if x16.size < int(MIN_SEC * SAMPLE_RATE * 0.5):
            return None
        # Fixed recent window: constant inference cost, no growing-utterance
        # slowdown, and matches clip-tagger training conditions better than
        # zero-padded short hops or 24 s accumulations.
        win_n = int(TAG_WINDOW_SEC * SAMPLE_RATE)
        if x16.size > win_n:
            x16 = x16[-win_n:]

        import torch

        from .panns.labels import SPEECH_LABELS

        self._ensure_loaded()
        x = _resample_16k_to_32k(x16)
        max_n = int(MAX_SEC * MODEL_SR)
        min_n = int(MIN_SEC * MODEL_SR)
        if x.size > max_n:
            x = x[-max_n:]
        if x.size < min_n:
            x = np.pad(x, (0, min_n - x.size))

        wave = torch.from_numpy(x[None, :])
        with torch.inference_mode():
            out = self._model(wave)["clipwise_output"].cpu().numpy()[0]

        order = np.argsort(out)[::-1]
        top_labels = tuple(self._labels[int(i)] for i in order[:5])

        speech_score = max(
            (float(out[i]) for i, name in enumerate(self._labels) if name in SPEECH_LABELS),
            default=0.0,
        )

        best_name = ""
        best_score = 0.0
        for i in order:
            name = self._labels[int(i)]
            score = float(out[int(i)])
            if name in IGNORE_LABELS or name in SPEECH_LABELS:
                continue
            if score > best_score:
                best_name = name
                best_score = score
            if best_score >= self.min_score:
                break

        if not best_name or best_score < self.companion_score:
            return None

        short = _short_label(best_name)
        has_speech = speech_score >= self.speech_score

        # Companion events (music/singing) are display-only and never block.
        if best_name in COMPANION_LABELS:
            if best_score < self.companion_score:
                return None
            return EventTag(
                label=short,
                score=best_score,
                top_labels=top_labels,
                blocks_asr=False,
                is_companion=True,
            )

        # Speech present: let Qwen decide. A clip tagger must not veto or
        # relabel real speech (e.g. "[dog] hello world" was pure noise).
        if has_speech:
            return None

        if best_name in BLOCKING_LABELS and best_score >= self.block_score:
            return EventTag(
                label=short,
                score=best_score,
                top_labels=top_labels,
                blocks_asr=True,
                is_companion=False,
            )

        if best_score >= self.min_score:
            return EventTag(
                label=short,
                score=best_score,
                top_labels=top_labels,
                blocks_asr=False,
                is_companion=False,
            )

        return None


def get_tagger(
    min_score: float = 0.45,
    block_score: float = 0.55,
    companion_score: float = 0.38,
    speech_score: float = 0.35,
) -> Optional[PannCnn6Tagger]:
    if not pann_available():
        return None
    return PannCnn6Tagger(
        min_score=min_score,
        block_score=block_score,
        companion_score=companion_score,
        speech_score=speech_score,
    )
