import logging
import re
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import soundfile as sf

from app.config import Settings

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
MIN_REF_DURATION_S = 0.5
MAX_REF_DURATION_S = 60.0


class VoiceRegistryError(Exception):
    pass


class DuplicateVoiceError(VoiceRegistryError):
    pass


class InvalidVoiceError(VoiceRegistryError):
    pass


@dataclass
class VoiceInfo:
    voice_id: str
    ref_text: str
    duration_s: float
    wav_path: Path


class VoiceRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._voices: dict[str, VoiceInfo] = {}

    def _root(self) -> Path:
        return Path(self._settings.voices_dir)

    def scan(self) -> None:
        self._voices = {}
        root = self._root()
        root.mkdir(parents=True, exist_ok=True)
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            wav_path = entry / "ref.wav"
            txt_path = entry / "ref.txt"
            if not wav_path.exists() or not txt_path.exists():
                logger.warning(
                    "skipping malformed voice entry %s (missing ref.wav/ref.txt)", entry
                )
                continue
            try:
                info = sf.info(str(wav_path))
                duration_s = info.frames / info.samplerate
                ref_text = txt_path.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001 - one bad entry must not break boot
                logger.warning("skipping unreadable voice entry %s", entry, exc_info=True)
                continue
            self._voices[entry.name] = VoiceInfo(entry.name, ref_text, duration_s, wav_path)
        logger.info("voice registry loaded %d voice(s) from %s", len(self._voices), root)

    def register(self, name: str, audio_bytes: bytes, ref_text: str) -> VoiceInfo:
        if not _NAME_RE.match(name or ""):
            raise InvalidVoiceError("name must match [a-z0-9_-]{1,64}")
        if name in self._voices:
            raise DuplicateVoiceError(f"voice {name!r} already exists")
        ref_text = (ref_text or "").strip()
        if not ref_text:
            raise InvalidVoiceError("ref_text must not be empty")
        try:
            data, samplerate = sf.read(BytesIO(audio_bytes))
        except Exception as exc:  # noqa: BLE001 - any decode failure is a client error
            raise InvalidVoiceError(f"ref_audio is not decodable audio: {exc}") from exc
        duration_s = len(data) / samplerate
        if not (MIN_REF_DURATION_S <= duration_s <= MAX_REF_DURATION_S):
            raise InvalidVoiceError(
                f"ref_audio must be between {MIN_REF_DURATION_S}s and "
                f"{MAX_REF_DURATION_S}s, got {duration_s:.2f}s"
            )
        voice_dir = self._root() / name
        voice_dir.mkdir(parents=True, exist_ok=True)
        wav_path = voice_dir / "ref.wav"
        wav_path.write_bytes(audio_bytes)
        (voice_dir / "ref.txt").write_text(ref_text, encoding="utf-8")
        info = VoiceInfo(name, ref_text, duration_s, wav_path)
        self._voices[name] = info
        return info

    def get(self, voice_id: str) -> "VoiceInfo | None":
        return self._voices.get(voice_id)

    def list_voices(self) -> list[VoiceInfo]:
        return sorted(self._voices.values(), key=lambda v: v.voice_id)

    def delete(self, voice_id: str) -> bool:
        info = self._voices.pop(voice_id, None)
        if info is None:
            return False
        shutil.rmtree(self._root() / voice_id, ignore_errors=True)
        return True
