from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import pygame

PathLike = Union[str, Path]


class AudioManager:
    BGM_EXTENSIONS = (".ogg", ".wav", ".mp3", ".flac", ".opus", ".mod", ".mid", ".midi")
    SFX_EXTENSIONS = (".ogg", ".wav", ".mp3", ".flac", ".opus")

    DEFAULT_SCENE_BGM = {
        "map": "map",
        "start": "event",
        "event": "event",
        "sigil_choice": "event",
        "shop": "shop",
        "combat": "combat",
        "elite": "elite",
        "boss": "boss",
        "training": "training",
    }

    def __init__(
        self,
        audio_root: Optional[PathLike] = None,
        bgm_volume: float = 0.45,
        sfx_volume: float = 0.65,
        enabled: bool = True,
        channels: int = 16,
    ) -> None:
        base_assets = Path(__file__).resolve().parent / "assets"
        self.audio_root = Path(audio_root) if audio_root is not None else base_assets / "audio"
        self.bgm_dir = self.audio_root / "bgm"
        self.sfx_dir = self.audio_root / "sfx"
        self.enabled = enabled
        self.muted = False
        self.bgm_volume = self._clamp_volume(bgm_volume)
        self.sfx_volume = self._clamp_volume(sfx_volume)
        self.channels = max(1, int(channels))
        self.current_bgm_key: Optional[str] = None
        self.current_bgm_path: Optional[Path] = None
        self._sfx_cache: Dict[str, pygame.mixer.Sound] = {}
        self._ready = False
        self.last_error = ""

    def init(self) -> bool:
        if not self.enabled:
            return False
        if self._ready and pygame.mixer.get_init() is not None:
            return True
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init()
            pygame.mixer.set_num_channels(self.channels)
            pygame.mixer.music.set_volume(self._effective_bgm_volume())
            self._ready = True
            self.last_error = ""
            return True
        except Exception as exc:
            self.enabled = False
            self._ready = False
            self.last_error = str(exc)
            return False

    def play_bgm(
        self,
        name: PathLike,
        loops: int = -1,
        fade_ms: int = 700,
        restart: bool = False,
        volume: Optional[float] = None,
    ) -> bool:
        if not self.init():
            return False
        path = self._resolve_file(self.bgm_dir, name, self.BGM_EXTENSIONS)
        if path is None:
            self.last_error = f"BGM not found: {name}"
            return False

        key = str(name)
        if not restart and self.current_bgm_path == path and pygame.mixer.music.get_busy():
            return True

        try:
            pygame.mixer.music.load(str(path))
            if volume is not None:
                old_volume = self.bgm_volume
                self.bgm_volume = self._clamp_volume(volume)
                pygame.mixer.music.set_volume(self._effective_bgm_volume())
                self.bgm_volume = old_volume
            else:
                pygame.mixer.music.set_volume(self._effective_bgm_volume())
            pygame.mixer.music.play(loops=loops, fade_ms=max(0, int(fade_ms)))
            self.current_bgm_key = key
            self.current_bgm_path = path
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def play_scene_bgm(
        self,
        scene: str,
        scene_map: Optional[Dict[str, str]] = None,
        loops: int = -1,
        fade_ms: int = 700,
        restart: bool = False,
    ) -> bool:
        mapping = scene_map if scene_map is not None else self.DEFAULT_SCENE_BGM
        bgm_name = mapping.get(scene)
        if not bgm_name:
            return False
        return self.play_bgm(bgm_name, loops=loops, fade_ms=fade_ms, restart=restart)

    def stop_bgm(self, fade_ms: int = 500) -> None:
        if not self.init():
            return
        if fade_ms > 0:
            pygame.mixer.music.fadeout(int(fade_ms))
        else:
            pygame.mixer.music.stop()
        self.current_bgm_key = None
        self.current_bgm_path = None

    def pause_bgm(self) -> None:
        if self.init():
            pygame.mixer.music.pause()

    def resume_bgm(self) -> None:
        if self.init():
            pygame.mixer.music.unpause()

    def is_bgm_playing(self) -> bool:
        return self._ready and pygame.mixer.get_init() is not None and pygame.mixer.music.get_busy()

    def set_bgm_volume(self, volume: float) -> None:
        self.bgm_volume = self._clamp_volume(volume)
        if self.init():
            pygame.mixer.music.set_volume(self._effective_bgm_volume())

    def set_sfx_volume(self, volume: float) -> None:
        self.sfx_volume = self._clamp_volume(volume)

    def mute(self) -> None:
        self.muted = True
        if self.init():
            pygame.mixer.music.set_volume(0.0)

    def unmute(self) -> None:
        self.muted = False
        if self.init():
            pygame.mixer.music.set_volume(self._effective_bgm_volume())

    def toggle_mute(self) -> bool:
        if self.muted:
            self.unmute()
        else:
            self.mute()
        return self.muted

    def play_sfx(
        self,
        name: PathLike,
        volume: float = 1.0,
        loops: int = 0,
        fade_ms: int = 0,
    ) -> Optional[pygame.mixer.Channel]:
        if not self.init() or self.muted:
            return None
        sound = self._get_sfx(name)
        if sound is None:
            return None
        try:
            channel = sound.play(loops=loops, fade_ms=max(0, int(fade_ms)))
            if channel is not None:
                channel.set_volume(self.sfx_volume * self._clamp_volume(volume))
            return channel
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def preload_sfx(self, names: Iterable[PathLike]) -> None:
        for name in names:
            self._get_sfx(name)

    def clear_sfx_cache(self) -> None:
        self._sfx_cache.clear()

    def stop_all_sfx(self) -> None:
        if self.init():
            pygame.mixer.stop()

    def _get_sfx(self, name: PathLike) -> Optional[pygame.mixer.Sound]:
        key = str(name)
        if key in self._sfx_cache:
            return self._sfx_cache[key]

        path = self._resolve_file(self.sfx_dir, name, self.SFX_EXTENSIONS)
        if path is None:
            self.last_error = f"SFX not found: {name}"
            return None

        try:
            sound = pygame.mixer.Sound(str(path))
            sound.set_volume(1.0)
            self._sfx_cache[key] = sound
            self.last_error = ""
            return sound
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def _resolve_file(self, base_dir: Path, name: PathLike, extensions: tuple[str, ...]) -> Optional[Path]:
        raw = Path(name)
        candidates = []

        if raw.is_absolute():
            candidates.append(raw)
        elif raw.suffix:
            candidates.append(base_dir / raw)
            candidates.append(self.audio_root / raw)
        else:
            for ext in extensions:
                candidates.append(base_dir / f"{raw}{ext}")
                candidates.append(self.audio_root / f"{raw}{ext}")

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _effective_bgm_volume(self) -> float:
        return 0.0 if self.muted else self.bgm_volume

    @staticmethod
    def _clamp_volume(value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 1.0
