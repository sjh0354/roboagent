"""
Mock external music/audio API for cross-domain orchestration experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


VALID_ACTIONS = {"play", "stop"}
VALID_AUDIO_TYPES = {"white_noise", "rain_noise", "brown_noise", "ambient"}


@dataclass
class MockMusicResponse:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class MockMusicAPI:
    """In-process audio API with a small stable contract."""

    def play(self, payload: Dict[str, Any]) -> MockMusicResponse:
        action = payload.get("action", "play")
        audio_type = payload.get("audio_type") or payload.get("type")
        track = payload.get("track") or "default"
        volume = payload.get("volume", 35)

        if action not in VALID_ACTIONS:
            return MockMusicResponse(
                success=False,
                error_code="INVALID_ACTION",
                error_message="action must be 'play' or 'stop'",
            )
        if action == "play" and audio_type not in VALID_AUDIO_TYPES:
            return MockMusicResponse(
                success=False,
                error_code="INVALID_AUDIO_TYPE",
                error_message="audio_type must be one of: white_noise, rain_noise, brown_noise, ambient",
            )
        if not isinstance(volume, int) or volume < 0 or volume > 100:
            return MockMusicResponse(
                success=False,
                error_code="INVALID_VOLUME",
                error_message="volume must be an integer from 0 to 100",
            )

        return MockMusicResponse(
            success=True,
            data={
                "service": "mock_music",
                "action": action,
                "audio_type": audio_type,
                "track": track,
                "volume": volume,
            },
        )
