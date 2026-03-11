"""
Time-ordered observation buffer for transient visual memory.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class ObservationFrame:
    timestamp: float
    image_path: Optional[str]
    state: Dict[str, Any] = field(default_factory=dict)
    source: str = "buffer"


class ObservationBuffer:
    """Fixed-size buffer of timestamped observation frames."""

    def __init__(self, max_frames: int = 120):
        self._frames: Deque[ObservationFrame] = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    def add_frame(self, frame: ObservationFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def get_frames_between(
        self,
        start_ts: float,
        end_ts: float,
        max_frames: Optional[int] = None,
    ) -> List[ObservationFrame]:
        with self._lock:
            frames = [frame for frame in self._frames if start_ts <= frame.timestamp <= end_ts]
        if max_frames and len(frames) > max_frames:
            frames = _sample_frames(frames, max_frames)
        return frames


def _sample_frames(frames: List[ObservationFrame], max_frames: int) -> List[ObservationFrame]:
    if len(frames) <= max_frames:
        return frames
    if max_frames <= 1:
        return [frames[-1]]

    selected: List[ObservationFrame] = []
    last_index = len(frames) - 1
    for index in range(max_frames):
        frame_index = round(index * last_index / (max_frames - 1))
        selected.append(frames[frame_index])

    deduped: List[ObservationFrame] = []
    seen = set()
    for frame in selected:
        key = (frame.timestamp, frame.image_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(frame)
    return deduped


def now_ts() -> float:
    return time.time()
