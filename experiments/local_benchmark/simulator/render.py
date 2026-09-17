"""State-consistent synthetic RGB-D rendering.

The renderer is intentionally simple, but it never selects an image independently
of the world.  Every RGB and depth frame is derived from the same public snapshot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LOCATIONS = {
    # Frozen 4x4 map covering every endpoint in formal_40.jsonl.  Keeping
    # distinct cells is essential: otherwise many formal locations would
    # collapse onto the same fallback pixel and invalidate the vision study.
    "desk": (85, 100), "shelf": (240, 100), "basket": (395, 100), "kitchen": (550, 100),
    "living_room": (85, 200), "hallway": (240, 200), "counter": (395, 200), "storage_bin": (550, 200),
    "bedroom": (85, 300), "entryway": (240, 300), "pantry": (395, 300), "reading_corner": (550, 300),
    "side_table": (85, 400), "tray": (240, 400), "upper_shelf": (395, 400), "workbench": (550, 400),
}


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _stable_color(label: str) -> tuple[int, int, int]:
    raw = hashlib.sha256(label.encode("utf-8")).digest()
    return (60 + raw[0] % 150, 60 + raw[1] % 150, 60 + raw[2] % 150)


def _location_xy(location: str, view: str) -> tuple[int, int]:
    base = LOCATIONS.get(location)
    if base is None:
        if location.startswith("near_"):
            source = location[5:]
            x, y = LOCATIONS.get(source, LOCATIONS["hallway"])
            base = (x + 45, y + 30)
        elif location.startswith("floor_near_"):
            target = location[len("floor_near_"):]
            x, y = LOCATIONS.get(target, LOCATIONS["hallway"])
            base = (x - 35, y + 45)
        elif location.startswith("corridor_to_"):
            base = LOCATIONS["hallway"]
        else:
            base = LOCATIONS["hallway"]
    x, y = base
    if view == "side":
        return (620 - x, min(430, y + (x % 3) * 8))
    return base


def render_observation(snapshot: Dict[str, Any], output_dir: Path, frame_index: int) -> Dict[str, Any]:
    """Render two RGB views and one aligned synthetic depth map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_paths: List[str] = []
    for view in ("front", "side"):
        image = Image.new("RGB", (640, 480), (238, 241, 245))
        draw = ImageDraw.Draw(image)
        draw.rectangle((12, 12, 628, 468), outline=(42, 48, 60), width=2)
        draw.text((24, 22), f"SIMULATED RGB | {view} | t={snapshot['sim_time']:.1f}s", fill=(12, 18, 28), font=_font(18))
        for name in LOCATIONS:
            x, y = _location_xy(name, view)
            draw.rounded_rectangle((x - 66, y - 35, x + 66, y + 35), radius=8, outline=(135, 145, 160), width=2)
            draw.text((x - 58, y - 31), name, fill=(70, 78, 92), font=_font(11))

        objects = snapshot.get("objects", {})
        offsets: Dict[str, int] = {}
        for obj, location in sorted(objects.items()):
            if location == "held":
                x, y = _location_xy(snapshot.get("robot_location", "desk"), view)
                x, y = x + 18, y - 62
            else:
                x, y = _location_xy(str(location), view)
            offset = offsets.get(str(location), 0)
            offsets[str(location)] = offset + 1
            x += (offset % 3 - 1) * 34
            y += (offset // 3) * 25
            color = _stable_color(obj)
            draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=color, outline=(20, 20, 20), width=2)
            draw.text((x - 28, y + 17), obj, fill=(10, 10, 12), font=_font(11), anchor="ma")

        rx, ry = _location_xy(snapshot.get("robot_location", "desk"), view)
        draw.polygon(((rx, ry - 25), (rx - 22, ry + 20), (rx + 22, ry + 20)), fill=(30, 100, 220), outline=(10, 35, 90))
        draw.text((rx, ry + 25), "robot", fill=(15, 45, 110), font=_font(12), anchor="ma")
        devices = ", ".join(f"{k}={v}" for k, v in sorted(snapshot.get("devices", {}).items())) or "none"
        draw.text((24, 440), f"devices: {devices}", fill=(20, 25, 35), font=_font(13))
        path = output_dir / f"frame_{frame_index:04d}_{view}.png"
        image.save(path)
        rgb_paths.append(str(path.resolve()))

    depth = np.full((480, 640), 2400, dtype=np.uint16)
    for obj, location in sorted(snapshot.get("objects", {}).items()):
        x, y = _location_xy(str(location), "front")
        yy, xx = np.ogrid[:480, :640]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= 17 ** 2
        depth[mask] = 700 + int(hashlib.sha256(obj.encode()).hexdigest()[:3], 16) % 700
    depth_path = output_dir / f"frame_{frame_index:04d}_depth.npy"
    np.save(depth_path, depth)
    depth_png_path = output_dir / f"frame_{frame_index:04d}_depth.png"
    Image.fromarray(depth).save(depth_png_path)
    metadata = {
        "frame_index": frame_index,
        "rgb_paths": rgb_paths,
        "depth_npy": str(depth_path.resolve()),
        "depth_png": str(depth_png_path.resolve()),
        "intrinsics": {"width": 640, "height": 480, "fx": 520.0, "fy": 520.0, "cx": 320.0, "cy": 240.0, "depth_scale": 1000.0},
        "extrinsics": {
            "front": {"translation_m": [0.0, -1.6, 1.3], "rotation_rpy": [0.0, 0.0, 0.0]},
            "side": {"translation_m": [1.4, 0.0, 1.2], "rotation_rpy": [0.0, 0.0, 1.57]},
        },
        "snapshot_sha256": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
    }
    meta_path = output_dir / f"frame_{frame_index:04d}.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    metadata["metadata_path"] = str(meta_path.resolve())
    return metadata
