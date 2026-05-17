"""
Shared local-memory helpers for memory ablation experiments.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_MEMORY_ROOT = REPO_ROOT / "agent" / "memory"


def local_memory_root() -> Path:
    return Path(os.getenv("AGENT_LOCAL_MEMORY_ROOT", AGENT_MEMORY_ROOT / "local"))


def initialize_clean_local_memory() -> Path:
    root = local_memory_root()
    if root.exists():
        shutil.rmtree(root)
    (root / "hardware").mkdir(parents=True, exist_ok=True)
    (root / "runtime").mkdir(parents=True, exist_ok=True)
    _copy_or_seed(AGENT_MEMORY_ROOT / "user_preferences.md", root / "user_preferences.md")
    _copy_or_seed(AGENT_MEMORY_ROOT / "global_memory.md", root / "global_memory.md")
    for profile in ("humanoid_g1", "ur5e"):
        _copy_or_seed(
            AGENT_MEMORY_ROOT / "hardware" / f"{profile}_memory.md",
            root / "hardware" / f"{profile}_memory.md",
        )
    return root


def read_user_preferences_snapshot() -> str:
    path = local_memory_root() / "user_preferences.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _copy_or_seed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copyfile(source, target)
    else:
        target.write_text("# Memory\n\n", encoding="utf-8")
