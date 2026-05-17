"""
Memory sandbox helpers for UR5e memory-context ablation experiments.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from experiments.preference_write_then_recall_ablation import initialize_clean_local_memory


class LocalMemorySandbox:
    """Temporarily redirect AGENT_LOCAL_MEMORY_ROOT to an isolated directory."""

    def __init__(self):
        self._previous: Optional[str] = None
        self._tmpdir: Optional[str] = None

    def __enter__(self):
        self._previous = os.getenv("AGENT_LOCAL_MEMORY_ROOT")
        self._tmpdir = tempfile.mkdtemp(prefix="omniclaw-memory-")
        os.environ["AGENT_LOCAL_MEMORY_ROOT"] = str(Path(self._tmpdir) / "local")
        initialize_clean_local_memory()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._previous is None:
            os.environ.pop("AGENT_LOCAL_MEMORY_ROOT", None)
        else:
            os.environ["AGENT_LOCAL_MEMORY_ROOT"] = self._previous
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        return False
