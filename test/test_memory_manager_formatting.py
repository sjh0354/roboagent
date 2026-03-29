import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.memory_manager import MemoryManager


def test_store_memory_replaces_none_placeholder(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.memory_manager.MEMORY_ROOT", str(tmp_path / "memory"))
    monkeypatch.setattr("utils.memory_manager.LOCAL_MEMORY_ROOT", str(tmp_path / "memory" / "local"))

    manager = MemoryManager(profile_name="ur5e", verbose=False)
    target = manager._resolve_memory_write_path("user_preferences.md")
    with open(target, "w", encoding="utf-8") as file:
        file.write(
            "# User Preferences\n\n"
            "Current entries:\n"
            "- None.\n"
        )

    path = manager.store_memory(
        content="strictly sugar-free; do not offer sugary drinks",
        scope="global",
        category="preference",
    )

    assert path is not None
    text = open(path, "r", encoding="utf-8").read()
    assert "- None." not in text
    assert "- User preference: strictly sugar-free; do not offer sugary drinks" in text
