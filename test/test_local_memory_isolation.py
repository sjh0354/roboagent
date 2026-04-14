import os
import shutil

from experiments.preference_write_then_recall_ablation import (
    initialize_clean_local_memory,
    local_memory_root,
    read_user_preferences_snapshot,
)
from experiments.ur5e_memory_context_ablation import LocalMemorySandbox
from template.modular_prompt_loader import build_runtime_system_prompt
from utils.hierarchical_memory_manager import HierarchicalMemoryManager, get_runtime_root
from utils.memory_manager import MemoryManager, get_local_memory_root


def test_local_memory_override_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOCAL_MEMORY_ROOT", str(tmp_path / "local"))
    initialize_clean_local_memory()

    manager = MemoryManager(profile_name="ur5e", verbose=False)
    path = manager.store_memory(
        content="alpha unique preference",
        scope="global",
        category="preference",
    )

    snapshot = read_user_preferences_snapshot()
    prompt = build_runtime_system_prompt(
        profile_name="ur5e",
        original_request="recall test",
        execution_history=[],
        current_location="store",
        legacy_prompt="",
        include_memory=True,
    )

    assert path == str(tmp_path / "local" / "user_preferences.md")
    assert get_local_memory_root() == str(tmp_path / "local")
    assert local_memory_root() == tmp_path / "local"
    assert "alpha unique preference" in snapshot
    assert "alpha unique preference" in prompt


def test_local_memory_sandbox_isolated_and_restored():
    original_override = os.getenv("AGENT_LOCAL_MEMORY_ROOT")

    with LocalMemorySandbox():
        sandbox_root = local_memory_root()
        initialize_clean_local_memory()
        manager = MemoryManager(profile_name="ur5e", verbose=False)
        path = manager.store_memory(
            content="beta unique preference",
            scope="global",
            category="preference",
        )

        assert str(path).startswith(str(sandbox_root))
        assert "beta unique preference" in read_user_preferences_snapshot()

    if original_override is None:
        assert os.getenv("AGENT_LOCAL_MEMORY_ROOT") is None
    else:
        assert os.getenv("AGENT_LOCAL_MEMORY_ROOT") == original_override


def test_hierarchical_runtime_root_follows_local_memory_override(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    monkeypatch.setenv("AGENT_LOCAL_MEMORY_ROOT", str(local_root))

    manager = HierarchicalMemoryManager(profile_name="ur5e", verbose=False)

    assert get_runtime_root() == str(local_root / "runtime")
    assert manager.runtime_root == str(local_root / "runtime")
    assert (local_root / "runtime").exists()


def test_hierarchical_runtime_paths_survive_local_memory_copy(tmp_path, monkeypatch):
    source_root = tmp_path / "source_local"
    monkeypatch.setenv("AGENT_LOCAL_MEMORY_ROOT", str(source_root))
    manager = HierarchicalMemoryManager(profile_name="ur5e", verbose=False)
    session_id = manager.start_session()
    manager.append_raw_event(session_id=session_id, text="x" * 300, status="waiting_for_human")
    manager.maybe_rollup_raw(
        session_id=session_id,
        summary_callback=lambda level, source_text, metadata: {"topic": "demo", "summary": "demo summary"},
    )

    index = manager._load_index()
    entry = index["entries"][0]
    assert entry["path"].startswith("$LOCAL/runtime/")
    assert index["sessions"][session_id]["raw_path"].startswith("$LOCAL/runtime/")

    target_root = tmp_path / "target_local"
    shutil.copytree(source_root, target_root)
    monkeypatch.setenv("AGENT_LOCAL_MEMORY_ROOT", str(target_root))
    copied_manager = HierarchicalMemoryManager(profile_name="ur5e", verbose=False)

    copied_index = copied_manager._load_index()
    copied_entry = copied_index["entries"][0]
    copied_path = copied_manager._absolute_path(copied_entry["path"])

    assert copied_path.startswith(str(target_root))
    assert os.path.exists(copied_path)
