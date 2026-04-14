import os
import sys
from datetime import datetime


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hierarchical_memory_manager import HierarchicalMemoryManager


def _stub_summary_callback(level, source_text, metadata):
    topic = metadata.get("trigger_reason") or level
    if level == "long_term":
        topic = "session_rollup"
    summary = f"{level} summary for {topic}: {source_text.splitlines()[0][:80]}"
    return {"topic": topic, "summary": summary}


def test_raw_rollup_creates_short_term_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANNER_CONTEXT_WINDOW_TOKENS", "100")
    monkeypatch.setenv("MEMORY_TRIGGER_RATIO", "0.25")
    manager = HierarchicalMemoryManager(
        profile_name="humanoid_g1",
        verbose=False,
        runtime_root=str(tmp_path / "runtime"),
    )
    session_id = manager.start_session(
        session_id="humanoid_g1-20260327-120000",
        started_at=datetime(2026, 3, 27, 12, 0, 0),
    )
    manager.append_raw_event(
        session_id=session_id,
        text="A" * 240,
        status="waiting_for_human",
    )

    result = manager.maybe_rollup_raw(
        session_id=session_id,
        summary_callback=_stub_summary_callback,
    )

    assert result is not None
    index = manager._load_index()
    assert len(index["entries"]) == 1
    entry = index["entries"][0]
    assert entry["level"] == "short_term"
    assert entry["include_in_context"] is True
    assert index["sessions"][session_id]["last_raw_rollup_turn"] == 1


def test_long_term_promotion_retires_short_term_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_LONGTERM_PROMOTION_BATCH", "3")
    monkeypatch.setenv("MEMORY_OVERVIEW_PROMOTION_BATCH", "10")
    manager = HierarchicalMemoryManager(
        profile_name="ur5e",
        verbose=False,
        runtime_root=str(tmp_path / "runtime"),
    )
    session_id = manager.start_session(
        session_id="ur5e-20260327-120000",
        started_at=datetime(2026, 3, 27, 12, 0, 0),
    )

    for idx in range(3):
        entry = manager.store_active_context_summary(
            session_id=session_id,
            source_text=f"context chunk {idx}",
            summary_callback=_stub_summary_callback,
            created_at=datetime(2026, 3, 27, 12, idx, 0),
            trigger_reason=f"chunk_{idx}",
        )
        assert entry is not None

    index = manager._load_index()
    short_terms = [entry for entry in index["entries"] if entry["level"] == "short_term"]
    long_terms = [entry for entry in index["entries"] if entry["level"] == "long_term"]

    assert len(short_terms) == 3
    assert len(long_terms) == 1
    assert all(entry["include_in_context"] is False for entry in short_terms)
    assert all(entry["promoted_to"] == long_terms[0]["id"] for entry in short_terms)


def test_overview_promotion_retires_long_term_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_LONGTERM_PROMOTION_BATCH", "2")
    monkeypatch.setenv("MEMORY_OVERVIEW_PROMOTION_BATCH", "2")
    manager = HierarchicalMemoryManager(
        profile_name="humanoid_g1",
        verbose=False,
        runtime_root=str(tmp_path / "runtime"),
    )
    session_id = manager.start_session(
        session_id="humanoid_g1-20260327-120000",
        started_at=datetime(2026, 3, 27, 12, 0, 0),
    )

    for idx in range(4):
        manager.store_active_context_summary(
            session_id=session_id,
            source_text=f"memory block {idx}",
            summary_callback=_stub_summary_callback,
            created_at=datetime(2026, 3, 27, 12, idx, 0),
            trigger_reason=f"segment_{idx}",
        )

    index = manager._load_index()
    long_terms = [entry for entry in index["entries"] if entry["level"] == "long_term"]
    overviews = [entry for entry in index["entries"] if entry["level"] == "long_term_overview"]

    assert len(long_terms) == 2
    assert len(overviews) == 1
    assert all(entry["include_in_context"] is False for entry in long_terms)
    assert all(entry["promoted_to"] == overviews[0]["id"] for entry in long_terms)


def test_context_block_uses_budgeted_layers_and_pending_raw_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_LONGTERM_PROMOTION_BATCH", "2")
    monkeypatch.setenv("MEMORY_OVERVIEW_PROMOTION_BATCH", "2")
    monkeypatch.setenv("MEMORY_CONTEXT_BUDGET_TOKENS", "220")
    monkeypatch.setenv("MEMORY_OVERVIEW_BUDGET_RATIO", "0.50")
    monkeypatch.setenv("MEMORY_LONGTERM_BUDGET_RATIO", "0.20")
    monkeypatch.setenv("MEMORY_SHORTTERM_BUDGET_RATIO", "0.15")
    monkeypatch.setenv("MEMORY_RAW_BUDGET_RATIO", "0.15")
    manager = HierarchicalMemoryManager(
        profile_name="humanoid_g1",
        verbose=False,
        runtime_root=str(tmp_path / "runtime"),
    )
    session_id = manager.start_session(
        session_id="humanoid_g1-20260327-120000",
        started_at=datetime(2026, 3, 27, 12, 0, 0),
    )

    for idx in range(4):
        manager.store_active_context_summary(
            session_id=session_id,
            source_text=f"memory block {idx} " + ("X" * 120),
            summary_callback=_stub_summary_callback,
            created_at=datetime(2026, 3, 27, 12, idx, 0),
            trigger_reason=f"segment_{idx}",
        )

    manager.append_raw_event(
        session_id=session_id,
        text="unsummarized raw tail",
        status="returned",
        created_at=datetime(2026, 3, 27, 12, 30, 0),
    )

    context = manager.build_context_block(session_id=session_id)

    assert context is not None
    assert "[LONG-TERM OVERVIEW]" in context
    assert "overview" in context
    assert "unsummarized raw tail" in context


def test_short_term_context_only_injects_current_needs_and_guardrail(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_CONTEXT_BUDGET_TOKENS", "260")
    monkeypatch.setenv("MEMORY_OVERVIEW_BUDGET_RATIO", "0.10")
    monkeypatch.setenv("MEMORY_LONGTERM_BUDGET_RATIO", "0.10")
    monkeypatch.setenv("MEMORY_SHORTTERM_BUDGET_RATIO", "0.60")
    monkeypatch.setenv("MEMORY_RAW_BUDGET_RATIO", "0.20")

    manager = HierarchicalMemoryManager(
        profile_name="ur5e",
        verbose=False,
        runtime_root=str(tmp_path / "runtime"),
    )
    session_id = manager.start_session(
        session_id="ur5e-20260327-120000",
        started_at=datetime(2026, 3, 27, 12, 0, 0),
    )

    manager.store_active_context_summary(
        session_id=session_id,
        source_text="demo source",
        summary_callback=lambda level, source_text, metadata: {
            "topic": "coordination_completion",
            "summary": "\n".join(
                [
                    "Current needs:",
                    "- user is feeling sleepy",
                    "- user has a headache",
                    "",
                    "Forbidden items:",
                    "- Do not offer sugary drinks.",
                    "",
                    "Decision rules:",
                    "- prioritize those that offer sustained energy and mental stimulation.",
                    "",
                    "Soft preferences:",
                    "- prioritize items that provide sustained energy",
                ]
            ),
        },
        created_at=datetime(2026, 3, 27, 12, 1, 0),
        trigger_reason="coordination_completion",
    )

    context = manager.build_context_block(session_id=session_id)

    assert context is not None
    assert "[SHORT-TERM NEEDS UNDER LONG-TERM CONSTRAINTS]" in context
    assert "These temporary needs must not override long-term preferences" in context
    assert "need: user is feeling sleepy" in context
    assert "need: user has a headache" in context
    assert "forbidden: Do not offer sugary drinks." in context
    assert "prioritize those that offer sustained energy" not in context
    assert "prioritize items that provide sustained energy" not in context
