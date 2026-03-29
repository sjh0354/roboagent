"""
Runtime hierarchical memory management for planner context compaction.
"""

import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_ROOT = os.path.join(REPO_ROOT, "agent", "memory", "local", "runtime")
RAW_ROOT = os.path.join(RUNTIME_ROOT, "raw")
SHORT_TERM_ROOT = os.path.join(RUNTIME_ROOT, "short_term")
LONG_TERM_ROOT = os.path.join(RUNTIME_ROOT, "long_term")
OVERVIEW_ROOT = os.path.join(RUNTIME_ROOT, "overview")
INDEX_PATH = os.path.join(RUNTIME_ROOT, "index.json")


SummaryCallback = Callable[[str, str, Dict[str, Any]], Dict[str, str]]


class HierarchicalMemoryManager:
    """Manage raw, short-term, and long-term runtime summaries."""

    def __init__(
        self,
        profile_name: str,
        verbose: bool = True,
        runtime_root: Optional[str] = None,
    ):
        self.profile_name = profile_name
        self.verbose = verbose
        self.runtime_root = runtime_root or RUNTIME_ROOT
        self.raw_root = os.path.join(self.runtime_root, "raw")
        self.short_term_root = os.path.join(self.runtime_root, "short_term")
        self.long_term_root = os.path.join(self.runtime_root, "long_term")
        self.overview_root = os.path.join(self.runtime_root, "overview")
        self.index_path = os.path.join(self.runtime_root, "index.json")
        self.context_window_tokens = int(os.getenv("PLANNER_CONTEXT_WINDOW_TOKENS", "32000"))
        self.memory_trigger_ratio = float(os.getenv("MEMORY_TRIGGER_RATIO", "0.25"))
        self.force_summary_seconds = int(os.getenv("MEMORY_FORCE_SUMMARY_SECONDS", "7200"))
        self.memory_context_budget_ratio = float(os.getenv("MEMORY_CONTEXT_BUDGET_RATIO", "0.35"))
        self.memory_context_budget_tokens = int(os.getenv("MEMORY_CONTEXT_BUDGET_TOKENS", "0"))
        self.memory_overview_budget_ratio = float(os.getenv("MEMORY_OVERVIEW_BUDGET_RATIO", "0.35"))
        self.memory_long_term_budget_ratio = float(os.getenv("MEMORY_LONGTERM_BUDGET_RATIO", "0.25"))
        self.memory_short_term_budget_ratio = float(os.getenv("MEMORY_SHORTTERM_BUDGET_RATIO", "0.20"))
        self.memory_raw_budget_ratio = float(os.getenv("MEMORY_RAW_BUDGET_RATIO", "0.20"))
        self.overview_context_max_items = int(os.getenv("MEMORY_OVERVIEW_CONTEXT_MAX_ITEMS", "2"))
        self.short_term_context_max_items = int(os.getenv("MEMORY_SHORTTERM_MAX_ITEMS", "4"))
        self.long_term_context_max_items = int(os.getenv("MEMORY_LONGTERM_MAX_ITEMS", "4"))
        self.raw_tail_max_tokens = int(os.getenv("MEMORY_RAW_TAIL_MAX_TOKENS", "1200"))
        self.long_term_promotion_batch = int(os.getenv("MEMORY_LONGTERM_PROMOTION_BATCH", "3"))
        self.overview_promotion_batch = int(os.getenv("MEMORY_OVERVIEW_PROMOTION_BATCH", "3"))

        os.makedirs(self.raw_root, exist_ok=True)
        os.makedirs(self.short_term_root, exist_ok=True)
        os.makedirs(self.long_term_root, exist_ok=True)
        os.makedirs(self.overview_root, exist_ok=True)
        self._ensure_index()

    def configure_runtime(
        self,
        *,
        context_window_tokens: Optional[int] = None,
    ) -> None:
        if context_window_tokens:
            self.context_window_tokens = int(context_window_tokens)

    def start_session(
        self,
        session_id: Optional[str] = None,
        started_at: Optional[datetime] = None,
    ) -> str:
        started_at = started_at or datetime.now()
        session_id = session_id or f"{self.profile_name}-{started_at.strftime('%Y%m%d-%H%M%S')}"
        index = self._load_index()
        sessions = index.setdefault("sessions", {})
        if session_id not in sessions:
            raw_filename = f"{session_id}.jsonl"
            sessions[session_id] = {
                "device": self.profile_name,
                "created_at": started_at.isoformat(),
                "raw_path": self._relative_path(os.path.join(self.raw_root, raw_filename)),
                "next_raw_turn": 1,
                "last_summary_at": None,
                "last_active_summary_at": None,
                "last_raw_rollup_turn": 0,
            }
            self._save_index(index)
        return session_id

    def append_raw_event(
        self,
        session_id: str,
        text: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        cleaned = (text or "").strip()
        if not cleaned:
            return None

        created_at = created_at or datetime.now()
        index = self._load_index()
        session = self._get_or_create_session(index, session_id, created_at)
        turn_number = int(session.get("next_raw_turn", 1))
        event = {
            "id": f"{session_id}:raw:{turn_number}",
            "turn_number": turn_number,
            "created_at": created_at.isoformat(),
            "status": status,
            "text": cleaned,
            "metadata": metadata or {},
        }
        raw_path = self._absolute_path(session["raw_path"])
        with open(raw_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        session["next_raw_turn"] = turn_number + 1
        self._save_index(index)
        return event

    def maybe_rollup_raw(
        self,
        session_id: str,
        summary_callback: SummaryCallback,
        *,
        force: bool = False,
        created_at: Optional[datetime] = None,
        trigger_reason: str = "raw_tail",
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        created_at = created_at or datetime.now()
        index = self._load_index()
        session = self._get_or_create_session(index, session_id, created_at)
        last_rollup_turn = int(session.get("last_raw_rollup_turn", 0))
        raw_events = self._load_raw_events(session)
        pending_events = [event for event in raw_events if int(event.get("turn_number", 0)) > last_rollup_turn]
        if not pending_events:
            return None

        source_text = self._render_raw_events_text(pending_events)
        if not force and not self._should_rollup_raw(session, source_text, created_at):
            return None

        first_turn = int(pending_events[0].get("turn_number", 0))
        last_turn = int(pending_events[-1].get("turn_number", 0))
        entry = self._create_summary_entry(
            index=index,
            session_id=session_id,
            level="short_term",
            source_text=source_text,
            summary_callback=summary_callback,
            created_at=created_at,
            metadata={
                "trigger_reason": trigger_reason,
                "source_kind": "raw_rollup",
                "source_turn_range": [first_turn, last_turn],
                **(metadata_extra or {}),
            },
        )
        if not entry:
            return None

        session["last_summary_at"] = created_at.isoformat()
        session["last_raw_rollup_turn"] = last_turn
        self._save_index(index)
        promoted = self.maybe_promote_long_term(
            session_id=session_id,
            summary_callback=summary_callback,
            created_at=created_at,
        )
        overview = self.maybe_promote_overview(
            session_id=session_id,
            summary_callback=summary_callback,
            created_at=created_at,
        )
        return {"short_term": entry, "long_term": promoted, "overview": overview}

    def store_active_context_summary(
        self,
        session_id: str,
        source_text: str,
        summary_callback: SummaryCallback,
        *,
        created_at: Optional[datetime] = None,
        trigger_reason: str = "active_context",
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        created_at = created_at or datetime.now()
        cleaned = (source_text or "").strip()
        if not cleaned:
            return None

        index = self._load_index()
        session = self._get_or_create_session(index, session_id, created_at)
        entry = self._create_summary_entry(
            index=index,
            session_id=session_id,
            level="short_term",
            source_text=cleaned,
            summary_callback=summary_callback,
            created_at=created_at,
            metadata={
                "trigger_reason": trigger_reason,
                "source_kind": "active_context",
                **(metadata_extra or {}),
            },
        )
        if not entry:
            return None

        session["last_active_summary_at"] = created_at.isoformat()
        self._save_index(index)
        self.maybe_promote_long_term(
            session_id=session_id,
            summary_callback=summary_callback,
            created_at=created_at,
        )
        self.maybe_promote_overview(
            session_id=session_id,
            summary_callback=summary_callback,
            created_at=created_at,
        )
        return entry

    def maybe_promote_long_term(
        self,
        session_id: str,
        summary_callback: SummaryCallback,
        *,
        created_at: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        created_at = created_at or datetime.now()
        index = self._load_index()
        source_entries = [
            entry for entry in index.get("entries", [])
            if entry.get("device") == self.profile_name
            and entry.get("level") == "short_term"
            and entry.get("include_in_context", True)
            and not entry.get("promoted_to")
        ]
        if not self._should_promote_long_term(source_entries):
            return None

        source_entries = sorted(source_entries, key=lambda item: item.get("created_at", ""))[-self.long_term_promotion_batch:]
        combined_text = "\n\n".join(
            f"Topic: {entry.get('topic', 'unknown')}\n{self._read_text(self._absolute_path(entry['path']))}"
            for entry in source_entries
        )
        entry = self._create_summary_entry(
            index=index,
            session_id=session_id,
            level="long_term",
            source_text=combined_text,
            summary_callback=summary_callback,
            created_at=created_at,
            metadata={
                "trigger_reason": "promotion",
                "source_kind": "short_term_rollup",
                "source_ids": [item["id"] for item in source_entries],
            },
        )
        if not entry:
            return None

        for source in source_entries:
            source["include_in_context"] = False
            source["promoted_to"] = entry["id"]
        self._save_index(index)
        return entry

    def maybe_promote_overview(
        self,
        session_id: str,
        summary_callback: SummaryCallback,
        *,
        created_at: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        created_at = created_at or datetime.now()
        index = self._load_index()
        source_entries = [
            entry for entry in index.get("entries", [])
            if entry.get("device") == self.profile_name
            and entry.get("level") == "long_term"
            and entry.get("include_in_context", True)
            and not entry.get("promoted_to")
        ]
        if not self._should_promote_overview(source_entries):
            return None

        source_entries = sorted(source_entries, key=lambda item: item.get("created_at", ""))[-self.overview_promotion_batch:]
        combined_text = "\n\n".join(
            f"Topic: {entry.get('topic', 'unknown')}\n{self._read_text(self._absolute_path(entry['path']))}"
            for entry in source_entries
        )
        entry = self._create_summary_entry(
            index=index,
            session_id=session_id,
            level="long_term_overview",
            source_text=combined_text,
            summary_callback=summary_callback,
            created_at=created_at,
            metadata={
                "trigger_reason": "overview_promotion",
                "source_kind": "long_term_rollup",
                "source_ids": [item["id"] for item in source_entries],
            },
        )
        if not entry:
            return None

        for source in source_entries:
            source["include_in_context"] = False
            source["promoted_to"] = entry["id"]
        self._save_index(index)
        return entry

    def build_context_block(self, session_id: Optional[str] = None) -> Optional[str]:
        index = self._load_index()
        budget_tokens = self._get_context_budget_tokens()
        bucket_tokens = self._allocate_bucket_tokens(budget_tokens)
        sections: List[str] = []

        overview_entries = [
            entry for entry in index.get("entries", [])
            if entry.get("device") == self.profile_name
            and entry.get("level") == "long_term_overview"
            and entry.get("include_in_context", True)
        ]
        overview_entries = sorted(overview_entries, key=lambda item: item.get("created_at", ""))[-self.overview_context_max_items:]
        overview_section = self._render_context_entries_with_budget(
            title="LONG-TERM OVERVIEW",
            entries=overview_entries,
            max_tokens=bucket_tokens["overview"],
        )
        if overview_section:
            sections.append(overview_section)

        long_term_entries = [
            entry for entry in index.get("entries", [])
            if entry.get("device") == self.profile_name
            and entry.get("level") == "long_term"
            and entry.get("include_in_context", True)
        ]
        long_term_entries = sorted(long_term_entries, key=lambda item: item.get("created_at", ""))[-self.long_term_context_max_items:]
        long_term_section = self._render_context_entries_with_budget(
            title="LONG-TERM HISTORY SUMMARIES",
            entries=long_term_entries,
            max_tokens=bucket_tokens["long_term"],
        )
        if long_term_section:
            sections.append(long_term_section)

        short_term_entries = [
            entry for entry in index.get("entries", [])
            if entry.get("device") == self.profile_name
            and entry.get("level") == "short_term"
            and entry.get("include_in_context", True)
        ]
        short_term_entries = sorted(short_term_entries, key=lambda item: item.get("created_at", ""))[-self.short_term_context_max_items:]
        short_term_section = self._render_context_entries_with_budget(
            title="SHORT-TERM HISTORY SUMMARIES",
            entries=short_term_entries,
            max_tokens=bucket_tokens["short_term"],
        )
        if short_term_section:
            sections.append(short_term_section)

        if session_id:
            session = index.get("sessions", {}).get(session_id)
            if session:
                last_rollup_turn = int(session.get("last_raw_rollup_turn", 0))
                raw_events = self._load_raw_events(session)
                pending_events = [event for event in raw_events if int(event.get("turn_number", 0)) > last_rollup_turn]
                raw_tail = self._truncate_text_to_tokens(
                    self._render_raw_events_text(pending_events),
                    min(self.raw_tail_max_tokens, bucket_tokens["raw"]),
                    from_end=True,
                )
                if raw_tail:
                    sections.append(f"[CURRENT UNSUMMARIZED MEMORY]\n{raw_tail}")

        if not sections:
            return None
        return "[HIERARCHICAL MEMORY]\n" + "\n\n".join(section for section in sections if section.strip())

    def _should_promote_long_term(self, source_entries: List[Dict[str, Any]]) -> bool:
        if len(source_entries) >= self.long_term_promotion_batch:
            return True
        short_term_budget = self._allocate_bucket_tokens(self._get_context_budget_tokens())["short_term"]
        total_tokens = sum(
            self._estimate_tokens(self._extract_summary_body(self._read_text(self._absolute_path(entry["path"]))))
            for entry in source_entries
        )
        return total_tokens > short_term_budget

    def _should_promote_overview(self, source_entries: List[Dict[str, Any]]) -> bool:
        if len(source_entries) >= self.overview_promotion_batch:
            return True
        long_term_budget = self._allocate_bucket_tokens(self._get_context_budget_tokens())["long_term"]
        total_tokens = sum(
            self._estimate_tokens(self._extract_summary_body(self._read_text(self._absolute_path(entry["path"]))))
            for entry in source_entries
        )
        return total_tokens > long_term_budget

    def _get_context_budget_tokens(self) -> int:
        if self.memory_context_budget_tokens > 0:
            return self.memory_context_budget_tokens
        return max(600, int(self.context_window_tokens * self.memory_context_budget_ratio))

    def _allocate_bucket_tokens(self, total_budget: int) -> Dict[str, int]:
        ratios = {
            "overview": self.memory_overview_budget_ratio,
            "long_term": self.memory_long_term_budget_ratio,
            "short_term": self.memory_short_term_budget_ratio,
            "raw": self.memory_raw_budget_ratio,
        }
        ratio_sum = sum(max(0.0, value) for value in ratios.values()) or 1.0
        allocated: Dict[str, int] = {}
        remaining = total_budget
        ordered_keys = ["overview", "long_term", "short_term", "raw"]
        for key in ordered_keys[:-1]:
            tokens = max(120, int(total_budget * ratios[key] / ratio_sum))
            allocated[key] = min(tokens, remaining)
            remaining -= allocated[key]
        allocated[ordered_keys[-1]] = max(120, remaining)
        return allocated

    def _should_rollup_raw(self, session: Dict[str, Any], source_text: str, created_at: datetime) -> bool:
        raw_trigger_tokens = max(1, int(self.context_window_tokens * self.memory_trigger_ratio))
        if self._estimate_tokens(source_text) >= raw_trigger_tokens:
            return True
        last_summary_at = session.get("last_summary_at")
        if last_summary_at:
            try:
                last_summary_dt = datetime.fromisoformat(last_summary_at)
            except ValueError:
                return False
            return created_at - last_summary_dt >= timedelta(seconds=self.force_summary_seconds)
        return False

    def _create_summary_entry(
        self,
        *,
        index: Dict[str, Any],
        session_id: str,
        level: str,
        source_text: str,
        summary_callback: SummaryCallback,
        created_at: datetime,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        payload = summary_callback(level, source_text, metadata) or {}
        summary_body = (payload.get("summary") or "").strip()
        topic = self._slugify(payload.get("topic") or metadata.get("fallback_topic") or "general")
        if not summary_body:
            return None

        timestamp = created_at.strftime("%Y%m%d-%H%M%S")
        entry_id = f"{timestamp}-{topic}-{self.profile_name}"
        filename = f"{entry_id}.md"
        if level == "short_term":
            base_dir = self.short_term_root
        elif level == "long_term":
            base_dir = self.long_term_root
        else:
            base_dir = self.overview_root
        path = os.path.join(base_dir, filename)
        entry = {
            "id": entry_id,
            "level": level,
            "device": self.profile_name,
            "topic": topic,
            "created_at": created_at.isoformat(),
            "path": self._relative_path(path),
            "source_session_id": session_id,
            "source_turn_range": metadata.get("source_turn_range"),
            "source_ids": metadata.get("source_ids", []),
            "source_kind": metadata.get("source_kind", "summary"),
            "trigger_reason": metadata.get("trigger_reason"),
            "include_in_context": True,
            "promoted_to": None,
            "pinned": False,
        }
        markdown = self._render_summary_markdown(entry=entry, summary_body=summary_body)
        with open(path, "w", encoding="utf-8") as file:
            file.write(markdown)
        index.setdefault("entries", []).append(entry)
        return entry

    def _render_summary_markdown(self, entry: Dict[str, Any], summary_body: str) -> str:
        lines = [
            f"# {entry['level'].replace('_', ' ').title()} Summary: {entry['topic']}",
            "",
            "Metadata:",
            f"- id: {entry['id']}",
            f"- device: {entry['device']}",
            f"- created_at: {entry['created_at']}",
        ]
        if entry.get("source_turn_range"):
            start_turn, end_turn = entry["source_turn_range"]
            lines.append(f"- source_turn_range: {start_turn}-{end_turn}")
        if entry.get("source_ids"):
            lines.append(f"- source_ids: {', '.join(entry['source_ids'])}")
        lines.extend(["", "Summary:", summary_body.strip(), ""])
        return "\n".join(lines)

    def _render_context_entries(self, title: str, entries: List[Dict[str, Any]]) -> str:
        lines = [f"[{title}]"]
        for entry in entries:
            path = self._absolute_path(entry["path"])
            summary_text = self._extract_summary_body(self._read_text(path))
            created_at = entry.get("created_at", "")[:19]
            lines.append(f"- {created_at} | {entry.get('topic', 'general')}: {summary_text}")
        return "\n".join(lines)

    def _render_context_entries_with_budget(
        self,
        title: str,
        entries: List[Dict[str, Any]],
        max_tokens: int,
    ) -> str:
        if not entries or max_tokens <= 0:
            return ""
        selected: List[str] = []
        used_tokens = self._estimate_tokens(f"[{title}]")
        for entry in reversed(entries):
            path = self._absolute_path(entry["path"])
            summary_text = self._extract_summary_body(self._read_text(path))
            created_at = entry.get("created_at", "")[:19]
            line = f"- {created_at} | {entry.get('topic', 'general')}: {summary_text}"
            line_tokens = self._estimate_tokens(line)
            if selected and used_tokens + line_tokens > max_tokens:
                continue
            if not selected and line_tokens > max_tokens:
                clipped_summary = self._truncate_text_to_tokens(summary_text, max(40, max_tokens - 20))
                line = f"- {created_at} | {entry.get('topic', 'general')}: {clipped_summary}"
                line_tokens = self._estimate_tokens(line)
            selected.insert(0, line)
            used_tokens += line_tokens
            if used_tokens >= max_tokens:
                break
        if not selected:
            return ""
        return "\n".join([f"[{title}]"] + selected)

    def _render_raw_events_text(self, raw_events: List[Dict[str, Any]]) -> str:
        chunks = []
        for event in raw_events:
            created_at = event.get("created_at", "")[:19]
            status = event.get("status", "unknown")
            text = (event.get("text") or "").strip()
            if not text:
                continue
            chunks.append(f"[{created_at}] ({status})\n{text}")
        return "\n\n".join(chunks)

    def _load_raw_events(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_path = self._absolute_path(session["raw_path"])
        if not os.path.exists(raw_path):
            return []
        events: List[Dict[str, Any]] = []
        with open(raw_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def _ensure_index(self) -> None:
        if os.path.exists(self.index_path):
            return
        with open(self.index_path, "w", encoding="utf-8") as file:
            json.dump({"sessions": {}, "entries": []}, file, ensure_ascii=False, indent=2)

    def _load_index(self) -> Dict[str, Any]:
        self._ensure_index()
        with open(self.index_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {"sessions": {}, "entries": []}
        data.setdefault("sessions", {})
        data.setdefault("entries", [])
        return data

    def _save_index(self, index: Dict[str, Any]) -> None:
        with open(self.index_path, "w", encoding="utf-8") as file:
            json.dump(index, file, ensure_ascii=False, indent=2)

    def _get_or_create_session(
        self,
        index: Dict[str, Any],
        session_id: str,
        created_at: datetime,
    ) -> Dict[str, Any]:
        sessions = index.setdefault("sessions", {})
        session = sessions.get(session_id)
        if session:
            return session
        raw_filename = f"{session_id}.jsonl"
        session = {
            "device": self.profile_name,
            "created_at": created_at.isoformat(),
            "raw_path": self._relative_path(os.path.join(self.raw_root, raw_filename)),
            "next_raw_turn": 1,
            "last_summary_at": None,
            "last_active_summary_at": None,
            "last_raw_rollup_turn": 0,
        }
        sessions[session_id] = session
        return session

    def _extract_summary_body(self, markdown_text: str) -> str:
        marker = "Summary:\n"
        if marker not in markdown_text:
            return self._truncate_text_to_tokens(markdown_text.strip(), 80)
        summary = markdown_text.split(marker, 1)[1].strip()
        return self._truncate_text_to_tokens(summary, 80)

    def _truncate_text_to_tokens(self, text: str, max_tokens: int, from_end: bool = False) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        if self._estimate_tokens(cleaned) <= max_tokens:
            return cleaned
        approx_chars = max_tokens * 4
        if from_end:
            truncated = cleaned[-approx_chars:].lstrip()
            return "... " + truncated
        truncated = cleaned[:approx_chars].rstrip()
        return truncated + " ..."

    def _estimate_tokens(self, text: str) -> int:
        cleaned = (text or "").strip()
        if not cleaned:
            return 0
        return max(1, len(cleaned) // 4)

    def _slugify(self, raw_value: str) -> str:
        compact = re.sub(r"\s+", "_", (raw_value or "").strip().lower())
        compact = re.sub(r"[^a-z0-9_\-]+", "_", compact)
        compact = re.sub(r"_+", "_", compact).strip("_")
        return compact[:48] or "general"

    def _relative_path(self, path: str) -> str:
        return os.path.relpath(path, REPO_ROOT)

    def _absolute_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(REPO_ROOT, path)

    def _read_text(self, path: str) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
