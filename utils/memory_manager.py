"""
Utilities for writing memory candidates after planner interactions.
"""

import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_ROOT = os.path.join(REPO_ROOT, "agent", "memory")
DEFAULT_LOCAL_MEMORY_ROOT = os.path.join(MEMORY_ROOT, "local")
LOCAL_MEMORY_ROOT = DEFAULT_LOCAL_MEMORY_ROOT
INBOX_ROOT = os.path.join(MEMORY_ROOT, "inbox")
GLOBAL_MEMORY_REL = "global_memory.md"
USER_PREFERENCES_REL = "user_preferences.md"


def get_local_memory_root() -> str:
    return os.getenv("AGENT_LOCAL_MEMORY_ROOT", LOCAL_MEMORY_ROOT)


class MemoryManager:
    """Writes hardware/global memory candidates to the inbox."""

    def __init__(self, profile_name: str, verbose: bool = True):
        self.profile_name = profile_name
        self.verbose = verbose
        self.hardware_inbox = os.path.join(INBOX_ROOT, "hardware")
        self.global_inbox = os.path.join(INBOX_ROOT, "global")
        os.makedirs(get_local_memory_root(), exist_ok=True)
        os.makedirs(self.hardware_inbox, exist_ok=True)
        os.makedirs(self.global_inbox, exist_ok=True)

    def record_interaction(
        self,
        status: str,
        original_request: Optional[str],
        execution_history: List[Dict],
        result: Dict,
    ) -> Dict[str, str]:
        """Create hardware and global memory candidate files for one interaction."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = f"{timestamp}-{self.profile_name}"

        hardware_path = os.path.join(self.hardware_inbox, f"{stem}.md")
        global_path = os.path.join(self.global_inbox, f"{stem}.md")

        with open(hardware_path, "w", encoding="utf-8") as file:
            file.write(
                self._render_candidate(
                    level="hardware",
                    status=status,
                    original_request=original_request,
                    execution_history=execution_history,
                    result=result,
                )
            )

        with open(global_path, "w", encoding="utf-8") as file:
            file.write(
                self._render_candidate(
                    level="global",
                    status=status,
                    original_request=original_request,
                    execution_history=execution_history,
                    result=result,
                )
            )

        persisted_memory = None
        if not any(step.get("action") == "store_memory" for step in execution_history):
            persisted_memory = self._persist_explicit_long_term_memory(original_request)

        if self.verbose:
            print("🧠 Memory candidates written:")
            print(f"   Hardware: {hardware_path}")
            print(f"   Global:   {global_path}")
            if persisted_memory:
                print(f"   Long-term: {persisted_memory}")

        return {
            "hardware": hardware_path,
            "global": global_path,
            **({"long_term": persisted_memory} if persisted_memory else {}),
        }

    def store_memory(
        self,
        content: str,
        scope: str = "global",
        category: str = "note",
    ) -> Optional[str]:
        entry = self._normalize_memory_entry(content=content, category=category)
        if not entry:
            return None

        normalized_scope = (scope or "global").strip().lower()
        normalized_category = (category or "note").strip().lower()
        if normalized_category == "preference":
            path = self._resolve_memory_write_path(USER_PREFERENCES_REL)
        elif normalized_scope in {"hardware", "profile", self.profile_name.lower()}:
            path = self._resolve_memory_write_path(os.path.join("hardware", f"{self.profile_name}_memory.md"))
        else:
            path = self._resolve_memory_write_path(GLOBAL_MEMORY_REL)

        if not os.path.exists(path):
            return None

        self._append_memory_entry(path, entry)
        return path

    def normalize_memory_entry(self, content: str, category: str = "note") -> Optional[str]:
        return self._normalize_memory_entry(content=content, category=category)

    def is_memory_entry_visible(self, path: str, content: str, category: str = "note") -> bool:
        normalized = self._normalize_memory_entry(content=content, category=category)
        if not normalized or not path or not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as file:
            snapshot = file.read()
        return normalized in snapshot

    def _render_candidate(
        self,
        level: str,
        status: str,
        original_request: Optional[str],
        execution_history: List[Dict],
        result: Dict,
    ) -> str:
        lessons = self._extract_lessons(level, status, execution_history, result)
        latest_action = execution_history[-1]["action"] if execution_history else "none"

        lines = [
            f"# Memory Candidate ({level})",
            "",
            "## Metadata",
            f"- profile: {self.profile_name}",
            f"- status: {status}",
            f"- created_at: {datetime.now().isoformat()}",
            "",
            "## Task",
            f"- request: {original_request or 'N/A'}",
            f"- steps_executed: {len(execution_history)}",
            f"- latest_action: {latest_action}",
            "",
            "## Candidate Lessons",
        ]

        lines.extend([f"- {lesson}" for lesson in lessons])
        lines.extend(
            [
                "",
                "## Result Snapshot",
                f"- is_complete: {result.get('is_complete', False)}",
                f"- waiting_for_input: {status.startswith('waiting_')}",
                f"- has_error: {'error' in result}",
            ]
        )

        if execution_history:
            lines.extend(["", "## Executed Actions"])
            lines.extend(
                [
                    f"- step {step.get('step_number')}: {step.get('action')} {step.get('parameters', {})}"
                    for step in execution_history[-5:]
                ]
            )

        return "\n".join(lines) + "\n"

    def _extract_lessons(
        self,
        level: str,
        status: str,
        execution_history: List[Dict],
        result: Dict,
    ) -> List[str]:
        lessons: List[str] = []

        if level == "hardware":
            if execution_history:
                latest_action = execution_history[-1].get("action")
                lessons.append(
                    f"The profile reached interaction status '{status}' after using action '{latest_action}'."
                )
            else:
                lessons.append("No executable step was produced before the interaction returned control.")

            if status.startswith("waiting_"):
                lessons.append("Clarification or coordination was required before safe continuation.")
            if result.get("is_complete"):
                lessons.append("The current one-step planning loop was sufficient to complete the task.")
            if "error" in result:
                lessons.append(f"An error surfaced to the planner: {result['error']}")
        else:
            if status.startswith("waiting_"):
                lessons.append("When uncertainty blocks execution, asking for targeted clarification is valuable.")
            if execution_history:
                lessons.append("Short step-wise plans remain easier to recover than long speculative plans.")
            if result.get("is_complete"):
                lessons.append("Completion summaries should preserve the final state and compact action history.")
            if "error" in result:
                lessons.append("Global memory should prefer robust recovery patterns over brittle assumptions.")

        return lessons

    def _persist_explicit_long_term_memory(self, original_request: Optional[str]) -> Optional[str]:
        request = (original_request or "").strip()
        if not request:
            return None

        lesson = self._extract_explicit_memory_lesson(request)
        if not lesson:
            return None

        path = self._resolve_memory_write_path(GLOBAL_MEMORY_REL)
        if not os.path.exists(path):
            return None

        self._append_memory_entry(path, lesson)
        return path

    def _extract_explicit_memory_lesson(self, request: str) -> Optional[str]:
        lowered = request.lower()
        explicit_memory_signals = [
            "记到memory",
            "记到 memory",
            "记进memory",
            "记进 memory",
            "记入memory",
            "记入 memory",
            "记住",
            "长期记忆",
            "长期偏好",
            "remember this",
            "remember that",
            "record this",
            "save this",
            "save to memory",
            "store this in memory",
            "preference",
        ]
        if not any(signal in lowered for signal in explicit_memory_signals):
            return None

        if any(token in lowered for token in ["strictly sugar-free", "无糖", "控糖", "拒绝摄入任何含糖饮料", "含糖饮料"]):
            return "User preference: strictly sugar-free; do not offer sugary drinks."

        if any(token in lowered for token in ["allergic", "allergy", "过敏"]):
            cleaned = self._compact_request(request)
            return f"User health/safety preference to remember: {cleaned}"

        if any(token in lowered for token in ["prefer", "preference", "喜欢", "偏好", "不喜欢", "忌口", "不要"]):
            cleaned = self._compact_request(request)
            return f"User preference to remember: {cleaned}"

        cleaned = self._compact_request(request)
        if not cleaned:
            return None
        return f"User explicitly asked to remember: {cleaned}"

    def _compact_request(self, request: str) -> str:
        compact = re.sub(r"\s+", " ", request).strip()
        compact = compact.rstrip("。.!！?？")
        return compact

    def _normalize_memory_entry(self, content: str, category: str) -> Optional[str]:
        cleaned = self._compact_request(content or "")
        if not cleaned:
            return None

        normalized_category = (category or "note").strip().lower()
        if normalized_category == "preference":
            if cleaned.lower().startswith("user preference:"):
                return cleaned
            return f"User preference: {cleaned}"
        if normalized_category in {"constraint", "restriction"}:
            if cleaned.lower().startswith("user constraint:"):
                return cleaned
            return f"User constraint: {cleaned}"
        if normalized_category == "safety":
            if cleaned.lower().startswith("user safety note:"):
                return cleaned
            return f"User safety note: {cleaned}"
        return cleaned

    def _append_memory_entry(self, path: str, entry: str) -> None:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

        if entry in content:
            return

        content = content.replace("- None.\n", "")
        content = content.replace("- None.", "")

        marker = "Current entries:\n"
        bullet = f"- {entry}\n"
        if marker in content:
            updated = content.replace(marker, f"{marker}{bullet}", 1)
        else:
            updated = content.rstrip() + f"\n\nCurrent entries:\n{bullet}"

        with open(path, "w", encoding="utf-8") as file:
            file.write(updated)

    def _resolve_memory_write_path(self, relative_path: str) -> str:
        local_path = os.path.join(get_local_memory_root(), relative_path)
        base_path = os.path.join(MEMORY_ROOT, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            if os.path.exists(base_path):
                shutil.copyfile(base_path, local_path)
            else:
                with open(local_path, "w", encoding="utf-8") as file:
                    file.write("")
        return local_path
