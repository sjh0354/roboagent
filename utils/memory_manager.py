"""
Utilities for writing memory candidates after planner interactions.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_ROOT = os.path.join(REPO_ROOT, "agent", "memory")
INBOX_ROOT = os.path.join(MEMORY_ROOT, "inbox")


class MemoryManager:
    """Writes hardware/global memory candidates to the inbox."""

    def __init__(self, profile_name: str, verbose: bool = True):
        self.profile_name = profile_name
        self.verbose = verbose
        self.hardware_inbox = os.path.join(INBOX_ROOT, "hardware")
        self.global_inbox = os.path.join(INBOX_ROOT, "global")
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

        if self.verbose:
            print("🧠 Memory candidates written:")
            print(f"   Hardware: {hardware_path}")
            print(f"   Global:   {global_path}")

        return {
            "hardware": hardware_path,
            "global": global_path,
        }

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
