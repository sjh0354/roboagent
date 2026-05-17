"""
Helpers for assembling modular agent prompts from markdown files.
"""

import os
from typing import List, Tuple

from utils.skill_resolver import SkillResolver


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_ROOT = os.path.join(REPO_ROOT, "agent")
DEFAULT_LOCAL_MEMORY_ROOT = os.path.join(AGENT_ROOT, "memory", "local")


def _local_memory_root() -> str:
    return os.getenv("AGENT_LOCAL_MEMORY_ROOT", DEFAULT_LOCAL_MEMORY_ROOT)


def _read_text(relative_path: str) -> str:
    path = os.path.join(AGENT_ROOT, relative_path)
    return _read_text_from_path(path)


def _read_text_from_path(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def _optional_read(relative_path: str) -> str:
    path = _resolve_optional_path(relative_path)
    if not os.path.exists(path):
        return ""
    return _read_text_from_path(path)


def _resolve_optional_path(relative_path: str) -> str:
    if relative_path.startswith("memory/"):
        suffix = relative_path[len("memory/"):]
        local_path = os.path.join(_local_memory_root(), suffix)
        if os.path.exists(local_path):
            return local_path
    return os.path.join(AGENT_ROOT, relative_path)


def _build_sections(
    profile_name: str,
    include_memory: bool,
    include_skill_details: bool,
    skill_paths: List[str] | None = None,
    scenario_paths: List[str] | None = None,
) -> List[Tuple[str, str]]:
    sections = [
        ("Bootstrap", _read_text("bootstrap.md")),
        ("Soul", _read_text("soul.md")),
        ("High Priority User Preferences", _optional_read("memory/user_preferences.md")),
        ("Identity", _read_text(f"profiles/{profile_name}/identity.md")),
        ("Skills Index", _read_text(f"profiles/{profile_name}/skills_index.md")),
    ]

    if include_memory:
        sections.append(("Global Memory", _optional_read("memory/global_memory.md")))
        hardware_memory = _optional_read(f"memory/hardware/{profile_name}_memory.md")
        if hardware_memory:
            sections.append(("Hardware Memory", hardware_memory))

    if include_skill_details:
        for relative_path in scenario_paths or []:
            sections.append((f"Scenario: {relative_path}", _read_text(relative_path)))
        for relative_path in skill_paths or _list_skill_paths(profile_name):
            sections.append((f"Skill: {relative_path}", _read_text(relative_path)))

    return [(title, body) for title, body in sections if body]


def _list_skill_paths(profile_name: str) -> List[str]:
    common_paths = [
        "skills/common/speak_and_report/SKILL.md",
        "skills/common/send_agent_message/SKILL.md",
        "skills/common/observe_scene/SKILL.md",
        "skills/common/query_weather/SKILL.md",
    ]

    profile_paths = {
        "humanoid_g1": [
            "skills/common/search_web/SKILL.md",
            "skills/humanoid_g1/control_home_devices/SKILL.md",
            "skills/humanoid_g1/navigate_rooms/SKILL.md",
            "skills/humanoid_g1/manipulate_objects/SKILL.md",
        ],
        "ur5e": [
            "skills/ur5e/store_pick_and_place/SKILL.md",
            "skills/ur5e/observe_workspace/SKILL.md",
        ],
    }

    return common_paths + profile_paths.get(profile_name, [])


def build_modular_system_prompt(
    profile_name: str,
    legacy_prompt: str = "",
    include_memory: bool = True,
    include_skill_details: bool = False,
) -> str:
    """
    Assemble a modular system prompt from agent markdown files.

    Legacy prompt text can remain enabled during the migration by setting
    `AGENT_INCLUDE_LEGACY_PROMPT=1` (default).
    """
    sections = _build_sections(profile_name, include_memory, include_skill_details)
    rendered_sections = [f"# {title}\n\n{body}" for title, body in sections]
    prompt = "\n\n".join(rendered_sections)

    include_legacy = os.getenv("AGENT_INCLUDE_LEGACY_PROMPT", "1") == "1"
    if include_legacy and legacy_prompt:
        prompt = (
            f"{prompt}\n\n"
            "# Legacy Compatibility Prompt\n\n"
            "The following legacy prompt remains enabled during migration to keep "
            "behavior stable. Prefer the modular sections above when they are more specific.\n\n"
            f"{legacy_prompt.strip()}"
        )

    return prompt


def build_runtime_system_prompt(
    profile_name: str,
    original_request: str,
    execution_history: List[dict] | None,
    current_location: str | None = None,
    legacy_prompt: str = "",
    include_memory: bool = True,
) -> str:
    """
    Assemble a runtime prompt with only the currently relevant skill details.
    """
    resolver = SkillResolver(profile_name=profile_name)
    active_skill_paths = resolver.resolve(
        original_request=original_request,
        execution_history=execution_history or [],
        current_location=current_location,
    )
    active_scenario_paths = _resolve_runtime_scenarios(
        profile_name=profile_name,
        original_request=original_request,
    )
    sections = _build_sections(
        profile_name=profile_name,
        include_memory=include_memory,
        include_skill_details=True,
        skill_paths=active_skill_paths,
        scenario_paths=active_scenario_paths,
    )
    rendered_sections = [f"# {title}\n\n{body}" for title, body in sections]
    prompt = "\n\n".join(rendered_sections)

    active_skill_names = "\n".join([f"- {os.path.basename(os.path.dirname(path))}" for path in active_skill_paths])
    if active_skill_names:
        prompt = f"{prompt}\n\n# Active Skills\n\n{active_skill_names}"

    active_scenario_names = "\n".join(
        [f"- {os.path.splitext(os.path.basename(path))[0]}" for path in active_scenario_paths]
    )
    if active_scenario_names:
        prompt = f"{prompt}\n\n# Active Scenarios\n\n{active_scenario_names}"

    include_legacy = os.getenv("AGENT_INCLUDE_LEGACY_PROMPT", "1") == "1"
    if include_legacy and legacy_prompt:
        prompt = (
            f"{prompt}\n\n"
            "# Legacy Compatibility Prompt\n\n"
            "The following legacy prompt remains enabled during migration to keep "
            "behavior stable. Prefer the modular sections above when they are more specific.\n\n"
            f"{legacy_prompt.strip()}"
        )

    return prompt


def _resolve_runtime_scenarios(profile_name: str, original_request: str) -> List[str]:
    request = (original_request or "").lower()
    if profile_name != "humanoid_g1":
        return []

    reading_terms = [
        "看书",
        "读书",
        "阅读",
        "学习",
        "study",
        "read",
        "reading",
        "desk",
        "书桌",
    ]
    setup_terms = [
        "收拾",
        "整理",
        "清理",
        "准备",
        "setup",
        "prepare",
        "clean",
        "tidy",
        "organize",
    ]
    if any(term in request for term in reading_terms) and any(term in request for term in setup_terms):
        return ["scenarios/reading_environment_setup.md"]
    return []
