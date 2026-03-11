"""
Resolve which skills should be loaded for the active planning step.
"""

from typing import Dict, List

from utils.action_registry import ACTION_SCHEMAS, get_action_schema, get_action_to_skill

COMMON_SKILLS = {
    "speak_and_report": "skills/common/speak_and_report/SKILL.md",
    "send_agent_message": "skills/common/send_agent_message/SKILL.md",
    "observe_scene": "skills/common/observe_scene/SKILL.md",
    "search_web": "skills/common/search_web/SKILL.md",
}

PROFILE_SKILLS = {
    "humanoid_g1": {
        "control_home_devices": "skills/humanoid_g1/control_home_devices/SKILL.md",
        "navigate_rooms": "skills/humanoid_g1/navigate_rooms/SKILL.md",
        "manipulate_objects": "skills/humanoid_g1/manipulate_objects/SKILL.md",
    },
    "ur5e": {
        "store_pick_and_place": "skills/ur5e/store_pick_and_place/SKILL.md",
        "observe_workspace": "skills/ur5e/observe_workspace/SKILL.md",
    },
}


class SkillResolver:
    """Heuristic resolver for choosing skill documents at runtime."""

    def __init__(self, profile_name: str):
        self.profile_name = profile_name

    def resolve(
        self,
        original_request: str = "",
        execution_history: List[Dict] | None = None,
        current_location: str | None = None,
    ) -> List[str]:
        history = execution_history or []
        skill_names = {"observe_scene" if self.profile_name == "humanoid_g1" else "observe_workspace"}
        request = (original_request or "").lower()
        action_to_skill = get_action_to_skill(self.profile_name)

        skill_names.update(self._resolve_from_request(request))
        skill_names.update(self._resolve_from_history(history))

        latest_action = history[-1].get("action") if history else None
        if latest_action and latest_action in action_to_skill:
            skill_names.add(action_to_skill[latest_action])
        if self.profile_name == "humanoid_g1" and current_location == "Room 02":
            skill_names.add("navigate_rooms")

        return self._to_paths(skill_names)

    def _resolve_from_request(self, request: str) -> set[str]:
        skill_names: set[str] = set()
        for action_name, schema in ACTION_SCHEMAS.get(self.profile_name, {}).items():
            keywords = schema.get("keywords", [])
            if any(term in request for term in keywords):
                skill_name = schema.get("skill")
                if skill_name:
                    skill_names.add(str(skill_name))
                for follow_up_skill in schema.get("follow_up_skills", []):
                    skill_names.add(str(follow_up_skill))

        if self.profile_name == "humanoid_g1" and any(
            term in request for term in ["water", "snack", "snacks", "fruit", "medicine", "bring", "fetch", "get me"]
        ):
            skill_names.add("navigate_rooms")
        return skill_names

    def _resolve_from_history(self, history: List[Dict]) -> set[str]:
        skill_names: set[str] = set()
        for step in history[-2:]:
            action_name = step.get("action")
            if not action_name:
                continue
            schema = get_action_schema(self.profile_name, action_name)
            skill_name = schema.get("skill")
            if skill_name:
                skill_names.add(str(skill_name))
            for follow_up_skill in schema.get("follow_up_skills", []):
                skill_names.add(str(follow_up_skill))
        return skill_names

    def _to_paths(self, skill_names: set[str]) -> List[str]:
        all_skills = {}
        all_skills.update(COMMON_SKILLS)
        all_skills.update(PROFILE_SKILLS.get(self.profile_name, {}))
        return [all_skills[name] for name in sorted(skill_names) if name in all_skills]
