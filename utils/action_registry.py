"""
Shared action schemas for embodied agent profiles.
"""

from typing import Dict, List


ACTION_SCHEMAS: Dict[str, Dict[str, Dict[str, object]]] = {
    "humanoid_g1": {
        "speak": {
            "action_type": "talk",
            "skill": "speak_and_report",
            "keywords": ["say", "speak", "tell", "report", "notify", "ask"],
            "follow_up_skills": [],
        },
        "send_agent_message": {
            "action_type": "talk",
            "skill": "send_agent_message",
            "keywords": ["message robot", "contact store robot", "send message", "notify robot", "group chat", "lark", "feishu"],
            "follow_up_skills": [],
        },
        "control_air_conditioner": {
            "action_type": "tool",
            "skill": "control_home_devices",
            "keywords": ["air conditioner", "ac", "temperature", "cold", "hot"],
            "follow_up_skills": ["speak_and_report"],
        },
        "control_light": {
            "action_type": "tool",
            "skill": "control_home_devices",
            "keywords": ["light", "lights", "lamp", "dark", "bright"],
            "follow_up_skills": ["speak_and_report"],
        },
        "web_search": {
            "action_type": "tool",
            "skill": "search_web",
            "keywords": ["search", "look up", "find online", "web", "internet"],
            "follow_up_skills": ["speak_and_report"],
        },
        "query_weather_api": {
            "action_type": "tool",
            "skill": "query_weather",
            "keywords": ["weather", "temperature", "forecast", "cold", "hot"],
            "follow_up_skills": ["speak_and_report"],
        },
        "navigate_to": {
            "action_type": "act",
            "skill": "navigate_rooms",
            "keywords": ["go", "come", "bring", "fetch", "store", "room", "water", "snack", "fruit", "medicine"],
            "follow_up_skills": ["speak_and_report", "manipulate_objects"],
        },
        "pick": {
            "action_type": "act",
            "skill": "manipulate_objects",
            "keywords": ["pick", "grab", "grasp", "take", "hold"],
            "follow_up_skills": ["manipulate_objects", "navigate_rooms"],
        },
        "place": {
            "action_type": "act",
            "skill": "manipulate_objects",
            "keywords": ["place", "put", "set down", "leave"],
            "follow_up_skills": ["speak_and_report"],
        },
        "wait_for": {
            "action_type": "act",
            "skill": "observe_scene",
            "keywords": ["wait", "pause", "hold on"],
            "follow_up_skills": ["observe_scene"],
        },
        "get_observation": {
            "action_type": "sense",
            "skill": "observe_scene",
            "keywords": ["observe", "look", "check", "inspect"],
            "follow_up_skills": [],
        },
    },
    "ur5e": {
        "speak": {
            "action_type": "talk",
            "skill": "speak_and_report",
            "keywords": ["say", "report", "notify", "tell"],
            "follow_up_skills": [],
        },
        "send_agent_message": {
            "action_type": "talk",
            "skill": "send_agent_message",
            "keywords": ["message humanoid", "notify humanoid", "send message", "group chat", "lark", "feishu"],
            "follow_up_skills": [],
        },
        "pick_and_place": {
            "action_type": "act",
            "skill": "store_pick_and_place",
            "keywords": ["pick", "place", "move", "counter", "shelf", "water", "snack", "fruit", "medicine"],
            "follow_up_skills": ["speak_and_report", "observe_workspace"],
        },
        "get_observation": {
            "action_type": "sense",
            "skill": "observe_workspace",
            "keywords": ["observe", "look", "check", "inspect"],
            "follow_up_skills": ["store_pick_and_place"],
        },
    },
}


def get_allowed_actions(profile_name: str) -> List[str]:
    """Return the allowed action names for a profile."""
    return sorted(ACTION_SCHEMAS.get(profile_name, {}).keys())


def get_action_schema(profile_name: str, action_name: str) -> Dict[str, object]:
    """Return schema metadata for a specific action."""
    return ACTION_SCHEMAS.get(profile_name, {}).get(action_name, {})


def get_action_to_skill(profile_name: str) -> Dict[str, str]:
    """Return a mapping from action name to skill name for a profile."""
    return {
        action_name: str(schema["skill"])
        for action_name, schema in ACTION_SCHEMAS.get(profile_name, {}).items()
        if "skill" in schema
    }
