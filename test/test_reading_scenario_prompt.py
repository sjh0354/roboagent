from template.modular_prompt_loader import build_runtime_system_prompt
from utils.skill_resolver import SkillResolver


def test_reading_environment_request_injects_scenario_prompt():
    prompt = build_runtime_system_prompt(
        profile_name="humanoid_g1",
        original_request="我准备看书了，帮我收拾一下桌子",
        execution_history=[],
        current_location="Room 01",
        legacy_prompt="",
        include_memory=False,
    )

    assert "Scenario: Reading Environment Setup" in prompt
    assert "Required transient checklist" in prompt
    assert "control_light" in prompt
    assert "play_audio" in prompt
    assert "send_agent_message" in prompt
    assert "桌子已清理，阅读灯和背景音已为您打开。" in prompt


def test_reading_environment_request_loads_required_skills():
    paths = SkillResolver("humanoid_g1").resolve(
        original_request="我想看书，整理一下桌子",
        execution_history=[],
        current_location="Room 01",
    )

    assert "skills/common/play_audio/SKILL.md" in paths
    assert "skills/common/send_agent_message/SKILL.md" in paths
    assert "skills/common/speak_and_report/SKILL.md" in paths
    assert "skills/humanoid_g1/control_home_devices/SKILL.md" in paths


def test_arm_reading_environment_request_injects_single_device_scenario_prompt():
    prompt = build_runtime_system_prompt(
        profile_name="ur5e",
        original_request="我准备看书了，帮我收拾一下桌子",
        execution_history=[],
        current_location="store",
        legacy_prompt="",
        include_memory=False,
    )

    assert "Scenario: Arm Reading Environment Setup" in prompt
    assert "single-device multi-domain scene setup task" in prompt
    assert "pick_and_place" in prompt
    assert "control_light" in prompt
    assert "play_audio" in prompt
    assert "Do not use remote agent messaging" in prompt
    assert "Do not call `speak` repeatedly" in prompt
    assert "next_step: null" in prompt
    assert "桌子已清理，阅读灯和背景音已为您打开。" in prompt


def test_arm_reading_environment_request_loads_required_skills():
    paths = SkillResolver("ur5e").resolve(
        original_request="我想看书，整理一下桌子",
        execution_history=[],
        current_location="store",
    )

    assert "skills/ur5e/store_pick_and_place/SKILL.md" in paths
    assert "skills/common/play_audio/SKILL.md" in paths
    assert "skills/common/speak_and_report/SKILL.md" in paths
    assert "skills/humanoid_g1/control_home_devices/SKILL.md" in paths
    assert "skills/common/send_agent_message/SKILL.md" not in paths
