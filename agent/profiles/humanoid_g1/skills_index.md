# Skills Index: Unitree-G1

Available skills:
- `speak_and_report`
  - Use for user-facing updates, clarification, and task completion messages.
- `control_home_devices`
  - Use for air conditioner and lighting tasks in Room 01.
- `navigate_rooms`
  - Use when changing between Room 01 and Room 02.
- `manipulate_objects`
  - Use for generic `pick` and `place` actions.
- `search_web`
  - Use when the task requires external information lookup.
- `observe_scene`
  - Use when a fresh visual observation is needed before the next action.

Allowed actions summary:
- `speak`
- `control_air_conditioner`
- `control_light`
- `web_search`
- `navigate_to`
- `pick`
- `place`
- `wait_for`
- `get_observation`

Skill loading rule:
- Load the full skill only when the task requires its procedure or constraints.
