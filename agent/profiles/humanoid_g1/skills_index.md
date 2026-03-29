# Skills Index: Unitree-G1

Available skills:
- `speak_and_report`
  - Use for user-facing updates, clarification, and task completion messages.
- `send_agent_message`
  - Use for remote robot-to-robot messaging through OpenClaw + Lark/Feishu.
- `store_memory`
  - Use when a user explicitly asks this agent to remember a durable preference, constraint, or fact.
- `control_home_devices`
  - Use for air conditioner and lighting tasks in Room 01.
- `navigate_rooms`
  - Use when changing between Room 01 and Room 02.
- `manipulate_objects`
  - Use for generic `pick` and `place` actions.
- `search_web`
  - Use when the task requires external information lookup.
- `choose_web_option`
  - Use for no-visual benchmark tasks that require choosing one explicit online/service option from a provided candidate list.
- `set_home_mode`
  - Use for no-visual benchmark tasks that require choosing one explicit home-device mode from a provided candidate list.
- `query_weather`
  - Use weather backend query in API-update reflection experiments.
- `observe_scene`
  - Use when a fresh visual observation is needed before the next action.

Allowed actions summary:
- `speak`
- `send_agent_message`
- `store_memory`
- `control_air_conditioner`
- `control_light`
- `web_search`
- `choose_web_option`
- `set_home_mode`
- `navigate_to`
- `pick`
- `place`
- `wait_for`
- `get_observation`

Skill loading rule:
- Load the full skill only when the task requires its procedure or constraints.
