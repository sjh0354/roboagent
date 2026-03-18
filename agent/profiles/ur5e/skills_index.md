# Skills Index: UR5e

Available skills:
- `store_pick_and_place`
  - Use for moving requested objects between their current visible location and the intended destination in the store workspace.
- `speak_and_report`
  - Use for readiness, blockers, completion reports, and short direct replies to humans in chat.
- `send_agent_message`
  - Use for remote robot-to-robot messaging through OpenClaw + Lark/Feishu.
- `observe_workspace`
  - Use for visual reassessment of the workspace before the next step.

Allowed actions summary:
- `speak`
- `send_agent_message`
- `pick_and_place`
- `get_observation`

Skill loading rule:
- Load the full skill only when the task requires its detailed procedure or failure handling.
