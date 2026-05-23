# Skills Index: UR5e

Available skills:
- `store_pick_and_place`
  - Use for moving requested objects between their current visible location and the intended destination in the store workspace.
- `speak_and_report`
  - Use for readiness, blockers, completion reports, and short direct replies to the user.
- `store_memory`
  - Use when a user explicitly asks this agent to remember a durable preference, constraint, or fact.
- `control_home_devices`
  - Use for smart-home lighting tasks such as dimming background light or turning on a desk lamp.
- `play_audio`
  - Use for background sound, white noise, rain noise, or ambient audio playback.
- `observe_workspace`
  - Use for visual reassessment of the workspace before the next step.

Allowed actions summary:
- `speak`
- `store_memory`
- `control_light`
- `play_audio`
- `pick_and_place`
- `get_observation`

Skill loading rule:
- Load the full skill only when the task requires its detailed procedure or failure handling.
