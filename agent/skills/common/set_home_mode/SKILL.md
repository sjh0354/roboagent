# Skill: Set Home Mode

Use when:
- The task is a no-visual home-assistant benchmark with explicit candidate modes.
- The user needs one device mode selected from a provided A/B/C list.

Supported action:
- `set_home_mode`

Parameters:
- `option_id`: candidate ID copied exactly from the provided list, such as `A`, `B`, or `C`
- `option_text`: exact candidate text copied from the provided list
- `device`: short device label such as `smart_hvac`, `smart_lighting`, `robot_vacuum`, or `media_system`

Rules:
- Choose exactly one provided candidate. Do not invent a new mode.
- Use dialogue context for temporary needs such as sleeping, reading, taking a call, or reducing odor.
- Use long-term memory for durable constraints such as noise sensitivity, glare sensitivity, or dry-eye discomfort.
- Prefer the device mode that best balances the immediate task with the durable user preference.
- Copy `option_text` exactly so the evaluation runner can score the decision reliably.

Examples:
- Night rest with noise sensitivity:
  - `action`: `set_home_mode`
  - `parameters`: `{"option_id": "C", "option_text": "sleep mode", "device": "smart_hvac"}`
- Late-night media with another person sleeping nearby:
  - `action`: `set_home_mode`
  - `parameters`: `{"option_id": "A", "option_text": "private headphone mode", "device": "media_system"}`
