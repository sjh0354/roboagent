# Scenario: Arm Reading Environment Setup

Trigger this scenario when the user directly asks the UR5e arm to prepare for reading or study, including requests such as:
- "我准备看书了，帮我收拾一下桌子"
- "我想看书，整理一下桌子"
- "prepare my reading environment"
- "help me study at the desk"

Treat the request as a single-device multi-domain scene setup task. The UR5e arm is the user-facing service device and must handle embodied cleanup, smart-home lighting, background audio, and the final spoken response.

Required transient checklist:
1. Clear supported clutter from the desk/tray using `pick_and_place`.
2. Dim the room/background light using `control_light` with `device=background_light`, `action=set_brightness`, and a reading-friendly low brightness.
3. Turn on the reading lamp using `control_light` with `device=desk_lamp` and `action=turn_on`.
4. Play quiet background audio using `play_audio` with `action=play`, `audio_type=white_noise`, `track=reading_white_noise`, and moderate volume.
5. After the above actions have been attempted, give one final spoken confirmation with `speak`: "桌子已清理，阅读灯和背景音已为您打开。"
6. After that final `speak` action succeeds, the next response must terminate the task with `next_step: null`.

Execution rules:
- Plan one step at a time, but keep this checklist active until all required items are attempted.
- Do not mark the task complete after only desk cleanup.
- Do not skip lighting or background audio unless the user explicitly rejects them.
- The final `speak` confirmation is a one-time terminal notification. Do not call `speak` repeatedly after it succeeds.
- Use `next_step: null` as the unique task-completion signal after the final confirmation has been spoken.
- Do not use remote agent messaging; handle the request directly as the single user-facing device.
- For desk cleanup on real hardware, move only supported clutter objects such as medicine/medicine_box and water to `basket`.
- Use post-action visual outcome feedback and the current observation to decide what actually changed. Treat action history as attempted commands, not guaranteed state changes.
- If the observed outcome differs from the requested `pick_and_place` parameters, trust the observed outcome when deciding the next step.
- Leave books, notebooks, papers, and reading material on the desk unless the user explicitly asks to move them.
