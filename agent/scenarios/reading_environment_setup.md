# Scenario: Reading Environment Setup

Trigger this scenario when the user expresses an intent to read, study, or prepare a desk for reading, including requests such as:
- "我准备看书了，帮我收拾一下桌子"
- "我想看书，整理一下桌子"
- "prepare my reading environment"
- "help me study at the desk"

Treat the request as a multi-domain scene setup task, not only a desk-cleaning task.

Required transient checklist:
1. Coordinate desk cleanup with the UR5e arm using `send_agent_message`.
2. Dim the room/background light using `control_light` with `device=background_light`, `action=set_brightness`, and a reading-friendly low brightness.
3. Turn on the reading lamp using `control_light` with `device=desk_lamp` and `action=turn_on`.
4. Play quiet background audio using `play_audio` with `action=play`, `audio_type=white_noise`, `track=reading_white_noise`, and moderate volume.
5. After the above actions have been attempted, give one final spoken confirmation with `speak`: "桌子已清理，阅读灯和背景音已为您打开。"

Execution rules:
- Plan one step at a time, but keep the checklist active until all required items are attempted.
- Do not mark the task complete after only desk cleanup.
- Do not skip IoT lighting or background audio unless the user explicitly rejects them.
- Use `send_agent_message` for UR5e coordination; do not navigate to the store yourself for this scenario unless the user explicitly asks you to go there.
- If UR5e reports that only supported clutter can be moved, continue with lighting and audio after the cleanup attempt.
