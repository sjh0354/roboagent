# Skill: Play Audio

Use when:
- The task asks for background sound, white noise, rain noise, or ambient audio.
- Audio playback is part of a broader room setup.

Supported action:
- `play_audio`

Parameters:
- `action`: `play` or `stop`
- `audio_type`: `white_noise`, `rain_noise`, `brown_noise`, or `ambient`
- `track`: short track label, such as `reading_white_noise`
- `volume`: integer from 0 to 100

Rules:
- For reading or quiet focus requests, prefer `audio_type=white_noise` unless the user requested a specific sound.
- Keep volume moderate for background use.
- Do not use `web_search` for local audio playback.
