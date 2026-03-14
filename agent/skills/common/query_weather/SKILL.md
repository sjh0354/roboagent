# Skill: Query Weather API

Use when:
- The user asks about weather/temperature/forecast in simulation experiments.
- You need structured weather data from the weather backend.

Action:
- `query_weather_api`

Current known payload format (legacy knowledge):
```json
{
  "city": "Beijing"
}
```

Rules:
- Use the known payload schema from this skill document.
- If backend reports missing required parameters, treat it as an API contract drift signal.
- In reflection-enabled mode, revise this skill doc and retry with corrected payload.
