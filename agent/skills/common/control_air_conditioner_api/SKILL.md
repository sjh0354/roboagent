# Skill: Control Air Conditioner API

Use when:
- The user asks to adjust the home air conditioner in simulation experiments.
- You need structured device control through the AC backend.

Action:
- `control_air_conditioner_api`

Current known API knowledge:
```json
{
  "parameter_keys": {
    "action": "action",
    "temperature": "temperature",
    "wind": "wind",
    "mode": "mode",
    "swing": "swing",
    "preset": "preset",
    "humidity": "humidity",
    "purifier": "purifier",
    "display": "display",
    "sleep_timer": "sleep_timer",
    "energy_saver": "energy_saver",
    "airflow_pattern": "airflow_pattern",
    "ambience_light": "ambience_light",
    "dehumidify_level": "dehumidify_level"
  },
  "mode_value_style": "canonical",
  "preset_value_style": "canonical",
  "display_value_style": "canonical",
  "energy_saver_value_style": "canonical",
  "airflow_pattern_value_style": "canonical",
  "ambience_light_value_style": "canonical",
  "dehumidify_level_value_style": "canonical"
}
```

Rules:
- Build request payloads from the API knowledge in this skill file.
- If backend reports missing required parameters or invalid values, treat it as an API contract drift signal.
- In reflection-enabled mode, revise this skill doc and retry with corrected knowledge.
