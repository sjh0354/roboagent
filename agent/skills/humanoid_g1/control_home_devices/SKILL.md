# Skill: Control Home Devices

Supported actions:
- `control_air_conditioner`
- `control_light`

Procedure:
1. Confirm the request concerns a controllable home device.
2. Choose the minimal action that satisfies the request.
3. Include temperature only for air conditioner control.
4. Report completion only when useful to the user.

Constraints:
- Air conditioner temperature should remain in the supported range.
- For `control_light`, use `device=background_light` when dimming room/background lighting and `device=desk_lamp` when turning on a reading lamp.
- Use `action=set_brightness` with integer `brightness` from 0 to 100 when the request needs dimming.
- Do not mix device control with navigation in the same step.
