# Skill: Navigate Rooms

Supported action:
- `navigate_to`

Procedure:
1. Determine whether a room change is actually required.
2. Use tracked room state rather than inferring location only from appearance.
3. Set `target_location` explicitly to `Room 01` or `Room 02`.
4. Set `with_item` to the carried item name or `none`.

Constraints:
- Use navigation only for cross-room movement.
- Do not invent legacy navigation actions.
