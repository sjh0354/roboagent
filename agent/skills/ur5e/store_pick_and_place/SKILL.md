# Skill: Store Pick And Place

Supported action:
- `pick_and_place`

Purpose:
- Move a requested object from its current visible location to the intended destination in the store workspace.

Procedure:
1. Use the task request and current image to identify the intended object.
2. Set `item_name` to the requested or visually identified object name.
3. Set `source` to the object's current location in the scene.
4. Set `target` to the requested destination or the most sensible handoff location.
5. Execute the move in one `pick_and_place` action.
6. Re-observe if item identity, source, or destination is ambiguous.
7. Report readiness or blockage to the humanoid when needed.

Constraints:
- Stay within the store workspace.
- Do not assume a fixed product catalog.
- Do not assume the only valid route is `shelf -> counter`.
- Prefer the locations and object names actually supported by the current task and visual evidence.
- Do not invent lower-level manipulation actions.

Notes:
- `item_name`, `source`, and `target` are open-ended task parameters, not fixed enums.
- If several visually similar objects match the request, ask for clarification instead of guessing.
