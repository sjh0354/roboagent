# Skill: Store Pick And Place

Supported action:
- `pick_and_place`

Procedure:
1. Identify the requested item in the current workspace.
2. Move it from the source location to the target location in one action.
3. Re-observe if item visibility or workspace state is ambiguous.
4. Report readiness or blockage to the humanoid when needed.

Constraints:
- Stay within the store workspace.
- Prefer `shelf -> counter` for normal fulfillment.
- Do not invent lower-level manipulation actions.
