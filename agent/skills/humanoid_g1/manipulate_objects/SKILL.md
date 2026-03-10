# Skill: Manipulate Objects

Supported actions:
- `pick`
- `place`

Procedure:
1. Describe the target object or receptacle in plain language.
2. Keep each manipulation step atomic.
3. Re-observe if object identity or placement target is ambiguous.

Constraints:
- Do not combine pick and place into one step for the humanoid planner.
- Use manipulation only when the task cannot be handled by navigation, speech, or device control.
