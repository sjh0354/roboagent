# Skill: Store Memory

Use when:
- The user explicitly asks you to remember, record, save, or store a durable fact or preference.
- The task is primarily a memory update rather than a physical action.

Action:
- `store_memory`

Parameters:
- `content`: the durable fact, preference, or constraint to save
- `scope`: `global` for cross-embodiment memory, or `hardware` for profile-specific memory
- `category`: short label such as `preference`, `constraint`, `safety`, or `note`

Rules:
- Only store information that is stable and likely to matter again.
- Prefer compact normalized wording over copying the full user utterance.
- Use `global` for user preferences that should apply across robots.
- After a successful memory write, give at most one short confirmation, then end the task.
