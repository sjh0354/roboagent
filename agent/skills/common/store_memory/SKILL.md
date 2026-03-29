# Skill: Store Memory

Use when:
- The user explicitly asks you to remember, record, save, or store a durable fact or preference.
- The task is primarily a memory update rather than a physical action.
- Treat phrases like "请记住", "以后默认按这个来", "这是长期偏好", and "除非我明确要求，否则都按这个来" as explicit triggers.

Action:
- `store_memory`

Parameters:
- `content`: the durable fact, preference, or constraint to save
- `scope`: `global` for cross-embodiment memory, or `hardware` for profile-specific memory
- `category`: short label such as `preference`, `constraint`, `safety`, or `note`

Memory structure:
- `user_preferences.md`
  - High-priority user preferences and standing constraints.
  - Typical examples: dietary restrictions, personal likes/dislikes, accessibility needs, recurring delivery preferences.
  - Routing rule: if `category=preference`, the runtime stores it here with highest priority.
- `global_memory.md`
  - Durable cross-embodiment knowledge that should help all agents.
  - Typical examples: stable user facts, shared task conventions, reusable long-term constraints.
  - Use `scope=global` when the memory should matter to both humanoid and arm.
- `hardware/<profile>_memory.md`
  - Embodiment-specific operational memory.
  - Typical examples: UR5e workspace habits, G1 motion/device-handling lessons, hardware-specific caveats.
  - Use `scope=hardware` when the memory is mainly useful for the current robot profile.
- `memory/inbox/...`
  - Auto-generated candidate memories from interactions.
  - These are passive summaries, not the preferred destination for an explicit memory-write request.

Routing rules:
- User preference or standing user constraint:
  - `category=preference`
  - usually `scope=global`
- Cross-robot durable fact:
  - `category=note` or `constraint`
  - `scope=global`
- Robot-specific operational lesson:
  - `category=note` or `safety`
  - `scope=hardware`

Rules:
- Only store information that is stable and likely to matter again.
- Prefer compact normalized wording over copying the full user utterance.
- Use `global` for user preferences that should apply across robots.
- If the request is mainly to define a durable preference, the first action should usually be `store_memory`, not `speak`.
- After a successful memory write, give at most one short confirmation, then end the task.

Examples:
- User says: "I strictly avoid sugary drinks. Please remember this."
  - `content`: `strictly sugar-free; do not offer sugary drinks`
  - `scope`: `global`
  - `category`: `preference`
- User says: "Remember that the UR5e basket is the default handoff point."
  - `content`: `default UR5e handoff point is the basket`
  - `scope`: `hardware`
  - `category`: `note`
