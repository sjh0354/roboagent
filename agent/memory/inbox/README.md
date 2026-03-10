# Memory Inbox

New interaction-derived memory candidates are written here first.

- `hardware/`: candidate memories scoped to a specific embodiment profile.
- `global/`: candidate memories scoped across all embodiments.

These inbox files are intentionally append-only. A later compaction step should
merge, deduplicate, and promote useful entries into the stable memory files.
