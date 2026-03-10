# Agent Layout

This directory holds the modular agent definition that will gradually replace
the large in-code prompt templates.

## Structure

- `bootstrap.md`: top-level operating contract for all embodied agents.
- `soul.md`: interaction style and behavior principles shared across agents.
- `memory/`: layered long-term memories.
- `profiles/`: hardware-specific identity and skill indexes.
- `skills/`: on-demand operational instructions grouped by hardware.

## Memory Levels

- Session memory: handled by the active conversation context.
- Hardware memory: persistent notes on how to operate a specific embodiment.
- Global memory: persistent notes on how to be more useful to the user.

## Migration Strategy

Phase 1 keeps the existing Python planner APIs stable while loading these files
through a prompt assembler. Legacy prompt text can remain enabled during the
transition for compatibility.
