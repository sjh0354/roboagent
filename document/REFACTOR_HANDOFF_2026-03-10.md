# Refactor Handoff - 2026-03-10

## Branch And Commit

- Working branch: `refactor/prep`
- Latest commit: `40050e0 refactor: modularize agent runtime and prompts`

## Refactor Goal

This refactor moved the repository away from large monolithic prompt templates
and duplicated planner/executor logic toward a modular embodied-agent runtime.

The main goals achieved were:

- Split prompt knowledge into modular files.
- Introduce hardware profiles, identity, soul, and skills.
- Add layered persistent memory with inbox-style candidate writing.
- Centralize action schema and skill resolution.
- Share planner runtime between humanoid and arm planners.
- Share vision augmentation logic between humanoid and arm executors.
- Reduce CLI log noise and expose detailed logs behind `--log`.

## High-Level Architecture After Refactor

### 1. Agent Definition

New `agent/` directory:

- `agent/bootstrap.md`
- `agent/soul.md`
- `agent/profiles/humanoid_g1/identity.md`
- `agent/profiles/humanoid_g1/skills_index.md`
- `agent/profiles/ur5e/identity.md`
- `agent/profiles/ur5e/skills_index.md`
- `agent/skills/.../SKILL.md`
- `agent/memory/global_memory.md`
- `agent/memory/hardware/*.md`
- `agent/memory/inbox/`

Purpose:

- `identity.md`: defines embodiment, capabilities, and constraints.
- `soul.md`: defines interaction style and behavior principles.
- `skills_index.md`: short routing layer for skills.
- `SKILL.md`: detailed operational procedures.
- `memory/`: long-term memory layers.

### 2. Prompt Assembly

New files:

- `template/modular_prompt_loader.py`
- `utils/skill_resolver.py`
- `utils/action_registry.py`

Behavior:

- Base prompt is assembled from bootstrap, soul, identity, skills index, and memory.
- Runtime prompt loads only relevant skills.
- Legacy prompt compatibility remains available during migration.
- Action schema is now a shared source of truth.

### 3. Planner Runtime

New file:

- `planner/base_vlm_planner.py`

Refactored planners:

- `planner/humanoid_planner_vlm.py`
- `planner/arm_planner_vlm.py`

Shared in base runtime:

- task start/reset
- half-open-loop execution loop
- prompt assembly and VLM request flow
- waiting-for-input / resume flow
- task summary generation
- memory candidate writing at interaction boundaries

Planner-specific logic left in subclasses:

- observation acquisition differences
- hardware-specific execution behavior
- context message formatting
- response field naming (`human_question` vs `humanoid_question`)
- interrupt handling for humanoid

### 4. Executor Runtime

New file:

- `executor/vision_enabled_mixin.py`

Refactored executors:

- `executor/humanoid_executor_vision.py`
- `executor/arm_executor_vision.py`

Shared in mixin:

- VLM client initialization
- simulation image manager initialization
- current observation retrieval
- `_get_observation()` output construction
- attaching visual observation to action results
- reset/statistics/close helpers

Executor-specific logic left in subclasses:

- camera type selection
- arm PI0 execution path
- humanoid web search override
- arm simulation state initialization

## Memory Design Implemented

### Memory Levels

- Session memory:
  - handled by normal conversation context
- Hardware memory:
  - stable notes per embodiment
- Global memory:
  - stable notes across embodiments

### Inbox Flow

New file:

- `utils/memory_manager.py`

Current behavior:

- On task completion / waiting-for-input / interruption / planner error,
  memory candidates are written into:
  - `agent/memory/inbox/hardware/`
  - `agent/memory/inbox/global/`
- Inbox files are ignored by git except for scaffolding files.

Current limitation:

- There is not yet a compaction/promote step.
- Memory candidates are written, but no automated merge into stable memory files exists yet.

## Action Schema And Skill Resolution

Action registry:

- `utils/action_registry.py`

Skill resolution:

- `utils/skill_resolver.py`

Current behavior:

- Allowed actions for each profile come from shared action schema.
- Template validators use the same shared action definitions.
- Runtime skill selection uses:
  - original request text
  - action schema metadata
  - recent execution history
  - follow-up skill hints

This replaced multiple duplicated hard-coded action lists.

## Logging And CLI UX

Main change:

- Detailed logs are now intended to be shown only with `--log`.
- Default CLI output is reduced to key information.

CLI mode control:

- Default mode is now `real`.
- Simulation mode is enabled only with `--simulation`.

Supported commands:

- `python planner/humanoid_planner_vlm.py --log`
- `python planner/arm_planner_vlm.py --log`
- `python planner/humanoid_planner_vlm.py --simulation`
- `python planner/arm_planner_vlm.py --simulation`

Default mode:

- concise status output
- concise completion/error output

`--log` mode:

- planner/executor/ASR detailed logs

`--simulation` mode:

- use local simulation images instead of real hardware/camera flow
- useful for prompt/planner testing and safe end-to-end simulation runs

## Important Files Added

- `agent/README.md`
- `agent/bootstrap.md`
- `agent/soul.md`
- `agent/memory/...`
- `agent/profiles/...`
- `agent/skills/...`
- `template/modular_prompt_loader.py`
- `utils/action_registry.py`
- `utils/skill_resolver.py`
- `utils/memory_manager.py`
- `planner/base_vlm_planner.py`
- `executor/vision_enabled_mixin.py`

## Important Files Modified

- `planner/humanoid_planner_vlm.py`
- `planner/arm_planner_vlm.py`
- `executor/humanoid_executor_vision.py`
- `executor/arm_executor_vision.py`
- `template/humanoid_prompt_template_vlm.py`
- `template/arm_prompt_template_vlm.py`

## Tests Already Performed

### Structure / Syntax

Passed:

- `python -m py_compile` on key planners, executors, template loader, memory manager, action registry, and skill resolver.

### Runtime Import And Instantiation

Confirmed:

- `from google import genai` imports successfully in the current environment.
- Simulation-mode executor instantiation works after camera imports were made lazy.
- Simulation-mode planner instantiation works.
- Planner summaries return expected profile-specific keys.

### CLI

Confirmed:

- `python planner/humanoid_planner_vlm.py --help`
- `python planner/arm_planner_vlm.py --help`
- `--log` option is exposed.
- `--simulation` option is exposed.

### Behavioral Spot Checks

Confirmed:

- runtime skill resolution selects relevant skills for representative humanoid and arm requests
- action schema exports allowed actions consistently
- legacy compatibility prompt is not duplicated in runtime prompt assembly

## Known Limitations / Remaining Work

### 1. API / Network Integration Still Needs Validation

Observed during testing:

- live Gemini image analysis failed with a network/DNS-style error in the execution environment

Implication:

- local structure is working
- real end-to-end online VLM execution still needs to be validated after environment/API/network issues are resolved

### 2. Memory Pipeline Is Incomplete

Still missing:

- candidate deduplication
- compaction
- promotion from inbox to stable memory
- pruning strategy

### 3. Logging Is Improved But Not Fully Unified

Current state:

- main planner CLI now supports `--log`
- many lower-level utility classes still use `verbose` directly

Potential follow-up:

- introduce a shared logger / event sink
- separate operator-facing UX messages from internal debug logs

### 4. Test Coverage Is Still Thin

Refactor is large, but formal regression coverage is still minimal.

## Recommended Next Tests

### Priority 1: Local Regression Tests

1. Simulation planner smoke tests
   - instantiate both planners in `simulation_mode=True`
   - verify `start_new_task(..., run_autonomously=False)` returns a first-step plan or structured error

2. Prompt assembly tests
   - verify active skills differ for:
     - humanoid device-control requests
     - humanoid fetch-item requests
     - arm pick-and-place requests
   - verify legacy prompt appears at most once

3. Action schema tests
   - verify allowed actions used by validators match the registry
   - verify invalid actions are rejected

4. Memory candidate tests
   - verify completion writes hardware/global inbox files
   - verify waiting-for-input writes candidates
   - verify resume after clarification can write a second candidate

### Priority 2: Executor Tests

1. Simulation observation tests
   - humanoid vision executor returns state + image path
   - arm vision executor returns state + image path

2. Simulation action tests
   - humanoid tool action updates observation payload
   - arm `pick_and_place` updates observation payload

### Priority 3: End-To-End API Tests

After API/network is fixed:

1. run humanoid planner in simulation mode with real VLM call
2. run arm planner in simulation mode with real VLM call
3. verify planner output JSON passes validation
4. verify a multi-step task can complete end-to-end

### Priority 4: CLI UX Tests

1. compare default output vs `--log`
2. verify default mode remains concise during:
   - normal planning
   - clarification
   - interruption
   - completion

## Suggested Next Session Starting Points

If continuing in a new session, the best entry points are:

1. Fix live API/network interaction and rerun end-to-end simulation tests.
2. Add formal tests for prompt assembly, memory inbox writes, and executor simulation behavior.
3. Consider introducing a unified logging/event layer instead of raw `print()` plus `verbose`.

## Quick Commands For Next Session

Useful commands:

```bash
git branch --show-current
git log --oneline -1
python -m py_compile planner/base_vlm_planner.py planner/humanoid_planner_vlm.py planner/arm_planner_vlm.py executor/vision_enabled_mixin.py executor/humanoid_executor_vision.py executor/arm_executor_vision.py
python planner/humanoid_planner_vlm.py --help
python planner/arm_planner_vlm.py --help
python planner/humanoid_planner_vlm.py --simulation
python planner/arm_planner_vlm.py --simulation
```

## Additional Live API Validation Performed

After the original handoff draft, the environment was updated with a working API
key and a usable `google.genai` runtime. Additional online validation was then
performed.

### Live Image Analysis Tests

Passed with escalated network access:

1. Humanoid simulation image
   - image: `simulation_images/home/default.jpg`
   - result: non-empty `vlm_description`
   - conclusion: Gemini image analysis path works for humanoid executor

2. Arm simulation image
   - image: `simulation_images/store/default.jpg`
   - result: non-empty `vlm_description`
   - conclusion: Gemini image analysis path works for arm executor

Important note:

- In the default sandbox, image analysis still failed with a DNS/network error.
- The same requests succeeded with escalated network access.
- This indicates the remaining issue is sandbox/network policy, not API key validity or code structure.

### Live Planner Single-Step Test

Passed with escalated network access:

- Planner: `AutonomousVLMPlanner`
- Mode: `simulation_mode=True`
- Request: `I'm feeling cold.`
- Result: planner returned valid structured JSON with:
  - `action = control_air_conditioner`
  - `action_type = tool`
  - `parameters = {"action": "turn_on", "temperature": 24}`

Conclusion:

- The runtime prompt assembly, image input, and structured action generation path all work online.

### Live End-To-End Simulation Task Tests

#### Arm Planner Test

Request:

- `Get water for the humanoid robot and place it on the counter.`

Observed result:

- The task did not auto-complete.
- The planner returned:
  - `needs_human_input = true`
  - `next_step = null`
  - a clarification message indicating water could not be located

Interpretation:

- This is a valid behavior for the current simulation scene.
- The loop, clarification path, and memory candidate writing all worked.
- The task did not finish because the current store image did not provide visible water.

#### Humanoid Planner Completion Test

Request:

- `I am feeling cold. Please turn on the air conditioner to 24 degrees and let me know when it is done.`

Observed result:

- Task completed successfully.
- Executed actions:
  - `control_air_conditioner`
  - `speak`
- Final planner result contained:
  - `next_step = null`
  - `needs_human_input = false`
  - `task_summary.success = true`

Interpretation:

- Full online end-to-end simulation execution is working at least for the humanoid AC-control flow.
- This validated:
  - online planning
  - multi-step loop execution
  - final completion handling
  - memory candidate writing on completion

### Updated Testing Status Summary

Now confirmed working:

- online Gemini image analysis for simulation images
- online planner single-step action generation from image context
- online humanoid end-to-end simulation task completion
- online clarification path for arm task when the requested item is not visually available

Still recommended next:

- run an online arm task using a simulation image that clearly contains the requested item
- verify the arm planner can complete a full successful retrieval task, not only a clarification branch
- add repeatable automated tests for these successful online/simulation scenarios if practical
