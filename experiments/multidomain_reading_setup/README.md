# Multidomain Reading Setup Benchmark

This experiment evaluates cross-domain orchestration for a reading setup task:
voice intent, UR5e desk cleanup, smart lighting, external audio API, and final TTS.

Run deterministic mock evaluation:

```bash
python -m experiments.multidomain_reading_setup.run_eval \
  --execution-mode mock \
  --trace-mode scripted \
  --memory-modes full,none,fixed
```

Run planner-in-loop mock evaluation:

```bash
python -m experiments.multidomain_reading_setup.run_eval \
  --execution-mode mock \
  --trace-mode planner \
  --memory-modes full,none,fixed
```

Validate scorer failure discrimination:

```bash
python -m experiments.multidomain_reading_setup.run_eval \
  --execution-mode mock \
  --trace-mode planner \
  --memory-modes full \
  --failure-suite
```

Record golden traces:

```bash
python -m experiments.multidomain_reading_setup.run_eval \
  --execution-mode mock \
  --trace-mode planner \
  --memory-modes full,none,fixed \
  --record-golden
```

Replay golden traces offline:

```bash
python -m experiments.multidomain_reading_setup.run_eval \
  --execution-mode mock \
  --trace-mode replay \
  --memory-modes full,none,fixed
```

Outputs:
- `outputs/results.jsonl`: per-case traces and scores
- `outputs/summary.csv`: task success rate and submetric averages
- `golden_traces/<memory_mode>/<case_id>.json`: recorded traces for replay

Memory modes:
- `full`: stable memory plus hierarchical runtime memory
- `none`: no memory injection or runtime writeback
- `fixed`: naive fixed-frequency summaries

`scripted` emits fixed traces for scorer validation. `planner` exercises
`BaseVLMPlanner` plus the mock G1/UR5e executors with deterministic fake VLM
responses. The hardware mode is intentionally not enabled by default. It should
be wired to the live voice/Lark/UR5e runtime for demos after the mock benchmark
is stable.

Failure variants:
- `missing_audio`: removes the white-noise API call.
- `no_desk_lamp`: removes desk lamp activation.
- `wrong_route_type`: routes audio through the wrong action type.
- `moves_notebook`: moves protected reading material.
- `no_tts`: removes the final spoken completion report.
