# RoboAgent local simulated benchmark runbook

All scores produced by this harness are **SIMULATED PHYSICAL EXECUTION; NOT REAL ROBOT PERFORMANCE**.

## Frozen protocol

- Model: `kimi-k3`, using the OpenAI-compatible credentials parsed from `/Users/michaelshi/Localwork/test.sh` without executing that file.
- Tasks: 40 synthetic local tasks: 20 manipulation and 20 navigation.
- Repetitions: three per task/method, compatibility ids `0,1,2`; these ids never enter the physical RNG.
- Physical RNG: unseeded `secrets.SystemRandom`.
- Underlying single-attempt assumption: 0.50.
- Monitor-present methods: one aggregate Bernoulli draw with `p=0.85`; the allowance of at most two monitor recoveries and time cost is already integrated.
- Monitor-absent methods: one aggregate Bernoulli draw with `p=0.60`; model self-detection/interruption is already integrated.
- No physical phase, timeout, stop, or explicit physical retry is simulated. Reissuing the same normalized embodied goal is rejected rather than sampled again.

## Fixed versions

| Component | Version |
|---|---|
| RoboAgent base | branch `demo/reading-setup-base-20260602`, commit `4ee8399ded2c92ca51b2aac8eac7e436f212b062` plus local changes |
| Python | 3.11.15 |
| Model | `kimi-k3` |
| Hermes Agent | v0.19.0, checkout `7651764ce63f44f4e02b1595798e73a67f678ebc` |
| OpenClaw | 2026.7.1, commit `2d2ddc4` |
| Node | v24.16.0 |
| A-MEM | `ceffb860f0712bbae97b184d440df62bc910ca8d` |

## Code map

| Location | Purpose |
|---|---|
| `simulator/world.py` | Stateful atomic physical executor and private scorer |
| `simulator/render.py` | State-consistent synthetic RGB-D rendering |
| `adapters/omniact.py` | Repository-native `BaseVLMPlanner` / OmniAct methods |
| `adapters/hermes.py` | Official Hermes AIAgent integration and A-MEM combinations |
| `adapters/openclaw.py` | Official OpenClaw embedded-agent integration |
| `mcp_server.py` | Hermes stdio MCP tools |
| `http_service.py`, `openclaw_plugin/` | Loopback service and native OpenClaw tools |
| `amem_service.py`, `hermes_amem_plugin/` | Official A-MEM service/provider bridge |
| `configs/formal_simulation.yaml` | Frozen probability, model, repeat, and safety settings |
| `configs/matrix_plan.json` | Frozen pilot, core, external, and mechanism matrices |
| `run_background_chunk.py` | Resumable eight-minute scheduled shard runner with gates |
| `launch_matrix.py` | Process-isolated matrix runner |
| `evaluator/summarize.py` | Raw-row aggregation, CSV output, Wilson intervals, usage counts |
| `tests/` | Protocol, simulator, gateway, and adapter contract tests |
| `reports/` | Audit, limitations, integration, and experiment reports |

## Environment checks

```bash
cd /Users/michaelshi/Localwork/roboagent

.venv/bin/python -m experiments.local_benchmark doctor \
  --config experiments/local_benchmark/configs/formal_simulation.yaml

.venv/bin/python -m pytest experiments/local_benchmark/tests -q

.venv/bin/python -m experiments.local_benchmark calibrate \
  --config experiments/local_benchmark/configs/formal_simulation.yaml \
  --trials 2000
```

Expected contract-test result: 20 passed. The latest calibration observed 0.604 for the configured 0.60 arm and 0.860 for the configured 0.85 arm; both passed the three-sigma check.

## Recommended resumable execution

Run one bounded shard window. Re-running the command resumes missing `(method, task, repetition)` rows and never replaces completed or failed rows.

```bash
cd /Users/michaelshi/Localwork/roboagent

.venv/bin/python -m experiments.local_benchmark.run_background_chunk \
  --max-wall-seconds 480
```

The gated order is:

1. `formal_pilot_v9_p60_p85_atomic_random` — 24 episodes.
2. `core_ablation_p60_p85_atomic_random_v1` — 480 episodes.
3. `main_external_p60_p85_atomic_random_v1` — 600 episodes.
4. `mechanisms_p60_p85_atomic_random_v1` — 288 episodes.

The core stage cannot start until the pilot has exactly 24 unique completed rows and zero protocol violations.

## Direct matrix commands

Use one worker for transport stability.

```bash
cd /Users/michaelshi/Localwork/roboagent

.venv/bin/python -m experiments.local_benchmark.launch_matrix \
  --matrix formal_pilot --workers 1 \
  --config experiments/local_benchmark/configs/formal_simulation.yaml \
  --run-dir experiments/local_benchmark/outputs/formal_pilot_v9_p60_p85_atomic_random

.venv/bin/python -m experiments.local_benchmark.launch_matrix \
  --matrix core_ablation --workers 1 \
  --config experiments/local_benchmark/configs/formal_simulation.yaml \
  --run-dir experiments/local_benchmark/outputs/core_ablation_p60_p85_atomic_random_v1

.venv/bin/python -m experiments.local_benchmark.launch_matrix \
  --matrix main_external --workers 1 \
  --config experiments/local_benchmark/configs/formal_simulation.yaml \
  --run-dir experiments/local_benchmark/outputs/main_external_p60_p85_atomic_random_v1

.venv/bin/python -m experiments.local_benchmark.launch_matrix \
  --matrix mechanisms --workers 1 \
  --config experiments/local_benchmark/configs/formal_simulation.yaml \
  --run-dir experiments/local_benchmark/outputs/mechanisms_p60_p85_atomic_random_v1
```

## Progress and summaries

```bash
cd /Users/michaelshi/Localwork/roboagent

sed -n '1,240p' \
  experiments/local_benchmark/outputs/formal_pilot_v9_p60_p85_atomic_random/completion.json

sed -n '1,200p' \
  experiments/local_benchmark/outputs/formal_pilot_v9_p60_p85_atomic_random/method_summary.csv

.venv/bin/python -m experiments.local_benchmark summarize \
  --run-dir experiments/local_benchmark/outputs/formal_pilot_v9_p60_p85_atomic_random
```

Every matrix root contains:

- `effective_config.json`: exact model, versions, probability semantics, task hash, and repository state.
- `raw_trials.jsonl`: immutable raw episode rows.
- `trials.csv`: flattened per-episode table.
- `method_summary.csv`: method-level task success, confidence intervals, token usage, and time.
- `summary.json`: method/domain/complexity aggregates.
- `completion.json`: expected/actual/unique rows, errors, and protocol violations.
- `episodes/` or `shards/`: raw model calls, images, private traces, framework stdout/stderr, and result JSON.

## Interpretation constraints

- Task success is not the same as the physical draw rate: the model can choose illegal or additional actions, and multi-operation tasks compound probabilities.
- The 3-repeat pilot validates plumbing only and must not be reported as a stable method comparison.
- `hermes_cam` and `hermes_amem_cam` receive the monitor-present 0.85 aggregate assumption in formal scoring. Their rows do not measure runtime CaM detection quality.
- The earlier adapted CaM implementation is retained only as engineering evidence and is not a complete Code-as-Monitor reproduction.
- Never merge directories marked `INVALID_*`, `PROTOCOL_SUPERSEDED*`, or older `p60_r2` / `independent_timeout` outputs into the current tables.
