# Long-Horizon Cyber-Physical Orchestration Benchmark

This benchmark evaluates whether an agent can output the right action trace for
long-horizon cyber-physical tasks across two hardware profiles:

- Family A: UR5e desktop orchestration tasks.
- Family B: `humanoid_g1` mobile spatial-interaction tasks.

The batch benchmark is offline. A task can fail because the agent produced the
wrong action sequence, or because a simulated physical action failed. Physical
execution is sampled deterministically with a configurable per-action success
rate, defaulting to `0.7`.

Run:

```bash
python -m experiments.long_horizon_orchestration.run_eval \
  --methods omniact,react_vla,react_plus_vla \
  --physical-success-rate 0.7 \
  --seed 7
```

Outputs:

- `outputs/results.jsonl`: per-case traces, sampled physical results, and scores.
- `outputs/table1_summary.csv`: method-level Table 1 metrics.

Metrics:

- `overall_e2e_success`: all required cyber, physical, visual, and final-report
  checks pass, and sampled physical execution succeeds.
- `cyber_iot_success`: required tool/API/IoT actions are correct.
- `physical_command_success`: required physical action commands are correct.
- `physical_execution_success`: required physical actions are sampled as
  successful.
- `action_correctness`: every expected action is present with the required type
  and key parameters.

Baseline traces are deterministic capability models:

- `Reactive VLA`: physical actions only, no cyber/API/IoT or reporting tools.
- `ReAct + VLA`: regular tool and act calls, but no hierarchical memory or
  async visual verification.
- `OmniAct`: full expected trace in scripted mode. With `--trace-dir`, real
  planner outputs are scored directly, so OmniAct can fail from wrong action
  calls before physical execution is sampled.
