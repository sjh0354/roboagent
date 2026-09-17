# Local simulated-physical benchmark

This harness evaluates real agent/runtime and `kimi-k3` calls in a synthetic,
stateful RGB-D environment. It does **not** report real-robot performance and
does not reproduce the paper's hardware task set.

The formal local manifest has 40 independently designed tasks (20 manipulation
and 20 navigation, each split 4/8/8 across complexity levels). Per the updated
protocol, every selected task/method is run with repetition ids `0,1,2`. Physical
execution uses a direct aggregate probability: `p=0.85` for monitor-present
methods and `p=0.60` for monitor-absent methods. The assumed underlying single
attempt rate is 0.50; up to two monitor recoveries and time cost are already
folded into 0.85. No phase, timeout, or explicit physical retry is simulated.

The three public values `0,1,2` are repetition ids only. Each normalized
embodied goal is sampled exactly once with unseeded `SystemRandom`; the repeat
id never controls the draw. Reissuing the same physical goal is rejected so the
aggregate 60%/85% assumption cannot be counted twice. Digital tools remain
deterministic.

## Setup

```bash
cd "${ROBOAGENT_DIR:-$HOME/roboagent}"
python3.11 -m venv .venv
git clone https://github.com/agiresearch/A-mem.git third_party/A-MEM
git -C third_party/A-MEM checkout ceffb860f0712bbae97b184d440df62bc910ca8d
.venv/bin/python -m pip install -r experiments/local_benchmark/requirements-local-benchmark.lock.txt
```

Hermes v0.19.0 and OpenClaw v2026.7.1 are fixed external runtimes. The project
keeps their run state under each episode; it does not read or change personal
configuration. Credentials are read without executing
`$HOME/Localwork/test.sh`, unless `LOCAL_BENCHMARK_CREDENTIALS_SCRIPT` or explicit
`LOCAL_BENCHMARK_BASE_URL` / `LOCAL_BENCHMARK_API_KEY` variables are supplied.

## Commands

```bash
.venv/bin/python -m experiments.local_benchmark doctor --config experiments/local_benchmark/configs/formal_simulation.yaml
.venv/bin/python -m experiments.local_benchmark calibrate --config experiments/local_benchmark/configs/formal_simulation.yaml --trials 2000
.venv/bin/python -m experiments.local_benchmark smoke --config experiments/local_benchmark/configs/simulation.yaml --methods omniact,hermes_native,openclaw_native,hermes_amem,hermes_cam,hermes_amem_cam
.venv/bin/python -m experiments.local_benchmark run --config experiments/local_benchmark/configs/formal_simulation.yaml --matrix main --methods omniact,hermes_native,openclaw_native,hermes_amem,hermes_cam,hermes_amem_cam
.venv/bin/python -m experiments.local_benchmark run --config experiments/local_benchmark/configs/formal_simulation.yaml --matrix core_ablation --methods iterative_posthoc,iterative_memory,iterative_monitor,omniact
.venv/bin/python -m experiments.local_benchmark resume --config experiments/local_benchmark/configs/formal_simulation.yaml --run-dir <run-dir> --methods <same-method-list>
.venv/bin/python -m experiments.local_benchmark summarize --run-dir <run-dir>

# Frozen process-isolated launchers (exactly three repeat ids per task/method)
.venv/bin/python -m experiments.local_benchmark.launch_matrix --matrix formal_pilot --workers 4 --run-dir experiments/local_benchmark/outputs/formal_pilot_v1
.venv/bin/python -m experiments.local_benchmark.launch_matrix --matrix core_ablation --workers 4 --run-dir experiments/local_benchmark/outputs/core_ablation_v1
.venv/bin/python -m experiments.local_benchmark.launch_matrix --matrix main_external --workers 2 --run-dir experiments/local_benchmark/outputs/main_external_v1
.venv/bin/python -m experiments.local_benchmark.launch_matrix --matrix mechanisms --workers 4 --run-dir experiments/local_benchmark/outputs/mechanisms_v1
```

The launcher creates one process-isolated shard per method/task, resumes each
shard without replacing failures, and merges `raw_trials.jsonl`, `trials.csv`,
`method_summary.csv`, `summary.json`, and `completion.json` at the matrix root.
`formal_pilot` is only a protocol check and must not be reported as the full
40-task result.

`resume` skips every existing `(method, task, seed)` row, including failed
rows. It never retries failures until one happens to succeed.

All runs through `formal_pilot_v8_p60_r2_s3_independent_timeout` are retained
for audit but protocol-invalid. Formal reporting uses only the
`*_p60_p85_atomic_random*` directories.

In the simplified formal score, `hermes_cam` and `hermes_amem_cam` receive the
monitor-present 0.85 aggregate policy; their score is not a runtime measurement
of CaM detection quality. The earlier `cam_adapted_symbolic` runtime remains
separate engineering evidence and is not official/full Code-as-Monitor.
