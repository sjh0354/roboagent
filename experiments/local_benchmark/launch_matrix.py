"""Resumable process-isolated launcher for the frozen three-seed matrices."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .evaluator.summarize import summarize_run
from .io import REPO_ROOT, file_sha256, git_version, load_json_config, load_manifest


DEFAULT_CONFIG = REPO_ROOT / "experiments/local_benchmark/configs/formal_simulation.yaml"
DEFAULT_PLAN = REPO_ROOT / "experiments/local_benchmark/configs/matrix_plan.json"


def _job(command: list[str], log_path: Path) -> Dict[str, Any]:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + "\n[stderr]\n" + completed.stderr, encoding="utf-8")
    return {"returncode": completed.returncode, "log": str(log_path), "command": command}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config = load_json_config(args.config)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if args.matrix not in plan:
        raise ValueError(f"unknown frozen matrix: {args.matrix}")
    matrix = plan[args.matrix]
    seeds = [int(seed) for seed in plan["episode_seeds"]]
    if seeds != [0, 1, 2]:
        raise ValueError("formal protocol requires exactly three repeat ids 0..2")
    tasks = load_manifest(config["task_manifest"])
    task_ids = [task["task_id"] for task in tasks]
    if matrix["tasks"] != "all":
        selected = set(matrix["tasks"])
        task_ids = [task_id for task_id in task_ids if task_id in selected]
        missing = selected - set(task_ids)
        if missing:
            raise ValueError(f"matrix plan refers to unknown tasks: {sorted(missing)}")
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    launcher_lock = (run_dir / ".launcher.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(launcher_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        launcher_lock.close()
        raise RuntimeError(f"matrix launcher is already active: {run_dir}") from error
    launcher_lock.seek(0)
    launcher_lock.truncate()
    launcher_lock.write(f"pid={os.getpid()}\n")
    launcher_lock.flush()
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrix": args.matrix,
        "methods": matrix["methods"],
        "tasks": task_ids,
        "repeat_ids": seeds,
        "seed_controls_physics": False,
        "physical_rng_mode": "system_random_unseeded",
        "execution_model": "direct_aggregate_bernoulli_per_normalized_embodied_goal",
        "base_single_attempt_success_probability": 0.50,
        "effective_success_probability": {"monitor_present": 0.85, "monitor_absent": 0.60},
        "integrated_monitor_retry_limit": 2,
        "explicit_physical_retries_simulated": False,
        "workers": args.workers,
        "simulated_physical_execution": True,
        "matrix_plan": matrix,
        "config": config,
        "manifest_sha256": file_sha256(config["task_manifest"]),
        "repo": git_version(REPO_ROOT),
    }
    (run_dir / "effective_config.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    commands = []
    for method in matrix["methods"]:
        for task_id in task_ids:
            shard = run_dir / "shards" / method / task_id
            command = [
                str(REPO_ROOT / ".venv/bin/python"), "-m", "experiments.local_benchmark", "run",
                "--config", str(Path(args.config).resolve()), "--matrix", args.matrix,
                "--methods", method, "--seeds", ",".join(map(str, seeds)), "--tasks", task_id,
                "--run-dir", str(shard),
            ]
            commands.append((command, run_dir / "launcher_logs" / method / f"{task_id}.log"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, command, log) for command, log in commands]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    failed = [result for result in results if result["returncode"] != 0]
    all_rows = []
    for raw in sorted((run_dir / "shards").glob("*/*/raw_trials.jsonl")):
        all_rows.extend(json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip())
    all_rows.sort(key=lambda row: (row["method_id"], row["task_id"], int(row["seed"])))
    with (run_dir / "raw_trials.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    expected = len(matrix["methods"]) * len(task_ids) * len(seeds)
    keys = [(row["method_id"], row["task_id"], int(row["seed"])) for row in all_rows]
    unique_rows = len(set(keys))
    duplicate_rows = len(keys) - unique_rows
    protocol_violations = []
    for row in all_rows:
        if row.get("status") != "completed":
            continue
        key = f"{row['method_id']}/{row['task_id']}/seed_{row['seed']}"
        score = row.get("score", {})
        counts = score.get("counts", {})
        if score.get("physical_rng_mode") != "system_random_unseeded" or score.get("seed_controls_physics") is not False:
            protocol_violations.append(f"{key}: physics is not pure unseeded system randomness")
        if score.get("execution_model") != "direct_aggregate_bernoulli_per_normalized_embodied_goal":
            protocol_violations.append(f"{key}: wrong physical execution model")
        valid = int(counts.get("valid_physical_calls", 0))
        atomic = int(counts.get("atomic_physical_resolutions", -1))
        if atomic != valid or int(counts.get("explicit_physical_retries", -1)) != 0:
            protocol_violations.append(f"{key}: physical calls were not one-shot aggregate resolutions")
        monitor_present = score.get("monitor_present")
        probability = score.get("effective_physical_success_probability")
        expected_probability = 0.85 if monitor_present is True else 0.60 if monitor_present is False else None
        if expected_probability is None or probability != expected_probability:
            protocol_violations.append(f"{key}: monitor/probability assignment mismatch")
        if row.get("explicit_physical_retries_simulated") is not False:
            protocol_violations.append(f"{key}: explicit retries were not disabled")
    non_completed_rows = [
        {
            "method_id": row.get("method_id"), "task_id": row.get("task_id"),
            "seed": row.get("seed"), "status": row.get("status"),
        }
        for row in all_rows if row.get("status") != "completed"
    ]
    completion = {
        "expected_rows": expected,
        "actual_rows": len(all_rows),
        "unique_rows": unique_rows,
        "duplicate_rows": duplicate_rows,
        "complete": (
            len(all_rows) == expected and unique_rows == expected and duplicate_rows == 0
            and not failed and not protocol_violations and not non_completed_rows
        ),
        "failed_shards": failed,
        "non_completed_rows": non_completed_rows,
        "protocol_violations": protocol_violations,
    }
    (run_dir / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    if all_rows:
        summary = summarize_run(run_dir)
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(completion, indent=2))
    return 0 if completion["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
