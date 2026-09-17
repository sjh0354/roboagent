"""Advance the frozen campaign in resumable shards for scheduled background runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluator.summarize import summarize_run
from .io import REPO_ROOT, file_sha256, git_version, load_json_config, load_manifest


CONFIG_PATH = REPO_ROOT / "experiments/local_benchmark/configs/formal_simulation.yaml"
PLAN_PATH = REPO_ROOT / "experiments/local_benchmark/configs/matrix_plan.json"
OUTPUT_ROOT = REPO_ROOT / "experiments/local_benchmark/outputs"
CAMPAIGN_DIR = OUTPUT_ROOT / "background_campaign_p60_p85_atomic_random_v1"
STAGES = (
    ("formal_pilot", "formal_pilot_v9_p60_p85_atomic_random", True),
    ("core_ablation", "core_ablation_p60_p85_atomic_random_v1", False),
    ("main_external", "main_external_p60_p85_atomic_random_v1", False),
    ("mechanisms", "mechanisms_p60_p85_atomic_random_v1", False),
)
SEEDS = [0, 1, 2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stage_definition(matrix: str, directory_name: str, strict: bool) -> dict[str, Any]:
    config = load_json_config(CONFIG_PATH)
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    spec = plan[matrix]
    tasks = load_manifest(config["task_manifest"])
    if spec["tasks"] != "all":
        selected = set(spec["tasks"])
        tasks = [task for task in tasks if task["task_id"] in selected]
    return {
        "matrix": matrix,
        "run_dir": OUTPUT_ROOT / directory_name,
        "strict": strict,
        "methods": list(spec["methods"]),
        "task_ids": [task["task_id"] for task in tasks],
        "expected": len(spec["methods"]) * len(tasks) * len(SEEDS),
        "config": config,
    }


def _merge(stage: dict[str, Any]) -> dict[str, Any]:
    run_dir = stage["run_dir"]
    rows: list[dict[str, Any]] = []
    for raw_path in sorted((run_dir / "shards").glob("*/*/raw_trials.jsonl")):
        rows.extend(_read_rows(raw_path))
    rows.sort(key=lambda row: (row["method_id"], row["task_id"], int(row["seed"])))
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_trials.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    keys = [(row["method_id"], row["task_id"], int(row["seed"])) for row in rows]
    statuses: dict[str, int] = {}
    methods: dict[str, dict[str, int]] = {}
    protocol_violations: list[str] = []
    for row in rows:
        status = str(row.get("status", "missing"))
        statuses[status] = statuses.get(status, 0) + 1
        method = str(row["method_id"])
        entry = methods.setdefault(method, {"rows": 0, "successes": 0, "errors": 0})
        entry["rows"] += 1
        entry["successes"] += int(bool(row.get("score", {}).get("task_success")))
        entry["errors"] += int(status != "completed")
        if status != "completed":
            continue
        key = f"{method}/{row['task_id']}/seed_{row['seed']}"
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
    unique_rows = len(set(keys))
    errors = sum(count for status, count in statuses.items() if status != "completed")
    audit = {
        "expected_rows": stage["expected"],
        "actual_rows": len(rows),
        "unique_rows": unique_rows,
        "duplicate_rows": len(rows) - unique_rows,
        "statuses": statuses,
        "error_rows": errors,
        "error_rate": errors / unique_rows if unique_rows else 0.0,
        "methods": methods,
        "protocol_violations": protocol_violations,
    }
    complete = (
        len(rows) == stage["expected"]
        and unique_rows == stage["expected"]
        and audit["duplicate_rows"] == 0
        and not protocol_violations
    )
    audit["complete"] = complete
    _write_json(run_dir / "completion.json", audit)
    if rows:
        _write_json(run_dir / "summary.json", summarize_run(run_dir))
    return audit


def _write_provenance(stage: dict[str, Any]) -> None:
    path = stage["run_dir"] / "effective_config.json"
    if path.exists():
        return
    config = stage["config"]
    _write_json(path, {
        "created_at": _now(),
        "matrix": stage["matrix"],
        "methods": stage["methods"],
        "tasks": stage["task_ids"],
        "repeat_ids": SEEDS,
        "seed_controls_physics": False,
        "physical_rng_mode": "system_random_unseeded",
        "execution_model": "direct_aggregate_bernoulli_per_normalized_embodied_goal",
        "base_single_attempt_success_probability": 0.50,
        "effective_success_probability": {"monitor_present": 0.85, "monitor_absent": 0.60},
        "integrated_monitor_retry_limit": 2,
        "explicit_physical_retries_simulated": False,
        "workers": 1,
        "simulated_physical_execution": True,
        "config": config,
        "manifest_sha256": file_sha256(config["task_manifest"]),
        "repo": git_version(REPO_ROOT),
    })


def _next_shard(stage: dict[str, Any]) -> tuple[str, str] | None:
    for method in stage["methods"]:
        for task_id in stage["task_ids"]:
            rows = _read_rows(stage["run_dir"] / "shards" / method / task_id / "raw_trials.jsonl")
            keys = {(row.get("method_id"), row.get("task_id"), int(row.get("seed", -1))) for row in rows}
            expected = {(method, task_id, seed) for seed in SEEDS}
            if len(rows) != len(keys):
                raise RuntimeError(f"duplicate rows in shard {method}/{task_id}")
            if keys != expected:
                return method, task_id
    return None


def _unacceptable(stage: dict[str, Any], audit: dict[str, Any]) -> str | None:
    if audit["duplicate_rows"]:
        return "duplicate result keys detected"
    if audit["protocol_violations"]:
        return "execution-control or physical-randomness protocol violation detected"
    if stage["strict"] and audit["error_rows"]:
        return "pilot requires zero agent/infrastructure errors"
    if not stage["strict"] and audit["unique_rows"] >= 20 and audit["error_rate"] > 0.05:
        return "agent/infrastructure error rate exceeded 5%"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-wall-seconds", type=int, default=540)
    args = parser.parse_args()
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    lock = (CAMPAIGN_DIR / ".chunk.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    started = time.monotonic()
    while time.monotonic() - started < args.max_wall_seconds:
        advanced = False
        for matrix, directory_name, strict in STAGES:
            stage = _stage_definition(matrix, directory_name, strict)
            _write_provenance(stage)
            audit = _merge(stage)
            reason = _unacceptable(stage, audit)
            if reason:
                _write_json(CAMPAIGN_DIR / "state.json", {
                    "status": "stopped", "updated_at": _now(), "stage": matrix,
                    "run_dir": str(stage["run_dir"]), "reason": reason, "audit": audit,
                })
                return 2
            shard = _next_shard(stage)
            if shard is None:
                if not audit["complete"]:
                    raise RuntimeError(f"stage row contract incomplete: {matrix}")
                continue
            method, task_id = shard
            _write_json(CAMPAIGN_DIR / "state.json", {
                "status": "running_shard", "updated_at": _now(), "stage": matrix,
                "run_dir": str(stage["run_dir"]), "method": method, "task_id": task_id,
                "audit_before": audit, "repeat_ids": SEEDS,
                "base_single_attempt_success_probability": 0.50,
                "effective_success_probability": {"monitor_present": 0.85, "monitor_absent": 0.60},
                "physical_rng_mode": "system_random_unseeded",
                "seed_controls_physics": False,
            })
            log_path = CAMPAIGN_DIR / "chunk_logs" / matrix / method / f"{task_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(REPO_ROOT / ".venv/bin/python"), "-m", "experiments.local_benchmark", "run",
                "--config", str(CONFIG_PATH), "--matrix", matrix, "--methods", method,
                "--seeds", ",".join(map(str, SEEDS)), "--tasks", task_id,
                "--run-dir", str(stage["run_dir"] / "shards" / method / task_id),
            ]
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{_now()}] {' '.join(command)}\n")
                completed = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            audit = _merge(stage)
            if completed.returncode != 0:
                _write_json(CAMPAIGN_DIR / "state.json", {
                    "status": "stopped", "updated_at": _now(), "stage": matrix,
                    "run_dir": str(stage["run_dir"]), "reason": f"shard exited {completed.returncode}",
                    "method": method, "task_id": task_id, "audit": audit,
                })
                return completed.returncode
            reason = _unacceptable(stage, audit)
            if reason:
                _write_json(CAMPAIGN_DIR / "state.json", {
                    "status": "stopped", "updated_at": _now(), "stage": matrix,
                    "run_dir": str(stage["run_dir"]), "reason": reason,
                    "method": method, "task_id": task_id, "audit": audit,
                })
                return 2
            advanced = True
            break
        if not advanced:
            _write_json(CAMPAIGN_DIR / "state.json", {
                "status": "complete", "completed_at": _now(),
                "effective_success_probability": {"monitor_present": 0.85, "monitor_absent": 0.60},
                "physical_rng_mode": "system_random_unseeded", "seed_controls_physics": False,
            })
            return 0
    _write_json(CAMPAIGN_DIR / "state.json", {
        "status": "scheduled_pause", "updated_at": _now(),
        "reason": "wall-time slice ended; next heartbeat resumes the next shard",
        "effective_success_probability": {"monitor_present": 0.85, "monitor_absent": 0.60},
        "physical_rng_mode": "system_random_unseeded", "seed_controls_physics": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
