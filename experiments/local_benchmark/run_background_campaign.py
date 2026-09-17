"""Run the frozen simulated benchmark campaign sequentially with stability gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import REPO_ROOT


OUTPUT_ROOT = REPO_ROOT / "experiments/local_benchmark/outputs"
CONFIG = REPO_ROOT / "experiments/local_benchmark/configs/formal_simulation.yaml"
STAGES = (
    ("formal_pilot", "formal_pilot_v9_p60_p85_atomic_random", 24, True),
    ("core_ablation", "core_ablation_p60_p85_atomic_random_v1", 480, False),
    ("main_external", "main_external_p60_p85_atomic_random_v1", 600, False),
    ("mechanisms", "mechanisms_p60_p85_atomic_random_v1", 288, False),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _audit(run_dir: Path, expected: int) -> dict[str, Any]:
    raw_path = run_dir / "raw_trials.jsonl"
    rows = []
    if raw_path.exists():
        rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [(row.get("method_id"), row.get("task_id"), int(row.get("seed", -1))) for row in rows]
    status_counts: dict[str, int] = {}
    method_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        status = str(row.get("status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1
        method = str(row.get("method_id", "missing"))
        method_entry = method_counts.setdefault(method, {"rows": 0, "successes": 0, "errors": 0})
        method_entry["rows"] += 1
        method_entry["successes"] += int(bool(row.get("score", {}).get("task_success")))
        method_entry["errors"] += int(status != "completed")
    error_rows = sum(count for status, count in status_counts.items() if status != "completed")
    unique_rows = len(set(keys))
    return {
        "expected_rows": expected,
        "actual_rows": len(rows),
        "unique_rows": unique_rows,
        "duplicate_rows": len(rows) - unique_rows,
        "status_counts": status_counts,
        "error_rows": error_rows,
        "error_rate": error_rows / unique_rows if unique_rows else 0.0,
        "method_counts": method_counts,
    }


def _acceptable(audit: dict[str, Any], *, strict: bool) -> bool:
    exact = (
        audit["actual_rows"] == audit["expected_rows"]
        and audit["unique_rows"] == audit["expected_rows"]
        and audit["duplicate_rows"] == 0
    )
    if not exact:
        return False
    if strict:
        return audit["error_rows"] == 0
    return audit["error_rate"] <= 0.05


def main() -> int:
    campaign_dir = OUTPUT_ROOT / "background_campaign_p60_p85_atomic_random_v1"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "state.json"
    _write_json(state_path, {
        "status": "running",
        "pid": os.getpid(),
        "started_at": _now(),
        "label": "SIMULATED PHYSICAL EXECUTION; NOT REAL ROBOT PERFORMANCE",
        "base_single_attempt_success_probability": 0.50,
        "effective_success_probability": {"monitor_present": 0.85, "monitor_absent": 0.60},
        "integrated_monitor_retry_limit": 2,
        "explicit_physical_retries_simulated": False,
        "repeat_ids": [0, 1, 2],
        "seed_controls_physics": False,
        "physical_rng_mode": "system_random_unseeded",
    })
    completed_stages: list[dict[str, Any]] = []
    for matrix, directory_name, expected, strict in STAGES:
        run_dir = OUTPUT_ROOT / directory_name
        before = _audit(run_dir, expected)
        if not _acceptable(before, strict=strict):
            state = {
                "status": "running_stage",
                "pid": os.getpid(),
                "updated_at": _now(),
                "current_stage": matrix,
                "run_dir": str(run_dir),
                "audit_before": before,
                "completed_stages": completed_stages,
            }
            _write_json(state_path, state)
            command = [
                str(REPO_ROOT / ".venv/bin/python"), "-m", "experiments.local_benchmark.launch_matrix",
                "--matrix", matrix, "--run-dir", str(run_dir), "--workers", "1",
                "--config", str(CONFIG),
            ]
            completed = subprocess.run(command, cwd=REPO_ROOT)
            if completed.returncode != 0:
                audit = _audit(run_dir, expected)
                _write_json(state_path, {
                    **state,
                    "status": "stopped",
                    "stopped_at": _now(),
                    "reason": f"launcher exited with status {completed.returncode}",
                    "audit_after": audit,
                })
                return completed.returncode
        audit = _audit(run_dir, expected)
        if not _acceptable(audit, strict=strict):
            reason = "pilot requires zero agent/infrastructure errors" if strict else "formal-stage error rate exceeded 5% or row contract failed"
            _write_json(state_path, {
                "status": "stopped",
                "pid": os.getpid(),
                "stopped_at": _now(),
                "current_stage": matrix,
                "reason": reason,
                "audit_after": audit,
                "completed_stages": completed_stages,
            })
            return 2
        completed_stages.append({"stage": matrix, "run_dir": str(run_dir), "audit": audit})
        _write_json(state_path, {
            "status": "stage_complete",
            "pid": os.getpid(),
            "updated_at": _now(),
            "completed_stages": completed_stages,
        })
    _write_json(state_path, {
        "status": "complete",
        "pid": os.getpid(),
        "completed_at": _now(),
        "completed_stages": completed_stages,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
