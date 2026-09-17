"""Unified doctor/calibrate/smoke/run/resume/summarize command line."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .adapters.hermes import _hermes_runtime, run_hermes_episode
from .adapters.omniact import METHODS as OMNIACT_METHODS, run_omniact_episode
from .adapters.openclaw import _runtime as openclaw_runtime, run_openclaw_episode
from .evaluator.summarize import summarize_run
from .io import REPO_ROOT, append_jsonl, file_sha256, git_version, load_json_config, load_manifest, unique_run_dir
from .model import MODEL_NAME, resolve_credentials
from .simulator.world import StatefulWorld


DEFAULT_CONFIG = REPO_ROOT / "experiments/local_benchmark/configs/simulation.yaml"
OUTPUT_ROOT = REPO_ROOT / "experiments/local_benchmark/outputs"
EXTERNAL_METHODS = {"hermes_native", "openclaw_native", "hermes_amem", "hermes_cam", "hermes_amem_cam"}
METHODS = tuple(OMNIACT_METHODS) + tuple(sorted(EXTERNAL_METHODS))
MONITORED_EXTERNAL_METHODS = {"hermes_cam", "hermes_amem_cam"}


def _method_has_monitor(method: str) -> bool:
    if method in OMNIACT_METHODS:
        return OMNIACT_METHODS[method]["monitor"] != "none"
    return method in MONITORED_EXTERNAL_METHODS


def _print(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def doctor(config_path: str) -> int:
    config = load_json_config(config_path)
    tasks = load_manifest(config["task_manifest"])
    base_url, key = resolve_credentials()
    try:
        hermes_cli, hermes_python = _hermes_runtime()
        hermes_error = None
    except RuntimeError as exc:
        hermes_cli, hermes_python, hermes_error = None, None, str(exc)
    try:
        node_bin, openclaw_cli = openclaw_runtime()
        openclaw_error = None
    except RuntimeError as exc:
        node_bin, openclaw_cli, openclaw_error = None, None, str(exc)
    checks = {
        "config_valid": True,
        "model_locked_to_kimi_k3": config.get("model") == MODEL_NAME,
        "base_single_attempt_success_prob": config.get("base_single_attempt_success_prob"),
        "effective_physical_success_prob": config.get("effective_physical_success_prob"),
        "real_hardware_disabled": config.get("safety", {}).get("real_hardware_disabled") is True,
        "manifest_tasks": len(tasks),
        "manifest_sha256": file_sha256(config["task_manifest"]),
        "credentials_available": bool(base_url and key),
        "base_url_host": base_url.split("//", 1)[-1].split("/", 1)[0] if base_url else None,
        "dependencies": {name: importlib.util.find_spec(name) is not None for name in ("PIL", "numpy", "openai", "mcp")},
        "hermes_cli": str(hermes_cli) if hermes_cli else None,
        "hermes_python": str(hermes_python) if hermes_python else None,
        "hermes_error": hermes_error,
        "openclaw_cli": str(openclaw_cli) if openclaw_cli else None,
        "node_bin_dir": str(node_bin) if node_bin else None,
        "openclaw_error": openclaw_error,
        "repo": git_version(REPO_ROOT),
    }
    checks["ready_for_omniact"] = all([
        checks["model_locked_to_kimi_k3"], checks["real_hardware_disabled"], checks["credentials_available"],
        checks["dependencies"]["PIL"], checks["dependencies"]["numpy"], checks["dependencies"]["openai"],
    ])
    checks["ready_for_hermes"] = bool(
        checks["ready_for_omniact"] and hermes_cli and hermes_python and checks["dependencies"]["mcp"]
    )
    checks["ready_for_openclaw"] = bool(checks["ready_for_omniact"] and openclaw_cli and node_bin)
    checks["ready_for_full_campaign"] = bool(checks["ready_for_hermes"] and checks["ready_for_openclaw"])
    _print(checks)
    return 0 if checks["ready_for_omniact"] else 2


def calibrate(config_path: str, trials: int) -> int:
    config = load_json_config(config_path)
    task = {
        "task_id": "executor_calibration",
        "request": "calibration only",
        "initial_state": {"robot_location": "desk", "objects": {"calibration_block": "desk"}, "devices": {}, "preferences": []},
        "success": {"objects": {"calibration_block": "basket"}},
    }
    out = OUTPUT_ROOT / "calibration" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    policy_results = {}
    compatible = True
    for policy, configured_probability in config["effective_physical_success_prob"].items():
        monitor_present = policy == "monitor_present"
        world = StatefulWorld(
            task, seed=711, output_dir=out / policy,
            physical_success_prob=float(configured_probability),
            monitor_present=monitor_present,
            base_single_attempt_prob=float(config["base_single_attempt_success_prob"]),
            integrated_monitor_retry_limit=int(config["integrated_monitor_retry_limit"]),
            render_enabled=False,
        )
        successes = 0
        for index in range(trials):
            destination = f"{policy}_calibration_target_{index}"
            world.robot_location = "desk"
            world.start_skill("navigate", {"destination": destination})
            successes += world.robot_location == destination
        rate = successes / trials
        probability = float(configured_probability)
        tolerance = 3.0 * (probability * (1.0 - probability) / trials) ** 0.5
        policy_compatible = abs(rate - probability) <= tolerance
        compatible = compatible and policy_compatible
        policy_results[policy] = {
            "configured_probability": probability,
            "trials": trials,
            "successes": successes,
            "observed_rate": rate,
            "three_sigma_tolerance": tolerance,
            "compatible": policy_compatible,
            "counts": world.counts,
        }
    result = {
        "label": "executor calibration only; not agent task scores",
        "execution_model": "direct aggregate Bernoulli per normalized embodied goal",
        "base_single_attempt_probability": float(config["base_single_attempt_success_prob"]),
        "integrated_monitor_retry_limit": int(config["integrated_monitor_retry_limit"]),
        "explicit_physical_retries_simulated": False,
        "physical_rng_mode": "system_random_unseeded",
        "seed_controls_physics": False,
        "policies": policy_results,
        "compatible": compatible,
    }
    (out / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print(result)
    return 0 if compatible else 1


def _run(
    *,
    config_path: str,
    matrix: str,
    methods: List[str],
    seeds: List[int],
    task_ids: List[str],
    run_dir: Path | None = None,
    verbose: bool = False,
) -> Path:
    config = load_json_config(config_path)
    if config.get("model") != MODEL_NAME:
        raise ValueError("benchmark model must be exactly kimi-k3")
    tasks = load_manifest(config["task_manifest"])
    if task_ids:
        selected = set(task_ids)
        tasks = [task for task in tasks if task["task_id"] in selected]
        missing = selected - {task["task_id"] for task in tasks}
        if missing:
            raise ValueError(f"unknown task ids: {sorted(missing)}")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"methods not yet available in this runner: {sorted(unknown)}")
    run_dir = run_dir or unique_run_dir(OUTPUT_ROOT, matrix)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_dir / ".active.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"run directory is already active: {run_dir}") from error
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()}\n")
    lock_handle.flush()
    probability_map = config["effective_physical_success_prob"]
    base_probability = float(config["base_single_attempt_success_prob"])
    integrated_retry_limit = int(config["integrated_monitor_retry_limit"])
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matrix": matrix,
        "methods": methods,
        "repeat_ids": seeds,
        "seed_controls_physics": False,
        "physical_rng_mode": "system_random_unseeded",
        "execution_model": "direct_aggregate_bernoulli_per_normalized_embodied_goal",
        "explicit_physical_retries_simulated": False,
        "model": MODEL_NAME,
        "simulated_physical_execution": True,
        "config": config,
        "manifest_sha256": file_sha256(config["task_manifest"]),
        "repo": git_version(REPO_ROOT),
    }
    (run_dir / "effective_config.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    raw_path = run_dir / "raw_trials.jsonl"
    existing = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing.add((row["method_id"], row["task_id"], int(row["seed"])))
    for method in methods:
        monitor_present = _method_has_monitor(method)
        configured_probability = float(probability_map["monitor_present" if monitor_present else "monitor_absent"])
        for task in tasks:
            for seed in seeds:
                key = (method, task["task_id"], seed)
                if key in existing:
                    continue
                episode_dir = run_dir / "episodes" / method / task["task_id"] / f"seed_{seed}"
                try:
                    if method in OMNIACT_METHODS:
                        world = StatefulWorld(
                            task, seed=seed, output_dir=episode_dir,
                            physical_success_prob=configured_probability,
                            monitor_present=monitor_present,
                            base_single_attempt_prob=base_probability,
                            integrated_monitor_retry_limit=integrated_retry_limit,
                        )
                        result = run_omniact_episode(
                            world,
                            method,
                            episode_dir,
                            max_steps=min(int(config["max_planner_steps"]), int(task.get("max_actions", config["max_planner_steps"]))),
                            verbose=verbose,
                        )
                        result["status"] = "agent_error" if result.get("agent_error") else "completed"
                    elif method == "openclaw_native":
                        result = run_openclaw_episode(
                            task, seed, episode_dir, configured_probability,
                            integrated_monitor_retry_limit=integrated_retry_limit,
                        )
                    else:
                        result = run_hermes_episode(
                            task, seed, episode_dir, configured_probability,
                            integrated_monitor_retry_limit=integrated_retry_limit,
                            use_amem=method in {"hermes_amem", "hermes_amem_cam"},
                            cam_mode="cam_adapted_symbolic" if method in {"hermes_cam", "hermes_amem_cam"} else "none",
                        )
                except Exception as error:
                    result = {
                        "method_id": method,
                        "task_id": task["task_id"],
                        "seed": seed,
                        "model": MODEL_NAME,
                        "simulated_physical_execution": True,
                        "status": "infrastructure_error",
                        "error": f"{type(error).__name__}: {error}",
                        "score": {"task_success": False, "checks": [], "counts": {}},
                    }
                (episode_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                append_jsonl(raw_path, [result])
                print(f"[{result['status']}] {method} {task['task_id']} seed={seed} success={result['score']['task_success']}", flush=True)
                cooldown = float(config.get("inter_episode_cooldown_seconds", 0.0))
                if cooldown > 0:
                    time.sleep(cooldown)
    summary = summarize_run(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RoboAgent local simulated-physical benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor_p = sub.add_parser("doctor")
    doctor_p.add_argument("--config", default=str(DEFAULT_CONFIG))
    calibration_p = sub.add_parser("calibrate")
    calibration_p.add_argument("--config", default=str(DEFAULT_CONFIG))
    calibration_p.add_argument("--trials", type=int, default=2000)
    for name in ("smoke", "run", "resume"):
        current = sub.add_parser(name)
        current.add_argument("--config", default=str(DEFAULT_CONFIG))
        current.add_argument("--matrix", default="smoke" if name == "smoke" else "core_ablation")
        current.add_argument("--methods", default="omniact" if name == "smoke" else ",".join(METHODS))
        current.add_argument("--seeds", default="0" if name == "smoke" else "")
        current.add_argument("--tasks", default="dev_manipulation_clear_desk" if name == "smoke" else "")
        current.add_argument("--run-dir", required=name == "resume")
        current.add_argument("--verbose", action="store_true")
    summary_p = sub.add_parser("summarize")
    summary_p.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(args.config)
    if args.command == "calibrate":
        return calibrate(args.config, args.trials)
    if args.command == "summarize":
        summary = summarize_run(Path(args.run_dir))
        (Path(args.run_dir) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _print(summary)
        return 0
    methods = [item for item in args.methods.split(",") if item]
    config = load_json_config(args.config)
    seeds = [int(item) for item in args.seeds.split(",") if item] if args.seeds else [int(item) for item in config.get("episode_seeds", [0, 1, 2])]
    task_ids = [item for item in args.tasks.split(",") if item]
    run_dir = _run(config_path=args.config, matrix=args.matrix, methods=methods, seeds=seeds, task_ids=task_ids, run_dir=Path(args.run_dir) if args.run_dir else None, verbose=args.verbose)
    print(str(run_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
