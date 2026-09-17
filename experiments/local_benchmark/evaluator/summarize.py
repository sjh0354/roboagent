"""Aggregate raw episode records without dropping failures."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from ..io import REPO_ROOT


USAGE_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "total_tokens", "api_calls",
)


def _row_usage(row: Dict[str, Any]) -> Dict[str, int]:
    usage = {key: 0 for key in USAGE_KEYS}
    for source_name in ("usage", "amem_usage", "cam_generation_usage"):
        source = row.get(source_name) or {}
        for key in USAGE_KEYS:
            usage[key] += int(source.get(key) or 0)
    return usage


def _stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    successes = sum(bool(row["score"]["task_success"]) for row in rows)
    counts: Dict[str, int] = defaultdict(int)
    statuses: Dict[str, int] = defaultdict(int)
    usage = {key: 0 for key in USAGE_KEYS}
    for row in rows:
        statuses[str(row.get("status", "unknown"))] += 1
        for key, value in row["score"].get("counts", {}).items():
            counts[key] += int(value)
        for key, value in _row_usage(row).items():
            usage[key] += value
    return {
        "episodes": len(rows),
        "task_successes": successes,
        "task_success_rate": successes / len(rows),
        "wilson_95": _wilson(successes, len(rows)),
        "raw_counts": dict(counts),
        "statuses": dict(statuses),
        "usage": usage,
        "cost_status": "unknown: gateway/provider did not return billable price data",
        "wall_seconds": sum(float(row.get("wall_seconds", 0.0)) for row in rows),
    }


def _task_metadata(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    effective = run_dir / "effective_config.json"
    if not effective.exists():
        return {}
    provenance = json.loads(effective.read_text(encoding="utf-8"))
    manifest = Path(provenance["config"]["task_manifest"])
    if not manifest.is_absolute():
        recorded_root = Path(provenance.get("repo", {}).get("path", REPO_ROOT))
        recorded_manifest = recorded_root / manifest
        # A resumed run can legitimately move between hosts. Keep the original
        # provenance but read the same repo-relative manifest from this checkout.
        manifest = recorded_manifest if recorded_manifest.exists() else REPO_ROOT / manifest
    if not manifest.exists():
        return {}
    return {
        row["task_id"]: row
        for row in (json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def _write_tables(run_dir: Path, records: List[Dict[str, Any]], metadata: Dict[str, Dict[str, Any]], methods: Dict[str, Any]) -> None:
    count_keys = (
        "all_physical_attempts", "valid_physical_calls", "illegal_calls", "atomic_physical_resolutions",
        "semantic_successes", "semantic_failures", "duplicate_goal_rejections",
        "explicit_physical_retries", "digital_calls",
    )
    trial_fields = [
        "method_id", "task_id", "seed", "domain", "complexity", "status", "task_success",
        *count_keys, *USAGE_KEYS, "sim_time", "wall_seconds", "error",
    ]
    with (run_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_fields)
        writer.writeheader()
        for row in records:
            task = metadata.get(row["task_id"], {})
            counts = row.get("score", {}).get("counts", {})
            usage = _row_usage(row)
            writer.writerow({
                "method_id": row["method_id"], "task_id": row["task_id"], "seed": row["seed"],
                "domain": task.get("domain", "unknown"), "complexity": task.get("complexity", ""),
                "status": row.get("status", "unknown"), "task_success": bool(row["score"]["task_success"]),
                **{key: counts.get(key, 0) for key in count_keys}, **usage,
                "sim_time": row.get("score", {}).get("sim_time", ""),
                "wall_seconds": row.get("wall_seconds", ""), "error": row.get("error", ""),
            })
    method_fields = [
        "method_id", "episodes", "task_successes", "task_success_rate", "wilson_95_low", "wilson_95_high",
        *USAGE_KEYS, "wall_seconds", "status_counts",
    ]
    with (run_dir / "method_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=method_fields)
        writer.writeheader()
        for method_id, stats in methods.items():
            writer.writerow({
                "method_id": method_id, "episodes": stats["episodes"],
                "task_successes": stats["task_successes"], "task_success_rate": stats["task_success_rate"],
                "wilson_95_low": stats["wilson_95"][0], "wilson_95_high": stats["wilson_95"][1],
                **stats["usage"], "wall_seconds": stats["wall_seconds"],
                "status_counts": json.dumps(stats["statuses"], sort_keys=True),
            })


def _wilson(successes: int, total: int, z: float = 1.96) -> List[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [max(0.0, center - spread), min(1.0, center + spread)]


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    raw_path = run_dir / "raw_trials.jsonl"
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = _task_metadata(run_dir)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["method_id"]].append(record)
    methods = {}
    for method_id, rows in sorted(grouped.items()):
        methods[method_id] = _stats(rows)
    domain_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    complexity_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        task = metadata.get(row["task_id"], {})
        domain_groups[f"{row['method_id']}|{task.get('domain', 'unknown')}"] .append(row)
        complexity_groups[f"{row['method_id']}|{task.get('complexity', 'unknown')}"] .append(row)
    _write_tables(run_dir, records, metadata, methods)
    return {
        "label": "SIMULATED PHYSICAL EXECUTION; NOT REAL ROBOT PERFORMANCE",
        "run_dir": str(Path(run_dir).resolve()),
        "episodes": len(records),
        "methods": methods,
        "by_method_domain": {key: _stats(rows) for key, rows in sorted(domain_groups.items())},
        "by_method_complexity": {key: _stats(rows) for key, rows in sorted(complexity_groups.items())},
    }
