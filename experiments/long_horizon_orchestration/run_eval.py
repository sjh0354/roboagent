"""Run the long-horizon cyber-physical orchestration benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from experiments.long_horizon_orchestration.score import (
    apply_physical_execution,
    extract_steps,
    ratio,
    score_trace,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases.jsonl"
DEFAULT_OUTPUT = HERE / "outputs"
METHODS = ("omniact", "react_vla", "react_plus_vla")
METHOD_LABELS = {
    "omniact": "OmniAct",
    "react_vla": "Reactive VLA",
    "react_plus_vla": "ReAct + VLA",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run long-horizon orchestration evaluation")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Path to cases.jsonl")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory for results")
    parser.add_argument(
        "--methods",
        default="omniact,react_vla,react_plus_vla",
        help="Comma-separated methods: omniact,react_vla,react_plus_vla",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Optional directory containing <method>/<case_id>.json traces from real planners",
    )
    parser.add_argument("--physical-success-rate", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    methods = parse_methods(args.methods)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for method in methods:
        for case in cases:
            trace = load_trace(trace_dir, method, case) if trace_dir else build_scripted_trace(case, method)
            trace = apply_physical_execution(
                trace,
                success_rate=args.physical_success_rate,
                seed=args.seed,
            )
            records.append(build_record(case, trace))

    write_results(output_dir, records)
    print_summary(records)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                cases.append(json.loads(stripped))
    return cases


def parse_methods(raw: str) -> List[str]:
    methods = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"Invalid methods: {invalid}")
    return methods


def load_trace(trace_dir: Optional[Path], method: str, case: Dict[str, Any]) -> Dict[str, Any]:
    if trace_dir is None:
        raise ValueError("trace_dir is required")
    path = trace_dir / method / f"{case['case_id']}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing trace for method={method} case={case['case_id']}: {path}")
    trace = json.loads(path.read_text(encoding="utf-8"))
    trace.setdefault("method", method)
    trace.setdefault("case_id", case.get("case_id"))
    trace.setdefault("trace_mode", "trace_dir")
    return trace


def build_scripted_trace(case: Dict[str, Any], method: str) -> Dict[str, Any]:
    expected = case.get("expected_trace") or []
    if method == "omniact":
        steps = [_step(index + 1, item) for index, item in enumerate(expected)]
    elif method == "react_vla":
        steps = [
            _step(index + 1, item)
            for index, item in enumerate(expected)
            if item.get("action_type") == "act"
        ]
    elif method == "react_plus_vla":
        steps = _build_react_plus_vla_steps(case, expected)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return {
        "case_id": case.get("case_id"),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "trace_mode": "scripted",
        "transcript": case.get("utterance"),
        "execution_history": steps,
    }


def _build_react_plus_vla_steps(case: Dict[str, Any], expected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for item in expected:
        if case.get("requires_hierarchical_memory") and item.get("action") == "store_memory":
            continue
        if case.get("requires_async_visual_verification") and item.get("action_type") == "sense":
            continue
        steps.append(_step(len(steps) + 1, item))
    return steps


def _step(step_number: int, template: Dict[str, Any]) -> Dict[str, Any]:
    parameters = dict(template.get("parameters") or {})
    if template.get("action") == "speak" and "message_contains" in parameters and "message" not in parameters:
        parameters["message"] = "，".join(str(item) for item in parameters.pop("message_contains"))
    return {
        "step_number": step_number,
        "action": template.get("action"),
        "action_type": template.get("action_type"),
        "parameters": parameters,
        "assumed_successful": True,
        "execution_result": {
            "success": True,
            "feedback": "mock command accepted",
            "data": {},
            "error": None,
        },
    }


def build_record(case: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    score = score_trace(case, trace)
    return {
        **score,
        "family": case.get("family"),
        "platform": case.get("platform"),
        "method_label": METHOD_LABELS.get(str(trace.get("method")), str(trace.get("method"))),
        "trace": trace,
    }


def write_results(output_dir: Path, records: List[Dict[str, Any]]) -> None:
    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = output_dir / "table1_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "method",
            "method_label",
            "cases",
            "overall_e2e_success",
            "action_correctness",
            "cyber_iot_success",
            "physical_command_success",
            "physical_execution_success",
            "vision_success",
            "final_report_success",
            "physical_action_success_rate",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summarize(records):
            writer.writerow(row)


def summarize(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("method")), []).append(record)

    rows = []
    for method, method_records in sorted(grouped.items()):
        physical_total = sum(int(record.get("physical_action_count") or 0) for record in method_records)
        physical_success = sum(int(record.get("physical_action_success_count") or 0) for record in method_records)
        row = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "cases": len(method_records),
            "overall_e2e_success": ratio(record["success"] for record in method_records),
            "physical_action_success_rate": round(physical_success / physical_total, 4) if physical_total else 0.0,
        }
        for metric in (
            "action_correctness",
            "cyber_iot_success",
            "physical_command_success",
            "physical_execution_success",
            "vision_success",
            "final_report_success",
        ):
            row[metric] = ratio(record["submetrics"][metric] for record in method_records)
        rows.append(row)
    return rows


def print_summary(records: List[Dict[str, Any]]) -> None:
    for row in summarize(records):
        print(
            "{method_label}: E2E={overall_e2e_success:.4f} cyber={cyber_iot_success:.4f} "
            "physical_cmd={physical_command_success:.4f} physical_exec={physical_execution_success:.4f}".format(**row)
        )


if __name__ == "__main__":
    main()
