"""Configuration, manifest, provenance, and result I/O."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_manifest(path: str | Path) -> List[Dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    tasks = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task manifest contains duplicate task_id values")
    return tasks


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = REPO_ROOT / target
    return hashlib.sha256(target.read_bytes()).hexdigest()


def git_version(path: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    try:
        return {
            "path": str(path.resolve()),
            "sha": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "remote": run("remote", "get-url", "origin"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"path": str(path.resolve()), "unavailable": True}


def unique_run_dir(root: Path, matrix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / f"{timestamp}_{matrix}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{matrix}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")

