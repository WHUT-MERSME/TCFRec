#!/usr/bin/env python3
"""Canonical one-command reproduction driver for the anonymous TCF artifact.

The result-producing modes train every neural checkpoint from the configured
seed.  They never pass a checkpoint or warm-start option to training.  The
training-selected checkpoint from that same run is loaded only afterward for
strict test evaluation with the fixed, non-trainable Markov calibration.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts import evaluate_tcf as evaluator  # noqa: E402


DEFAULT_MANIFEST = SRC_DIR / "configs" / "tcf_reproduction.json"
DEFAULT_OUTPUT_ROOT = SRC_DIR / "output" / "tcf_reproduction"
DATASET_ORDER = ("Beauty", "Sports", "LastFM", "ML1M")
METRICS = ("HR@10", "NDCG@10", "HR@20", "NDCG@20")
FORBIDDEN_TRAINING_OPTIONS = ("--load_model", "--complex_dff_warmstart")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "missing"


def normalized_version(value: str) -> str:
    return value.split("+", 1)[0]


def environment_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest["environment"]
    actual = {
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": normalized_version(torch.__version__),
        "numpy": normalized_version(package_version("numpy")),
        "scipy": normalized_version(package_version("scipy")),
        "tqdm": normalized_version(package_version("tqdm")),
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
    }
    checked_fields = ("python_major_minor", "torch", "numpy", "scipy", "tqdm")
    mismatches = {
        field: {"expected": str(expected[field]), "actual": str(actual[field])}
        for field in checked_fields
        if str(actual[field]) != str(expected[field])
    }
    return {
        "expected": expected,
        "actual": actual,
        "exact_versions": not mismatches,
        "mismatches": mismatches,
    }


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tcf-reproduction-v1":
        raise ValueError(f"unsupported reproduction manifest: {path}")
    training = payload.get("training", {})
    required_training = {
        "load_model": None,
        "warmstart": None,
        "train_scope": "all",
        "strict_full_sort_mask": True,
        "skip_test_after_training": True,
        "paired_time_factorial": True,
    }
    violations = {
        field: training.get(field)
        for field, expected in required_training.items()
        if training.get(field) != expected
    }
    if violations:
        raise ValueError(f"fresh-training manifest violations: {violations}")
    datasets = payload.get("datasets", {})
    if tuple(datasets) != DATASET_ORDER:
        raise ValueError(
            f"dataset order must be {DATASET_ORDER}, got {tuple(datasets)}"
        )
    for dataset_key, preset in datasets.items():
        if set(preset.get("expected", {})) != set(METRICS):
            raise ValueError(f"expected metrics incomplete for {dataset_key}")
        if "_diff" in str(preset.get("data_name", "")).lower():
            raise ValueError(f"diff data are forbidden: {dataset_key}")
        if int(preset["complex_dff_rank"]) < 0:
            raise ValueError(f"negative Complex DFF rank: {dataset_key}")
        if float(preset["complex_dff_epsilon"]) < 0:
            raise ValueError(f"negative Complex DFF epsilon: {dataset_key}")
    return payload


def alignment_audit(item_path: Path, time_path: Path) -> dict[str, int]:
    users = 0
    interactions = 0
    maximum_time_id = 0
    with item_path.open(encoding="utf-8") as item_handle, time_path.open(
        encoding="utf-8"
    ) as time_handle:
        for line_number, pair in enumerate(
            itertools.zip_longest(item_handle, time_handle), start=1
        ):
            item_line, time_line = pair
            if item_line is None or time_line is None:
                raise ValueError(
                    f"item/time row-count mismatch at line {line_number}: "
                    f"{item_path}, {time_path}"
                )
            item_fields = item_line.strip().split()
            time_fields = time_line.strip().split()
            if len(item_fields) < 2 or len(time_fields) < 2:
                raise ValueError(
                    f"empty item/time sequence at line {line_number}: "
                    f"{item_path}, {time_path}"
                )
            if item_fields[0] != time_fields[0]:
                raise ValueError(
                    f"user-id mismatch at line {line_number}: "
                    f"{item_fields[0]} != {time_fields[0]}"
                )
            item_count = len(item_fields) - 1
            time_count = len(time_fields) - 1
            if item_count != time_count:
                raise ValueError(
                    f"sequence-length mismatch at line {line_number}: "
                    f"{item_count} != {time_count}"
                )
            time_ids = [int(value) for value in time_fields[1:]]
            if any(value < 0 for value in time_ids):
                raise ValueError(f"negative temporal ID at line {line_number}")
            maximum_time_id = max(maximum_time_id, max(time_ids))
            users += 1
            interactions += item_count
    return {
        "users": users,
        "interactions": interactions,
        "maximum_time_id": maximum_time_id,
    }


def data_audit(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for dataset_key, preset in manifest["datasets"].items():
        paths = {
            "item": (REPO_DIR / preset["item_file"]).resolve(),
            "time": (REPO_DIR / preset["time_file"]).resolve(),
        }
        data_name = str(preset["data_name"])
        expected_names = {
            "item": f"{data_name}.txt",
            "time": f"{data_name}_time_time.txt",
        }
        row: dict[str, Any] = {"data_name": data_name}
        for kind, path in paths.items():
            try:
                path.relative_to(REPO_DIR)
            except ValueError as exc:
                raise ValueError(f"data path leaves repository: {path}") from exc
            if path.name != expected_names[kind]:
                raise ValueError(
                    f"{dataset_key} {kind} filename does not match data_name: {path}"
                )
            if not path.is_file():
                raise FileNotFoundError(path)
            row[f"{kind}_file"] = str(path.relative_to(REPO_DIR))
            row[f"{kind}_bytes"] = path.stat().st_size
        row.update(alignment_audit(paths["item"], paths["time"]))
        audit[dataset_key] = row
    return audit


def git_state() -> dict[str, Any]:
    def output(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *arguments],
                cwd=str(REPO_DIR),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    status = output("status", "--porcelain")
    return {
        "commit": output("rev-parse", "HEAD"),
        "branch": output("branch", "--show-current"),
        "working_tree_clean": status == "",
    }


def preflight_payload(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    no_cuda: bool,
) -> dict[str, Any]:
    env = environment_audit(manifest)
    return {
        "schema": "tcf-reproduction-preflight-v1",
        "checked_at_utc": utc_now(),
        "status": "pass",
        "manifest": str(manifest_path.relative_to(REPO_DIR)),
        "environment": env,
        "device": (
            "cpu"
            if no_cuda or not torch.cuda.is_available()
            else f"cuda:{torch.cuda.current_device()}"
        ),
        "data": data_audit(manifest),
        "fresh_training_contract": {
            "load_model": None,
            "warmstart": None,
            "train_scope": "all",
            "all_components_trainable_from_epoch_1": True,
            "checkpoint_use": "same-run selected checkpoint only",
        },
        "repository": git_state(),
    }


@dataclass(frozen=True)
class ReproductionJob:
    dataset_key: str
    preset: dict[str, Any]
    training: dict[str, Any]
    output_dir: Path
    python: Path
    gpu_id: str
    no_cuda: bool

    @property
    def epsilon(self) -> float:
        return float(self.preset["complex_dff_epsilon"])

    @property
    def rank(self) -> int:
        return int(self.preset["complex_dff_rank"])

    @property
    def candidate(self) -> str:
        epsilon = f"{self.epsilon:g}".replace(".", "p")
        return f"eps_{epsilon}_rank_{self.rank}"

    @property
    def train_name(self) -> str:
        return f"TCF_{self.dataset_key}"

    @property
    def run_dir(self) -> Path:
        return self.output_dir / "runs"

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / f"{self.train_name}.pt"

    @property
    def training_log(self) -> Path:
        return self.run_dir / f"{self.train_name}.log"

    @property
    def console(self) -> Path:
        return self.output_dir / "console" / f"{self.train_name}.console.log"

    @property
    def data_dir(self) -> Path:
        return (REPO_DIR / self.preset["item_file"]).resolve().parent

    def command(self) -> list[str]:
        command = [
            str(self.python),
            str(SRC_DIR / "scripts" / "run_complex_dff_main.py"),
            "--model_type", "tcf",
            "--data_name", str(self.preset["data_name"]),
            "--data_dir", str(self.data_dir) + os.sep,
            "--lr", str(self.preset["lr"]),
            "--attention_probs_dropout_prob", str(self.preset["dropout"]),
            "--hidden_dropout_prob", str(self.preset["dropout"]),
            "--num_heads", str(self.preset["num_heads"]),
            "--alpha", str(self.preset["alpha"]),
            "--weight_decay", str(self.preset["weight_decay"]),
            "--time_thresholds", "preset",
            "--time_embedding_scale", str(self.preset["time_embedding_scale"]),
            "--complex_dff_epsilon", str(self.epsilon),
            "--complex_dff_rank", str(self.rank),
            "--complex_dff_train_scope", "all",
            "--batch_size", str(self.training["batch_size"]),
            "--epochs", str(self.training["epochs"]),
            "--patience", str(self.training["patience"]),
            "--num_workers", str(self.training["num_workers"]),
            "--seed", str(self.training["seed"]),
            "--gpu_id", self.gpu_id,
            "--strict_full_sort_mask",
            "--skip_test_after_training",
            "--paired_time_factorial",
            "--train_name", self.train_name,
            "--output_dir", str(self.run_dir) + os.sep,
        ]
        if self.no_cuda:
            command.append("--no_cuda")
        audit_training_command(command)
        return command

    def evaluator_preset(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir) + os.sep,
            "data_name": str(self.preset["data_name"]),
            "lr": float(self.preset["lr"]),
            "dropout": float(self.preset["dropout"]),
            "num_heads": int(self.preset["num_heads"]),
            "alpha": float(self.preset["alpha"]),
            "weight_decay": float(self.preset["weight_decay"]),
            "time_embedding_scale": float(self.preset["time_embedding_scale"]),
            "seed": int(self.training["seed"]),
        }


def audit_training_command(command: list[str]) -> None:
    for option in FORBIDDEN_TRAINING_OPTIONS:
        if option in command:
            raise ValueError(f"forbidden training option: {option}")
    required_options = {
        "--complex_dff_train_scope": "all",
        "--strict_full_sort_mask": None,
        "--skip_test_after_training": None,
        "--paired_time_factorial": None,
    }
    for option, expected_value in required_options.items():
        if option not in command:
            raise ValueError(f"required training option missing: {option}")
        if expected_value is not None:
            index = command.index(option)
            if index + 1 >= len(command) or command[index + 1] != expected_value:
                raise ValueError(f"invalid {option}: expected {expected_value}")


def audit_training_artifact(job: ReproductionJob) -> dict[str, Any]:
    if not job.checkpoint.is_file():
        raise FileNotFoundError(job.checkpoint)
    if not job.training_log.is_file():
        raise FileNotFoundError(job.training_log)
    lines = job.training_log.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    if not lines:
        raise ValueError(f"empty training log: {job.training_log}")
    namespace = lines[0]
    text = "\n".join(lines)
    required_namespace = (
        f"seed={job.training['seed']}",
        "do_eval=False",
        "load_model=None",
        "paired_time_factorial=True",
        "strict_full_sort_mask=True",
        "skip_test_after_training=True",
        "complex_dff_warmstart=None",
        "complex_dff_train_scope='all'",
    )
    missing = [token for token in required_namespace if token not in namespace]
    required_log = (
        "Complex DFF fresh training audit: load_model=None, "
        "warmstart=None, scope=all",
        "Test evaluation skipped",
    )
    missing.extend(token for token in required_log if token not in text)
    separator = "---------------Test Score---------------"
    forbidden_found = []
    if "Complex DFF warm start:" in text:
        forbidden_found.append("Complex DFF warm start:")
    separator_index = text.find(separator)
    if (
        separator_index >= 0
        and "Test evaluation skipped" not in text[:separator_index]
    ):
        forbidden_found.append(separator)
    if missing or forbidden_found:
        raise ValueError(
            f"fresh-training audit failed for {job.dataset_key}: "
            f"missing={missing}, forbidden={forbidden_found}"
        )
    return {
        "dataset": job.dataset_key,
        "checkpoint": str(job.checkpoint),
        "training_log": str(job.training_log),
        "load_model": None,
        "warmstart": None,
        "train_scope": "all",
        "fresh_training_audit_passed": True,
    }


def build_jobs(
    manifest: dict[str, Any],
    dataset_keys: tuple[str, ...],
    args: argparse.Namespace,
) -> list[ReproductionJob]:
    return [
        ReproductionJob(
            dataset_key=key,
            preset=dict(manifest["datasets"][key]),
            training=dict(manifest["training"]),
            output_dir=args.output_dir.resolve(),
            python=Path(sys.executable).resolve(),
            gpu_id=args.gpu_id,
            no_cuda=args.no_cuda,
        )
        for key in dataset_keys
    ]


def queue_rows(jobs: list[ReproductionJob]) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        status = "pending"
        if job.checkpoint.exists() or job.training_log.exists():
            try:
                audit_training_artifact(job)
                status = "complete"
            except (FileNotFoundError, ValueError):
                status = "invalid_partial_artifact"
        rows.append(
            {
                "dataset": job.dataset_key,
                "seed": job.training["seed"],
                "epsilon": job.epsilon,
                "rank": job.rank,
                "status": status,
                "checkpoint": str(job.checkpoint),
                "training_log": str(job.training_log),
                "command": shlex.join(job.command()),
            }
        )
    return rows


def train(job: ReproductionJob, *, dry_run: bool) -> dict[str, Any] | None:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    job.console.parent.mkdir(parents=True, exist_ok=True)
    if job.checkpoint.exists() or job.training_log.exists():
        try:
            audit = audit_training_artifact(job)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"partial or unaudited artifact exists for {job.dataset_key}; "
                f"use a new --output-dir instead of overwriting it"
            ) from exc
        print(f"[train-skip] audited same-run checkpoint: {job.dataset_key}")
        return audit
    command = job.command()
    print(f"[train] {job.dataset_key}: {shlex.join(command)}", flush=True)
    if dry_run:
        return None
    with job.console.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=str(SRC_DIR),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"training failed for {job.dataset_key}; see {job.console}"
        )
    return audit_training_artifact(job)


@dataclass(frozen=True)
class EvaluationJob:
    dataset_key: str
    preset: dict[str, Any]
    epsilon: float
    rank: int
    candidate: str
    checkpoint: Path


def evaluate(
    job: ReproductionJob,
    *,
    eval_batch_size: int,
    device: torch.device,
    tolerance: float,
) -> dict[str, Any]:
    adapter = EvaluationJob(
        dataset_key=job.dataset_key,
        preset=job.evaluator_preset(),
        epsilon=job.epsilon,
        rank=job.rank,
        candidate=job.candidate,
        checkpoint=job.checkpoint,
    )
    rows, transition_stats = evaluator.evaluate_test_grid(
        adapter,
        windows=(int(job.preset["markov_window"]),),
        taus=(float(job.preset["markov_tau"]),),
        lambdas=(float(job.preset["markov_lambda"]),),
        eval_batch_size=eval_batch_size,
        device=device,
    )
    if len(rows) != 1:
        raise RuntimeError(f"expected one frozen evaluation row, got {len(rows)}")
    measured = rows[0]
    result: dict[str, Any] = {
        "dataset": job.dataset_key,
        "seed": int(job.training["seed"]),
        "epsilon": job.epsilon,
        "rank": job.rank,
        "window": int(job.preset["markov_window"]),
        "tau": float(job.preset["markov_tau"]),
        "lambda": float(job.preset["markov_lambda"]),
        "checkpoint": str(job.checkpoint),
        "transition_source_items": transition_stats["source_items"],
        "transition_edges": transition_stats["transitions"],
    }
    metric_passes = []
    for metric in METRICS:
        actual = float(measured[metric])
        expected = float(job.preset["expected"][metric])
        difference = actual - expected
        passed = abs(difference) <= tolerance
        result[metric] = actual
        result[f"expected_{metric}"] = expected
        result[f"difference_{metric}"] = difference
        result[f"within_tolerance_{metric}"] = passed
        metric_passes.append(passed)
    result["absolute_tolerance"] = tolerance
    result["verification_passed"] = all(metric_passes)
    return result


def wearec_result_info(result: dict[str, Any]) -> dict[str, Any]:
    """Return the final metric dictionary in the original WEARec layout."""
    return {
        "Epoch": 0,
        "HR@10": f"{float(result['HR@10']):.8f}",
        "NDCG@10": f"{float(result['NDCG@10']):.8f}",
        "HR@20": f"{float(result['HR@20']):.8f}",
        "NDCG@20": f"{float(result['NDCG@20']):.8f}",
    }


def append_wearec_result_log(
    job: ReproductionJob,
    result: dict[str, Any],
) -> None:
    """Append and print the same final result block used by WEARec."""
    separator = "---------------Test Score---------------"
    existing = job.training_log.read_text(
        encoding="utf-8",
        errors="replace",
    )
    selection_marker = (
        "Test evaluation skipped until checkpoint selection completes."
    )
    trailing = existing.split(selection_marker, 1)[-1]
    already_recorded = separator in trailing

    result_info = wearec_result_info(result)
    messages = (separator, str(result_info), job.train_name, str(result_info))
    rendered = []
    for message in messages:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        rendered.append(f"{timestamp} - {message}")
    if not already_recorded:
        with job.training_log.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(rendered) + "\n")
    for line in rendered:
        print(line, flush=True)


def run_smoke(output_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
    ]
    console = output_dir / "smoke_tests.log"
    console.parent.mkdir(parents=True, exist_ok=True)
    with console.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=str(SRC_DIR),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return {
        "command": shlex.join(command),
        "returncode": process.returncode,
        "passed": process.returncode == 0,
        "log": str(console),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the frozen TCF result, with optional diagnostic modes."
    )
    parser.add_argument(
        "mode",
        choices=("preflight", "smoke", "single", "full"),
        nargs="?",
        default="full",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_ORDER,
        default="LastFM",
        help="dataset used by single mode",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="allow actual training outside the exact pinned environment",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.output_dir is None:
        suffix = (
            f"single_{args.dataset}" if args.mode == "single" else args.mode
        )
        args.output_dir = DEFAULT_OUTPUT_ROOT / suffix
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = preflight_payload(
        manifest_path,
        manifest,
        no_cuda=args.no_cuda,
    )
    write_json(output_dir / "preflight.json", preflight)
    print(
        "[preflight] PASS; exact environment="
        f"{preflight['environment']['exact_versions']}",
        flush=True,
    )
    if args.mode == "preflight":
        return 0
    if args.mode == "smoke":
        smoke = run_smoke(output_dir)
        write_json(
            output_dir / "summary.json",
            {
                "schema": "tcf-reproduction-smoke-v1",
                "completed_at_utc": utc_now(),
                "status": "pass" if smoke["passed"] else "fail",
                "preflight": preflight,
                "tests": smoke,
            },
        )
        print(f"[smoke] {'PASS' if smoke['passed'] else 'FAIL'}", flush=True)
        return 0 if smoke["passed"] else 1

    if (
        not preflight["environment"]["exact_versions"]
        and not args.allow_version_mismatch
        and not args.dry_run
    ):
        raise RuntimeError(
            "result reproduction requires the exact pinned environment; "
            "use the root reproduce.sh wrapper or explicitly pass "
            "--allow-version-mismatch for a non-canonical run"
        )

    dataset_keys = (
        (args.dataset,) if args.mode == "single" else DATASET_ORDER
    )
    jobs = build_jobs(manifest, tuple(dataset_keys), args)
    write_csv(output_dir / "training_queue.csv", queue_rows(jobs))
    protocol = {
        "schema": "tcf-reproduction-protocol-v1",
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "datasets": list(dataset_keys),
        "dry_run": args.dry_run,
        "training": manifest["training"],
        "fresh_training": {
            "load_model": None,
            "warmstart": None,
            "train_scope": "all",
            "commands_audited_before_execution": True,
            "all_trainable_components_optimized_from_epoch_1": True,
            "checkpoint_loaded_only_after_same_run": True,
        },
        "preflight": preflight,
        "commands": {
            job.dataset_key: shlex.join(job.command()) for job in jobs
        },
    }
    write_json(output_dir / "protocol.json", protocol)

    audits = []
    for job in jobs:
        audit = train(job, dry_run=args.dry_run)
        if audit is not None:
            audits.append(audit)
        write_csv(output_dir / "training_queue.csv", queue_rows(jobs))
    if args.dry_run:
        write_json(
            output_dir / "summary.json",
            {
                "schema": "tcf-reproduction-summary-v1",
                "status": "dry_run",
                "completed_at_utc": utc_now(),
                "training_executed": False,
                "fresh_training_command_audit_passed": True,
            },
        )
        print("[dry-run] PASS; no training was executed", flush=True)
        return 0

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.no_cuda
        else "cpu"
    )
    tolerance = float(
        manifest["evaluation"]["verification_absolute_tolerance"]
    )
    results = []
    for job in jobs:
        print(f"[evaluate] {job.dataset_key} on {device}", flush=True)
        result = evaluate(
            job,
            eval_batch_size=args.eval_batch_size,
            device=device,
            tolerance=tolerance,
        )
        append_wearec_result_log(job, result)
        results.append(result)
    write_csv(output_dir / "results.csv", results)
    all_passed = all(row["verification_passed"] for row in results)
    summary = {
        "schema": "tcf-reproduction-summary-v1",
        "status": "pass" if all_passed else "metric_mismatch",
        "completed_at_utc": utc_now(),
        "fresh_training_audits": audits,
        "datasets_completed": len(results),
        "expected_datasets": len(jobs),
        "all_metrics_within_tolerance": all_passed,
        "absolute_tolerance": tolerance,
        "results_csv": str(output_dir / "results.csv"),
    }
    write_json(output_dir / "summary.json", summary)
    print(
        f"[complete] {'PASS' if all_passed else 'METRIC MISMATCH'}; "
        f"summary={output_dir / 'summary.json'}",
        flush=True,
    )
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
