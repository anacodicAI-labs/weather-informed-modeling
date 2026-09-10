#!/usr/bin/env python3
"""Run the ERA5 spatial-resolution experiment locally or on one cluster node.

Each task builds one (region, resolution, weighting-scheme) weather series and
then runs the fixed chronological ML evaluation. Network staging is deliberately
excluded. Re-run with different ``--workers`` values to collect scaling data.

Examples
--------
python run_spatial_resolution_ladder.py --codes dk --resolutions 0.25 0.5 1 2
python run_spatial_resolution_ladder.py --workers 8 --codes dk ie nl
python run_spatial_resolution_ladder.py --workers 8 --skip-evaluation
"""

from __future__ import annotations

import argparse
import fcntl
import os
import socket
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import build_weighted_weather_local as weather_builder
import evaluate_spatial_weather as evaluator
from country_registry import EUROPE_CODES


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "post_covid_spatial_resolution"
ALL_CODES = EUROPE_CODES + ("tx",)
_THREAD_NETCDF_LOCK = threading.Lock()


def build_weather(task: dict):
    return weather_builder.run(
        code=task["code"],
        resolution=task["resolution"],
        scheme=task["scheme"],
        start=task["start"],
        end=task["end"],
        era5_dir=Path(task["era5_dir"]),
        capacity_dir=Path(task["capacity_dir"]),
        out_dir=Path(task["weather_out_dir"]),
        resolution_cache_dir=Path(task["resolution_cache_dir"]),
        use_resolution_cache=task["use_resolution_cache"],
    )


def execute_task(task: dict) -> dict:
    task_started = time.perf_counter()
    try:
        if task.get("backend") == "thread":
            # netCDF4/HDF5 reads can invalidate file handles when called from
            # several Python threads. Keep only this short I/O/aggregation
            # stage serial; independent model evaluations still overlap.
            with _THREAD_NETCDF_LOCK:
                weather_path, build_metrics = build_weather(task)
        else:
            weather_path, build_metrics = build_weather(task)
        model_rows = [] if task["skip_evaluation"] else evaluator.evaluate(
            task["code"], weather_path, task["split"], Path(task["data_dir"])
        )
        for row in model_rows:
            row.update(
                resolution_deg=task["resolution"],
                scheme=task["scheme"],
                capacity_layout=build_metrics["capacity_layout"],
                capacity_years=build_metrics["capacity_years"],
                capacity_dir=build_metrics["capacity_dir"],
            )
        return {
            "status": "ok",
            "task": task,
            "build_metrics": build_metrics,
            "model_rows": model_rows,
            "task_wall_s": time.perf_counter() - task_started,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "task": task,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "task_wall_s": time.perf_counter() - task_started,
        }


def append_csv(
    path: Path,
    frame: pd.DataFrame,
    latest_keys: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Array jobs can finish simultaneously. Lock the complete read/append/write
    # transaction so one task cannot erase another task's results.
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            old = pd.read_csv(path)
            frame = pd.concat([old, frame], ignore_index=True, sort=False)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
        if latest_keys:
            missing = sorted(set(latest_keys).difference(frame.columns))
            if missing:
                raise KeyError(
                    f"Cannot canonicalize {path.name}; missing key columns {missing}"
                )
            latest = (
                frame.sort_values("run_id", kind="stable")
                .drop_duplicates(latest_keys, keep="last")
                .reset_index(drop=True)
            )
            latest_path = path.with_name(f"{path.stem}_latest{path.suffix}")
            latest_tmp = latest_path.with_name(
                f".{latest_path.name}.{os.getpid()}.tmp"
            )
            latest.to_csv(latest_tmp, index=False)
            os.replace(latest_tmp, latest_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codes", nargs="+", choices=ALL_CODES, default=list(EUROPE_CODES)
    )
    parser.add_argument("--resolutions", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=("capacity", "uniform"),
        default=["capacity", "uniform"],
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=("process", "thread"),
        default="process",
        help=(
            "process is the normal CPU/HPC backend; thread is a fallback for "
            "restricted environments that prohibit process semaphores"
        ),
    )
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--era5-dir", default=str(weather_builder.DEFAULT_ERA5_DIR))
    parser.add_argument("--capacity-dir", default=str(weather_builder.DEFAULT_CAPACITY_DIR))
    parser.add_argument("--weather-out-dir", default=str(weather_builder.DEFAULT_OUT_DIR))
    parser.add_argument(
        "--resolution-cache-dir",
        default=str(weather_builder.DEFAULT_RESOLUTION_CACHE_DIR),
    )
    parser.add_argument("--no-resolution-cache", action="store_true")
    parser.add_argument("--data-dir", default=str(evaluator.DATA_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if not 0 < args.split < 1:
        raise ValueError("--split must be between 0 and 1")
    run_id = f"{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    common = {
        "start": args.start,
        "end": args.end,
        "split": args.split,
        "era5_dir": str(Path(args.era5_dir).expanduser().resolve()),
        "capacity_dir": str(Path(args.capacity_dir).expanduser().resolve()),
        "weather_out_dir": str(Path(args.weather_out_dir).expanduser().resolve()),
        "resolution_cache_dir": str(
            Path(args.resolution_cache_dir).expanduser().resolve()
        ),
        "use_resolution_cache": not args.no_resolution_cache,
        "backend": args.backend,
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "skip_evaluation": args.skip_evaluation,
    }
    tasks = [
        {**common, "code": code, "resolution": resolution, "scheme": scheme}
        for code in args.codes
        for scheme in args.schemes
        for resolution in args.resolutions
    ]
    print(
        f"Run {run_id}: {len(tasks)} tasks, workers={args.workers}, "
        f"host={socket.gethostname()}",
        flush=True,
    )
    batch_started = time.perf_counter()
    results = []
    if args.workers == 1:
        for index, task in enumerate(tasks, 1):
            print(
                f"[{index}/{len(tasks)}] {task['code']} {task['scheme']} "
                f"{task['resolution']:g}°",
                flush=True,
            )
            results.append(execute_task(task))
    else:
        executor_class = (
            ProcessPoolExecutor if args.backend == "process" else ThreadPoolExecutor
        )
        with executor_class(max_workers=args.workers) as pool:
            future_map = {pool.submit(execute_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_map), 1):
                result = future.result()
                task = result["task"]
                print(
                    f"[{index}/{len(tasks)}] {result['status']} {task['code']} "
                    f"{task['scheme']} {task['resolution']:g}°",
                    flush=True,
                )
                results.append(result)
    batch_wall_s = time.perf_counter() - batch_started

    build_rows, capacity_audit_rows, model_rows, failure_rows = [], [], [], []
    for result in results:
        task = result["task"]
        if result["status"] == "ok":
            build_metrics = dict(result["build_metrics"])
            annual_audit = build_metrics.pop("capacity_audit", [])
            build_rows.append(
                {
                    **build_metrics,
                    "run_id": run_id,
                    "workers": args.workers,
                    "task_wall_s": result["task_wall_s"],
                }
            )
            for row in annual_audit:
                capacity_audit_rows.append(
                    {
                        **row,
                        "run_id": run_id,
                        "code": task["code"],
                        "scheme": task["scheme"],
                        "resolution_deg": task["resolution"],
                    }
                )
            for row in result["model_rows"]:
                model_rows.append(
                    {**row, "run_id": run_id, "workers": args.workers}
                )
        else:
            failure_rows.append(
                {
                    "run_id": run_id,
                    "code": task["code"],
                    "scheme": task["scheme"],
                    "resolution_deg": task["resolution"],
                    "error": result["error"],
                    "traceback": result["traceback"],
                }
            )
            print(result["traceback"], flush=True)

    results_dir = Path(args.results_dir).expanduser().resolve()
    if build_rows:
        append_csv(
            results_dir / "weather_build_metrics.csv",
            pd.DataFrame(build_rows),
            latest_keys=["code", "scheme", "resolution_deg"],
        )
    if capacity_audit_rows:
        append_csv(
            results_dir / "capacity_mapping_audit.csv",
            pd.DataFrame(capacity_audit_rows),
            latest_keys=["code", "scheme", "resolution_deg", "year"],
        )
    if model_rows:
        append_csv(
            results_dir / "model_results.csv",
            pd.DataFrame(model_rows),
            latest_keys=["code", "scheme", "resolution_deg", "task", "model"],
        )
    if failure_rows:
        append_csv(results_dir / "failures.csv", pd.DataFrame(failure_rows))

    task_wall_sum = sum(result["task_wall_s"] for result in results)
    point_hours = sum(row["point_hours"] for row in build_rows)
    scaling_row = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                "host": socket.gethostname(),
                "workers": args.workers,
                "backend": args.backend,
                "tasks": len(tasks),
                "successful_tasks": len(build_rows),
                "failed_tasks": len(failure_rows),
                "batch_wall_s": batch_wall_s,
                "sum_task_wall_s": task_wall_sum,
                "worker_utilization": task_wall_sum / (args.workers * batch_wall_s),
                "total_point_hours": point_hours,
                "aggregate_point_hours_per_s": point_hours / batch_wall_s,
                "codes": ",".join(args.codes),
                "resolutions": ",".join(map(str, args.resolutions)),
                "schemes": ",".join(args.schemes),
                "start": args.start,
                "end": args.end,
                "evaluation_included": not args.skip_evaluation,
                "method_version": "padded_cache_v2",
                "resolution_cache_enabled": not args.no_resolution_cache,
            }
        ]
    )
    append_csv(results_dir / "scaling_runs.csv", scaling_row)
    print(
        f"Finished in {batch_wall_s:.1f}s: {len(build_rows)} succeeded, "
        f"{len(failure_rows)} failed. Results: {results_dir}",
        flush=True,
    )
    if failure_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
