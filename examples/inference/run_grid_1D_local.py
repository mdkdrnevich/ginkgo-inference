#!/usr/bin/env python3
"""Run the trellis 1D grid jobs locally in parallel.

This mirrors the SLURM array job in run-trellis-grid-1D.s by launching
`run_trellis_1D.py <task_id>` for each task id in a range.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trellis 1D grid jobs locally in parallel."
    )
    parser.add_argument("--start", type=int, default=0, help="First task id (inclusive).")
    parser.add_argument("--end", type=int, default=149, help="Last task id (inclusive).")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for each task.",
    )
    parser.add_argument(
        "--script",
        default="../../scripts/run_trellis_1D.py",
        help="Target script to launch for each task id.",
    )
    parser.add_argument(
        "--working-dir",
        default=str(Path(__file__).resolve().parent),
        help="Working directory where the script is executed.",
    )
    parser.add_argument(
        "--log-dir",
        default="local_logs/trellis_1d",
        help="Directory for per-task log files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Optional timeout in seconds per task (0 disables timeout).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop remaining jobs as soon as one task fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    return parser.parse_args()


def run_one_job(args: Tuple[int, str, str, str, str, int]) -> Tuple[int, int, float, str]:
    task_id, python_exe, script_path, working_dir, log_dir, timeout = args
    start = time.time()

    log_path = Path(log_dir) / f"task_{task_id:03d}.log"
    cmd = [python_exe, script_path, str(task_id)]

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[command] {' '.join(cmd)}\n")
        log_file.write(f"[cwd] {working_dir}\n\n")
        log_file.flush()

        try:
            completed = subprocess.run(
                cmd,
                cwd=working_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=None if timeout <= 0 else timeout,
                text=True,
            )
            code = completed.returncode
        except subprocess.TimeoutExpired:
            log_file.write("\n[error] Task timed out.\n")
            code = 124

    duration = time.time() - start
    return task_id, code, duration, str(log_path)


def build_jobs(
    task_ids: Iterable[int],
    python_exe: str,
    script_path: str,
    working_dir: str,
    log_dir: str,
    timeout: int,
) -> list[Tuple[int, str, str, str, str, int]]:
    return [
        (task_id, python_exe, script_path, working_dir, log_dir, timeout)
        for task_id in task_ids
    ]


def main() -> int:
    args = parse_args()

    if args.start > args.end:
        raise ValueError("--start must be <= --end")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    working_dir = Path(args.working_dir).resolve()
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = (working_dir / script_path).resolve()

    if not working_dir.is_dir():
        raise FileNotFoundError(f"Working directory not found: {working_dir}")
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = (working_dir / log_dir).resolve()
    os.makedirs(log_dir, exist_ok=True)

    task_ids = list(range(args.start, args.end + 1))

    print(f"Launching {len(task_ids)} jobs with {args.workers} workers")
    print(f"Script: {script_path}")
    print(f"Working dir: {working_dir}")
    print(f"Logs: {log_dir}")

    if args.dry_run:
        for task_id in task_ids:
            print(f"{args.python} {script_path} {task_id}")
        return 0

    jobs = build_jobs(
        task_ids=task_ids,
        python_exe=args.python,
        script_path=str(script_path),
        working_dir=str(working_dir),
        log_dir=str(log_dir),
        timeout=args.timeout,
    )

    failures = []
    completed = 0

    try:
        with mp.Pool(processes=args.workers) as pool:
            for task_id, code, duration, log_path in pool.imap_unordered(run_one_job, jobs):
                completed += 1
                status = "OK" if code == 0 else f"FAIL({code})"
                print(
                    f"[{completed:03d}/{len(task_ids):03d}] task={task_id:03d} "
                    f"status={status} time={duration:.1f}s log={log_path}"
                )

                if code != 0:
                    failures.append((task_id, code, log_path))
                    if args.fail_fast:
                        pool.terminate()
                        break
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130

    if failures:
        print("\nFailed tasks:")
        for task_id, code, log_path in sorted(failures):
            print(f"- task {task_id}: exit code {code} ({log_path})")
        return 1

    print("\nAll tasks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
