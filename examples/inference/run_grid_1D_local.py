#!/usr/bin/env python3
"""Run the trellis 1D grid jobs locally in parallel.

This mirrors the SLURM array job in run-trellis-grid-1D.s by launching
`run_trellis_1D.py <task_id>` for each task id in a range.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 1D grid jobs locally in parallel."
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="trellis",
        choices=["trellis", "ginkgo"],
        help="Framework to use (trellis or ginkgo).",
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
        default="",
        help="Target script to launch for each task id (auto-detected if not provided).",
    )
    parser.add_argument(
        "--working-dir",
        default=str(Path(__file__).resolve().parent),
        help="Working directory where the script is executed.",
    )
    parser.add_argument(
        "--log-dir",
        default="",
        help="Directory for per-task log files (auto-detected if not provided).",
    )
    parser.add_argument(
        "--config-file",
        default="",
        help=(
            "Optional config filename with extra CLI arguments. "
            "Whitespace-separated args are read from this file and appended "
            "to each launched job command."
        ),
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
    args = parser.parse_args()
    args.script = args.script or f"../../scripts/run_{args.framework}_grid_1D.py"
    args.log_dir = args.log_dir or f"logs_{args.framework}"
    args.config_file = args.config_file or f"{args.framework}_1D.config"
    return args


def load_config_args(config_file: Path) -> list[str]:
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    raw_text = config_file.read_text(encoding="utf-8")
    lines = [line.split("#", 1)[0].strip() for line in raw_text.splitlines()]
    payload = "\n".join(line for line in lines if line)
    if not payload:
        return []
    return shlex.split(payload)


def run_one_job(
    args: Tuple[int, str, str, str, str, int, tuple[str, ...]]
) -> Tuple[int, int, float, str]:
    task_id, python_exe, script_path, working_dir, log_dir, timeout, extra_args = args
    start = time.time()

    log_path = Path(log_dir) / f"task_{task_id:03d}.log"
    cmd = [python_exe, script_path, str(task_id), *extra_args]

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
    extra_args: Sequence[str],
) -> list[Tuple[int, str, str, str, str, int, tuple[str, ...]]]:
    frozen_extra_args = tuple(extra_args)
    return [
        (
            task_id,
            python_exe,
            script_path,
            working_dir,
            log_dir,
            timeout,
            frozen_extra_args,
        )
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

    extra_args: list[str] = []
    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.is_absolute():
            config_path = (working_dir / config_path).resolve()
        extra_args = load_config_args(config_path)
    
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = (working_dir / log_dir).resolve()
    os.makedirs(log_dir, exist_ok=True)

    task_ids = list(range(args.start, args.end + 1))

    print(f"Launching {len(task_ids)} jobs with {args.workers} workers")
    print(f"Script: {script_path}")
    print(f"Working dir: {working_dir}")
    print(f"Logs: {log_dir}")
    if args.config_file:
        print(f"Config file: {config_path}")
        print(f"Extra args: {extra_args}")

    if args.dry_run:
        for task_id in task_ids:
            cmd = [args.python, str(script_path), str(task_id), *extra_args]
            print(" ".join(shlex.quote(part) for part in cmd))
        return 0

    jobs = build_jobs(
        task_ids=task_ids,
        python_exe=args.python,
        script_path=str(script_path),
        working_dir=str(working_dir),
        log_dir=str(log_dir),
        timeout=args.timeout,
        extra_args=extra_args,
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
