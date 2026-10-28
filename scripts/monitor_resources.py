#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor CPU, memory, disk and a matched process tree while a "
            "MINISO replay is running. Writes a small CSV for capacity review."
        )
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--pid", type=int, default=None)
    target.add_argument(
        "--match",
        default="replay_prelaunch_features.py",
        help="Substring used to find root processes (default: replay driver).",
    )
    parser.add_argument("--interval-seconds", type=positive_float, default=5.0)
    parser.add_argument("--duration-minutes", type=positive_float, default=None)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--disk-path", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path; defaults to logs/resource_monitor_<timestamp>.csv.",
    )
    parser.add_argument("--alert-free-gb", type=nonnegative_float, default=15.0)
    parser.add_argument(
        "--alert-available-memory-percent",
        type=nonnegative_float,
        default=15.0,
    )
    parser.add_argument(
        "--alert-load-per-cpu",
        type=nonnegative_float,
        default=1.0,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        number = float(raw.strip().split()[0])
        values[key] = number * 1024.0
    return values


def _read_cpu_ticks(path: Path = Path("/proc/stat")) -> tuple[int, int]:
    fields = path.read_text(encoding="utf-8").splitlines()[0].split()
    if not fields or fields[0] != "cpu":
        raise RuntimeError("/proc/stat does not contain aggregate CPU counters")
    ticks = [int(value) for value in fields[1:]]
    idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
    return sum(ticks), idle


def _process_rows() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "ps",
            "-eo",
            "pid=,ppid=,pcpu=,pmem=,rss=,etime=,args=",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "cpu_percent": float(parts[2]),
                "memory_percent": float(parts[3]),
                "rss_bytes": int(parts[4]) * 1024,
                "elapsed": parts[5],
                "command": parts[6],
            }
        )
    return rows


def _matched_process_tree(
    rows: list[dict[str, Any]], *, pid: int | None, match: str | None
) -> list[dict[str, Any]]:
    own_pid = os.getpid()
    if pid is not None:
        roots = {pid}
    else:
        token = str(match or "")
        roots = {
            int(row["pid"])
            for row in rows
            if token in str(row["command"]) and int(row["pid"]) != own_pid
        }
    selected = set(roots)
    changed = True
    while changed:
        changed = False
        for row in rows:
            child = int(row["pid"])
            if int(row["ppid"]) in selected and child not in selected:
                selected.add(child)
                changed = True
    return [row for row in rows if int(row["pid"]) in selected]


def _gib(value: float) -> float:
    return value / 1024**3


def _snapshot(
    *,
    args: argparse.Namespace,
    previous_cpu: tuple[int, int] | None,
) -> tuple[dict[str, Any], tuple[int, int]]:
    cpu = _read_cpu_ticks()
    if previous_cpu is None:
        system_cpu = None
    else:
        total_delta = cpu[0] - previous_cpu[0]
        idle_delta = cpu[1] - previous_cpu[1]
        system_cpu = (
            100.0 * (1.0 - idle_delta / total_delta) if total_delta > 0 else 0.0
        )
    memory = _read_meminfo()
    total_memory = memory.get("MemTotal", 0.0)
    available_memory = memory.get("MemAvailable", memory.get("MemFree", 0.0))
    swap_total = memory.get("SwapTotal", 0.0)
    swap_free = memory.get("SwapFree", 0.0)
    disk = shutil.disk_usage(args.disk_path)
    process_rows = _matched_process_tree(
        _process_rows(), pid=args.pid, match=args.match
    )
    process_cpu = sum(float(row["cpu_percent"]) for row in process_rows)
    process_rss = sum(int(row["rss_bytes"]) for row in process_rows)
    load_1m, load_5m, load_15m = os.getloadavg()
    cpu_count = os.cpu_count() or 1
    available_pct = (
        100.0 * available_memory / total_memory if total_memory else 0.0
    )
    alerts: list[str] = []
    if _gib(disk.free) < args.alert_free_gb:
        alerts.append("disk_low")
    if available_pct < args.alert_available_memory_percent:
        alerts.append("memory_low")
    if load_1m / cpu_count > args.alert_load_per_cpu:
        alerts.append("cpu_load_high")
    if not process_rows:
        alerts.append("process_not_found")
    return (
        {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "alert" if alerts else "ok",
            "alerts": "|".join(alerts),
            "system_cpu_percent": "" if system_cpu is None else round(system_cpu, 2),
            "cpu_count": cpu_count,
            "load_1m": round(load_1m, 3),
            "load_5m": round(load_5m, 3),
            "load_15m": round(load_15m, 3),
            "load_per_cpu": round(load_1m / cpu_count, 3),
            "memory_total_gb": round(_gib(total_memory), 3),
            "memory_available_gb": round(_gib(available_memory), 3),
            "memory_available_percent": round(available_pct, 2),
            "swap_used_gb": round(_gib(max(swap_total - swap_free, 0.0)), 3),
            "disk_total_gb": round(_gib(disk.total), 3),
            "disk_used_gb": round(_gib(disk.used), 3),
            "disk_free_gb": round(_gib(disk.free), 3),
            "matched_processes": len(process_rows),
            "process_cpu_percent_sum": round(process_cpu, 2),
            "process_rss_gb_sum": round(_gib(process_rss), 3),
            "root_pids": "|".join(
                str(row["pid"])
                for row in process_rows
                if args.pid == row["pid"]
                or (args.pid is None and str(args.match) in str(row["command"]))
            ),
        },
        cpu,
    )


def main() -> int:
    args = parse_args()
    output = args.output or (
        PROJECT_ROOT
        / "logs"
        / f"resource_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    disk_path = args.disk_path
    if not disk_path.is_absolute():
        disk_path = PROJECT_ROOT / disk_path
    args.disk_path = disk_path
    plan = {
        "pid": args.pid,
        "match": None if args.pid is not None else args.match,
        "interval_seconds": args.interval_seconds,
        "duration_minutes": args.duration_minutes,
        "disk_path": str(args.disk_path),
        "output": str(output),
    }
    if args.dry_run:
        print(plan)
        return 0
    if not Path("/proc/meminfo").is_file() or not Path("/proc/stat").is_file():
        print("ERROR: monitor_resources.py currently requires Linux /proc", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous_cpu: tuple[int, int] | None = None
    wrote_header = output.is_file() and output.stat().st_size > 0
    print(f"resource monitor output: {output}")
    try:
        while True:
            if previous_cpu is None:
                previous_cpu = _read_cpu_ticks()
                time.sleep(min(args.interval_seconds, 0.25))
            row, previous_cpu = _snapshot(args=args, previous_cpu=previous_cpu)
            with output.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not wrote_header:
                    writer.writeheader()
                    wrote_header = True
                writer.writerow(row)
            print(
                f"{row['timestamp']} status={row['status']} alerts={row['alerts'] or '-'} "
                f"cpu={row['system_cpu_percent']}% load/cpu={row['load_per_cpu']} "
                f"mem_available={row['memory_available_gb']}GiB "
                f"disk_free={row['disk_free_gb']}GiB processes={row['matched_processes']} "
                f"process_cpu_sum={row['process_cpu_percent_sum']}% "
                f"process_rss={row['process_rss_gb_sum']}GiB",
                flush=True,
            )
            if args.once:
                break
            if (
                args.duration_minutes is not None
                and time.monotonic() - started >= args.duration_minutes * 60.0
            ):
                break
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("resource monitor stopped; CSV is complete through the last sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
