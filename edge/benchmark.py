"""Measure single-image ONNX latency, on-disk size, and process peak RSS."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .deploy import cpu_session


def rss_high_water_mib():
    if sys.platform.startswith("linux"):
        # VmHWM starts fresh at exec; getrusage may inherit the parent's high-water mark.
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
        except OSError:
            pass
        return None
    if sys.platform == "darwin":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    return None


def profile(path, resolution, threads, warmup, repeats):
    start_rss = rss_high_water_mib()
    session = cpu_session(path, threads)
    x = np.random.default_rng(9).normal(size=(1, 3, resolution, resolution)).astype(np.float32)
    inputs = {session.get_inputs()[0].name: x}
    for _ in range(warmup):
        session.run(None, inputs)
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        session.run(None, inputs)
        latencies.append((time.perf_counter_ns() - start) / 1e6)
    peak_rss = rss_high_water_mib()
    return {"storage_bytes": path.stat().st_size, "p50_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)), "peak_rss_mib": peak_rss,
            "peak_rss_growth_mib": peak_rss - start_rss if peak_rss is not None else None,
            "threads": threads, "warmup": warmup, "repeats": repeats}


def isolated_profile(path, resolution, threads=1, warmup=20, repeats=100):
    # A fresh interpreter gives each model a separate process memory high-water mark.
    command = [sys.executable, "-m", "edge.benchmark", "--profile", str(path),
               "--resolution", str(resolution), "--threads", str(threads),
               "--warmup", str(warmup), "--repeats", str(repeats)]
    return json.loads(subprocess.check_output(command, text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--profile", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resolution", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, args.repeats) < 1 or args.warmup < 0:
        parser.error("Threads and repeats must be positive, warmup must be nonnegative.")
    if args.profile is not None:
        print(json.dumps(profile(args.profile, args.resolution, args.threads,
                                 args.warmup, args.repeats)))
        return
    from .train import write_csv, write_json
    from importlib.metadata import version

    rows = []
    for path in sorted(args.root.glob("models/*.onnx")):
        session = cpu_session(path, args.threads)
        spec = json.loads(session.get_modelmeta().custom_metadata_map["edge_model"])
        del session
        row = {"model": path.stem, "resolution": spec["resolution"],
               "test_accuracy": spec.get("test_accuracy")}
        row.update(isolated_profile(path, spec["resolution"], args.threads,
                                    args.warmup, args.repeats))
        rows.append(row)
        print(f"{path.stem:<25} p50 {row['p50_ms']:>7.3f} ms | "
              f"p95 {row['p95_ms']:>7.3f} ms | {row['storage_bytes']:>10,} bytes")
    if not rows:
        parser.error("Export at least one ONNX model before benchmarking.")
    cpu = platform.processor() or platform.machine()
    info = Path("/proc/cpuinfo")
    if info.is_file():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    write_csv(args.root / "results" / "benchmark.csv", rows)
    write_json(args.root / "results" / "benchmark_environment.json",
               {"cpu": cpu, "platform": platform.platform(), "python": platform.python_version(),
                "pytorch": version("torch"), "onnxruntime": version("onnxruntime"),
                "provider": "CPUExecutionProvider", "precision": "FP32", "batch_size": 1,
                "timing_scope": "ONNX session.run with already prepared tensor",
                "memory_scope": "process-wide peak RSS, including interpreter and runtime"})


if __name__ == "__main__":
    main()
