"""Benchmark ONNX Runtime CPU inference latency, throughput, and memory footprint."""

import argparse
import json
import time
import platform
from pathlib import Path
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU ONNX Runtime latency.")
    parser.add_argument("--model", type=str, default="artifacts/models/model.onnx")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-json", type=str, default="artifacts/metrics/latency.json")
    args = parser.parse_args()

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    model_p = Path(args.model)
    if not model_p.exists():
        # Export dummy ONNX first if not yet created
        from scripts.export_onnx import main as export_fn
        export_fn()

    print(f"=== Running CPU Latency Benchmark ({args.iterations} Iterations) ===")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_p), sess_options=opts, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 384, 384).astype(np.float32)

    # Warm-up
    for _ in range(args.warmup):
        session.run(None, {inp_name: dummy_input})

    # Timed runs
    timings = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        session.run(None, {inp_name: dummy_input})
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0) # ms

    timings = np.array(timings)
    median_lat = float(np.median(timings))
    p95_lat = float(np.percentile(timings, 95))
    p99_lat = float(np.percentile(timings, 99))
    fps = float(1000.0 / np.mean(timings))
    model_size_mb = float(model_p.stat().st_size / (1024 * 1024))

    results = {
        "system_platform": platform.platform(),
        "processor": platform.processor() or "CPU",
        "onnxruntime_version": ort.__version__,
        "threads": 4,
        "batch_size": 1,
        "image_size": [384, 384],
        "iterations": args.iterations,
        "latency_median_ms": round(median_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "latency_p99_ms": round(p99_lat, 2),
        "throughput_fps": round(fps, 2),
        "model_file_size_mb": round(model_size_mb, 2)
    }

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Median Latency: {median_lat:.2f} ms")
    print(f"p95 Latency:    {p95_lat:.2f} ms")
    print(f"p99 Latency:    {p99_lat:.2f} ms")
    print(f"Throughput:     {fps:.2f} images/sec")
    print(f"Model Size:     {model_size_mb:.2f} MB")
    print(f"Saved benchmark results to: {out_p}")


if __name__ == "__main__":
    main()
