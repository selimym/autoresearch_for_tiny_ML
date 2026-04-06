"""
benchmark_pi.py — Pi 3B+ latency wrapper. Runs natively on ARM Pi.

Usage (on Pi):
  python benchmark_pi.py --model checkpoints/phase3_champion.onnx --runs 100

Requires: pip install onnxruntime (ARM build)
"""
import argparse, time, os, platform
import numpy as np

IMG_SIZE = 320


def measure_latency(onnx_path: str, runs: int) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    # Warmup
    for _ in range(10):
        sess.run(None, {input_name: dummy})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    median_ms = float(np.median(times))
    return {
        "device": f"{platform.node()} ({platform.machine()})",
        "model": onnx_path,
        "median_latency_ms": median_ms,
        "fps": round(1000 / median_ms, 1),
        "model_size_mb": round(os.path.getsize(onnx_path) / (1024 * 1024), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    results = measure_latency(args.model, args.runs)
    print("---")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("---")


if __name__ == "__main__":
    main()
