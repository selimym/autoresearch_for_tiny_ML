"""
shinka_evaluate.py — Phase 1 ShinkaEvolve evaluator adapter.

ShinkaEvolve calls: python shinka_evaluate.py --program_path <evolved.py> --results_dir <dir>

The evolved program must expose:
  run_experiment() -> dict with keys: score, mAP50, model_size_mb, cpu_latency_ms

ShinkaEvolve reads metrics.json and correct.json from results_dir after this runs.
"""
import argparse
from typing import Any, Dict, List, Optional, Tuple

from shinka.core import run_shinka_eval

NUM_RUNS = 1  # Each experiment is deterministic


def validate_experiment(result: Any) -> Tuple[bool, Optional[str]]:
    """Validate run_experiment() output."""
    if not isinstance(result, dict):
        return False, f"Expected dict, got {type(result).__name__}"
    required = {"score", "mAP50", "model_size_mb", "cpu_latency_ms"}
    missing = required - set(result.keys())
    if missing:
        return False, f"Missing keys: {missing}"
    if not isinstance(result.get("model_size_mb"), (int, float)) or result["model_size_mb"] <= 0:
        return False, f"model_size_mb must be a positive number, got {result.get('model_size_mb')}"
    return True, None


def aggregate_metrics(results: List[Any], results_dir: str = "") -> Dict[str, Any]:
    """Map run_experiment() output to ShinkaEvolve's expected format."""
    result = results[0]
    return {
        "combined_score": float(result["score"]),
        "mAP50": float(result["mAP50"]),
        "model_size_mb": float(result["model_size_mb"]),
        "cpu_latency_ms": float(result["cpu_latency_ms"]),
        "public": {
            "score": float(result["score"]),
            "mAP50": float(result["mAP50"]),
            "model_size_mb": float(result["model_size_mb"]),
        },
    }


def main(program_path: str, results_dir: str) -> None:
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=NUM_RUNS,
        validate_fn=validate_experiment,
        aggregate_metrics_fn=aggregate_metrics,
        default_metrics_on_error={
            "combined_score": 0.0,
            "mAP50": 0.0,
            "model_size_mb": 999.0,
            "cpu_latency_ms": 99999.0,
        },
    )
    if error_msg:
        print(f"[shinka_evaluate] Error: {error_msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial_compress.py")
    parser.add_argument("--results_dir", type=str, default="results/phase1")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
