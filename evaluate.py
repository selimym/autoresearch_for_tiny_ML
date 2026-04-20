"""
evaluate.py — Phase 2-3 CLI evaluator. Fixed, never modified by the agent.

Usage (agent):
  python evaluate.py --checkpoint path/to/model.onnx

Prints --- block (agent greps this):
  ---
  mAP50:          0.612300
  model_size_mb:  3.84
  cpu_latency_ms: 142.7
  params_M:       1.97
  score:          0.159453
  ---

OpenEvolve-compatible function:
  from evaluate import evaluate
  metrics = evaluate("path/to/model.onnx")  -> dict[str, float]

Also writes eval_result.json alongside the checkpoint.
"""
import argparse
import json
from pathlib import Path

from evaluate_core import evaluate_model


def evaluate(checkpoint_path: str) -> dict:
    """OpenEvolve-compatible entry point."""
    return evaluate_model(checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a .onnx checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .onnx checkpoint")
    args = parser.parse_args()

    metrics = evaluate_model(args.checkpoint)

    # Print --- block (agent greps this)
    print("---")
    print(f"mAP50:          {metrics['mAP50']:.6f}")
    print(f"mAP:            {metrics['mAP']:.6f}")
    print(f"AP_small:       {metrics['AP_small']:.6f}")
    print(f"model_size_mb:  {metrics['model_size_mb']:.2f}")
    print(f"cpu_latency_ms: {metrics['cpu_latency_ms']:.1f}")
    print(f"params_M:       {metrics['params_M']:.2f}")
    print(f"score:          {metrics['score']:.6f}")
    print("---")

    # Write eval_result.json alongside checkpoint
    result_path = Path(args.checkpoint).parent / "eval_result.json"
    result_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
