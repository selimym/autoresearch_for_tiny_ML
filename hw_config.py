"""
hw_config.py — Hardware-aware configuration advisor for autoresearch-tinydet.

Reads system RAM, VRAM, and CPU count, then recommends safe values for
batch_size, num_workers, and pin_memory given the project's memory profile
(COCO annotation JSONs dominate RAM; small model fits easily in VRAM).

Usage:
    uv run python hw_config.py              # print report + recommended config
    uv run python hw_config.py --export     # also write .env.local with the values

    from hw_config import recommend_config  # import in other scripts
    cfg = recommend_config()
    loader = make_dataloader("train", cfg["batch_size"], 320,
                             num_workers=cfg["num_workers"],
                             pin_memory=cfg["pin_memory"])
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Project-specific memory constants (measured / estimated for this codebase)
# ---------------------------------------------------------------------------

# pycocotools loads the full JSON into Python dicts — empirical WSL2 footprint.
_COCO_TRAIN_JSON_GB = 2.0   # instances_train2017.json in RAM after COCO()
_COCO_VAL_JSON_GB   = 0.5   # instances_val2017.json in RAM after COCO()

# Rough model + optimizer state (frozen backbone, float32 AdamW).
_MODEL_OVERHEAD_GB  = 0.5

# OS + Python interpreter + libraries baseline.
_OS_OVERHEAD_GB     = 2.0

# Activation + gradient memory per image at 320×320, float32, during training.
# Measured empirically for TinyDetector (frozen backbone → only neck+head grads).
_ACTIVATION_PER_IMAGE_MB = 60.0   # conservative upper bound

# GPU: input + feature maps + activations per image at 320×320 during training.
# Forward + backward peaks; frozen backbone reduces this significantly.
_GPU_MB_PER_IMAGE = 120.0  # float32, frozen backbone, empirical estimate


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    total_ram_gb:   float
    free_ram_gb:    float
    vram_gb:        float
    free_vram_gb:   float
    cpu_count:      int
    has_cuda:       bool
    gpu_name:       str
    is_wsl:         bool


def detect_hardware() -> HardwareInfo:
    import psutil
    vm = psutil.virtual_memory()
    total_ram = vm.total / 1024**3
    free_ram  = vm.available / 1024**3

    has_cuda    = False
    vram_gb     = 0.0
    free_vram   = 0.0
    gpu_name    = "none"

    try:
        import torch
        if torch.cuda.is_available():
            has_cuda  = True
            props     = torch.cuda.get_device_properties(0)
            vram_gb   = props.total_memory / 1024**3
            free_vram = (props.total_memory - torch.cuda.memory_reserved(0)) / 1024**3
            gpu_name  = props.name
    except ImportError:
        pass

    is_wsl = os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop") or \
             "microsoft" in open("/proc/version").read().lower() if os.path.exists("/proc/version") else False

    return HardwareInfo(
        total_ram_gb = total_ram,
        free_ram_gb  = free_ram,
        vram_gb      = vram_gb,
        free_vram_gb = free_vram,
        cpu_count    = multiprocessing.cpu_count(),
        has_cuda     = has_cuda,
        gpu_name     = gpu_name,
        is_wsl       = is_wsl,
    )


# ---------------------------------------------------------------------------
# Config recommendation
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    batch_size:          int
    num_workers:         int
    pin_memory:          bool
    persistent_workers:  bool
    device:              str
    notes:               list[str]


def recommend_config(hw: HardwareInfo | None = None) -> TrainingConfig:
    """Return recommended DataLoader and training settings for the current hardware."""
    if hw is None:
        hw = detect_hardware()

    notes: list[str] = []

    # ------------------------------------------------------------------
    # batch_size
    # ------------------------------------------------------------------
    if hw.has_cuda:
        # GPU budget: leave 1 GB headroom for CUDA kernels / ONNX runtime.
        usable_vram_mb = (hw.vram_gb - 1.0) * 1024
        gpu_batch = max(1, int(usable_vram_mb / _GPU_MB_PER_IMAGE))
        # Round down to nearest power of 2 for efficiency
        gpu_batch = _prev_power_of_two(gpu_batch)
        batch_size = min(gpu_batch, 32)   # cap: large batches hurt generalisation
        device = "cuda"
        notes.append(
            f"GPU batch_size={batch_size} "
            f"({_GPU_MB_PER_IMAGE * batch_size:.0f} MB / {hw.vram_gb:.1f} GB VRAM)"
        )
    else:
        # CPU: keep activations under 25% of free RAM
        cpu_ram_budget_mb = hw.free_ram_gb * 1024 * 0.25
        cpu_batch = max(1, int(cpu_ram_budget_mb / _ACTIVATION_PER_IMAGE_MB))
        batch_size = min(_prev_power_of_two(cpu_batch), 8)
        device = "cpu"
        notes.append(f"CPU-only: batch_size capped at {batch_size}")

    # ------------------------------------------------------------------
    # num_workers
    # ------------------------------------------------------------------
    # Each DataLoader worker forks the process, which copies the COCO
    # annotation JSON into its own address space.  On WSL2 the kernel
    # uses copy-on-write but Python dict/list structures defeat CoW, so
    # every worker effectively costs ~_COCO_TRAIN_JSON_GB of real RAM.
    #
    # Budget = currently free RAM minus what training will consume before
    # any worker starts (train JSON + model).  Val JSON is loaded later
    # (at eval time), after training frees its objects.
    training_baseline_gb = _COCO_TRAIN_JSON_GB + _MODEL_OVERHEAD_GB
    budget_for_workers   = hw.free_ram_gb - training_baseline_gb
    # Keep a 20% headroom on whatever remains — WSL2 balloon driver can
    # silently consume memory that looks "free" in psutil.
    budget_for_workers  *= 0.8
    max_by_ram  = max(0, int(budget_for_workers / _COCO_TRAIN_JSON_GB))
    max_by_cpu  = max(0, hw.cpu_count // 2)
    num_workers = min(max_by_ram, max_by_cpu)

    if hw.is_wsl:
        # WSL2 memory reporting is unreliable and swap is slow; be conservative.
        num_workers = min(num_workers, 1)
        if num_workers > 0:
            notes.append("WSL2 detected: capping num_workers=1 (balloon driver risk)")

    if num_workers == 0:
        projected_use = training_baseline_gb + _OS_OVERHEAD_GB
        notes.append(
            f"num_workers=0 — only {hw.free_ram_gb:.1f} GB free; "
            f"training baseline alone needs ~{training_baseline_gb:.1f} GB "
            f"({budget_for_workers:.1f} GB left for workers, need "
            f"{_COCO_TRAIN_JSON_GB:.0f} GB each)"
        )
    else:
        projected = hw.free_ram_gb - training_baseline_gb - num_workers * _COCO_TRAIN_JSON_GB
        notes.append(
            f"num_workers={num_workers} "
            f"(~{projected:.1f} GB RAM headroom after training + workers)"
        )

    # ------------------------------------------------------------------
    # pin_memory
    # ------------------------------------------------------------------
    # pin_memory is only useful with real CUDA DMA — it wastes memory in
    # WSL2 (no DMA path) and on CPU-only systems.
    pin_memory = hw.has_cuda and not hw.is_wsl
    if hw.has_cuda and hw.is_wsl:
        notes.append("pin_memory=False: WSL2 has no real CUDA DMA path")

    # ------------------------------------------------------------------
    # persistent_workers
    # ------------------------------------------------------------------
    # Keeps worker processes alive between epochs — avoids re-forking and
    # re-loading the COCO JSON each epoch.  Only meaningful when workers > 0.
    persistent_workers = num_workers > 0
    if persistent_workers:
        notes.append("persistent_workers=True: workers stay alive between epochs (no re-fork cost)")

    return TrainingConfig(
        batch_size         = batch_size,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        persistent_workers = persistent_workers,
        device             = device,
        notes              = notes,
    )


def _prev_power_of_two(n: int) -> int:
    """Return the largest power of two ≤ n."""
    if n <= 0:
        return 1
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(hw: HardwareInfo, cfg: TrainingConfig) -> None:
    W = 62
    print("=" * W)
    print("  Hardware")
    print("=" * W)
    print(f"  RAM total  : {hw.total_ram_gb:.1f} GB   free: {hw.free_ram_gb:.1f} GB")
    if hw.has_cuda:
        print(f"  GPU        : {hw.gpu_name}")
        print(f"  VRAM       : {hw.vram_gb:.1f} GB   free: {hw.free_vram_gb:.1f} GB")
    else:
        print("  GPU        : none (CPU-only)")
    print(f"  CPU cores  : {hw.cpu_count}")
    print(f"  WSL2       : {'yes' if hw.is_wsl else 'no'}")

    print()
    print("=" * W)
    print("  Recommended config")
    print("=" * W)
    print(f"  device             = {cfg.device}")
    print(f"  batch_size         = {cfg.batch_size}")
    print(f"  num_workers        = {cfg.num_workers}")
    print(f"  pin_memory         = {cfg.pin_memory}")
    print(f"  persistent_workers = {cfg.persistent_workers}")

    print()
    print("=" * W)
    print("  Reasoning")
    print("=" * W)
    for note in cfg.notes:
        print(f"  • {note}")

    print()
    print("=" * W)
    print("  DataLoader snippet")
    print("=" * W)
    print(f"  make_dataloader(")
    print(f'      split="train", batch_size={cfg.batch_size}, img_size=320,')
    print(f"      num_workers={cfg.num_workers},")
    print(f"      pin_memory={cfg.pin_memory},")
    print(f"      persistent_workers={cfg.persistent_workers},")
    print(f"  )")
    print("=" * W)

    # Multiprocessing / async advisory
    if cfg.num_workers == 0:
        print()
        print("  Parallelism note")
        print("-" * W)
        print("  num_workers=0 means data loading is synchronous (no speedup")
        print("  from prefetching). To enable workers safely, free up RAM or")
        print("  reduce the number of concurrent annotation loads. Options:")
        print("    • Run fewer subset sizes at once (--sizes 5000 only)")
        print("    • Add swap space (sudo fallocate -l 8G /swapfile)")
        print("    • Set DATALOADER_WORKERS=1 in .env once RAM is freed")
    else:
        print()
        print("  Parallelism note")
        print("-" * W)
        print(f"  {cfg.num_workers} worker process(es) will prefetch batches in parallel.")
        print("  Each holds a copy of the COCO train JSON in RAM — watch htop")
        print("  if memory climbs unexpectedly and drop to DATALOADER_WORKERS=0.")
        if cfg.persistent_workers:
            print("  persistent_workers avoids the re-fork + JSON reload cost each epoch.")
    print()


# ---------------------------------------------------------------------------
# .env export (merge — preserves existing keys like API keys)
# ---------------------------------------------------------------------------

_HW_KEYS = {
    "DATALOADER_WORKERS",
    "DATALOADER_PIN_MEMORY",
    "DATALOADER_PERSISTENT_WORKERS",
    "TRAINING_BATCH_SIZE",
    "TRAINING_DEVICE",
}


def export_env(cfg: TrainingConfig, path: str = ".env") -> None:
    """Merge hardware config into `path` (.env by default).

    Existing lines (e.g. API keys) are preserved.  HW config keys are
    updated in-place if already present, or appended under a section
    header if not.
    """
    new_values = {
        "DATALOADER_WORKERS":            str(cfg.num_workers),
        "DATALOADER_PIN_MEMORY":         "1" if cfg.pin_memory else "0",
        "DATALOADER_PERSISTENT_WORKERS": "1" if cfg.persistent_workers else "0",
        "TRAINING_BATCH_SIZE":           str(cfg.batch_size),
        "TRAINING_DEVICE":               cfg.device,
    }

    # Read existing file (tolerate missing)
    existing: list[str] = []
    try:
        with open(path) as f:
            existing = f.readlines()
    except FileNotFoundError:
        pass

    # Update keys that already exist in the file
    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in existing:
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in new_values:
            new_lines.append(f"{key}={new_values[key]}\n")
            updated_keys.add(key)
        else:
            new_lines.append(line)

    # Append keys that weren't already present
    pending = {k: v for k, v in new_values.items() if k not in updated_keys}
    if pending:
        if new_lines and not new_lines[-1].endswith("\n\n"):
            new_lines.append("\n")
        new_lines.append("# Hardware config (auto-set by hw_config.py)\n")
        for k, v in pending.items():
            new_lines.append(f"{k}={v}\n")

    with open(path, "w") as f:
        f.writelines(new_lines)

    print(f"  Merged into {path}  ({len(updated_keys)} updated, {len(pending)} added)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware config advisor")
    parser.add_argument("--export", action="store_true",
                        help="Merge recommended values into .env (preserves existing keys)")
    parser.add_argument("--env-file", default=".env",
                        help="Path for --export output (default: .env)")
    args = parser.parse_args()

    hw  = detect_hardware()
    cfg = recommend_config(hw)
    print_report(hw, cfg)

    if args.export:
        export_env(cfg, args.env_file)


if __name__ == "__main__":
    main()
