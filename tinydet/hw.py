"""Hardware detection and DataLoader config recommendation."""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass, field

# Empirical memory constants for this project
_COCO_TRAIN_JSON_GB = 2.0
_COCO_VAL_JSON_GB = 0.5
_MODEL_OVERHEAD_GB = 0.5
_OS_OVERHEAD_GB = 2.0
_ACTIVATION_PER_IMAGE_MB = 60.0
_GPU_MB_PER_IMAGE = 120.0


@dataclass
class HardwareInfo:
    total_ram_gb: float
    free_ram_gb: float
    vram_gb: float
    free_vram_gb: float
    cpu_count: int
    has_cuda: bool
    gpu_name: str
    is_wsl: bool


@dataclass
class TrainingConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    device: str
    notes: list[str] = field(default_factory=list)


def detect_hardware() -> HardwareInfo:
    import psutil
    vm = psutil.virtual_memory()

    has_cuda = False
    vram_gb = 0.0
    free_vram = 0.0
    gpu_name = "none"

    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            free_vram = (props.total_memory - torch.cuda.memory_reserved(0)) / 1024**3
            gpu_name = props.name
    except ImportError:
        pass

    is_wsl = False
    if os.path.exists("/proc/version"):
        with open("/proc/version") as f:
            is_wsl = "microsoft" in f.read().lower()

    return HardwareInfo(
        total_ram_gb=vm.total / 1024**3,
        free_ram_gb=vm.available / 1024**3,
        vram_gb=vram_gb,
        free_vram_gb=free_vram,
        cpu_count=multiprocessing.cpu_count(),
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        is_wsl=is_wsl,
    )


def recommend_config(hw: HardwareInfo | None = None) -> TrainingConfig:
    """Return recommended DataLoader and training settings for current hardware."""
    if hw is None:
        hw = detect_hardware()

    notes: list[str] = []

    # batch_size
    if hw.has_cuda:
        usable_vram_mb = (hw.vram_gb - 1.0) * 1024
        gpu_batch = _prev_power_of_two(max(1, int(usable_vram_mb / _GPU_MB_PER_IMAGE)))
        batch_size = min(gpu_batch, 32)
        device = "cuda"
        notes.append(f"GPU batch_size={batch_size} ({_GPU_MB_PER_IMAGE * batch_size:.0f} MB / {hw.vram_gb:.1f} GB VRAM)")
    else:
        cpu_batch = max(1, int(hw.free_ram_gb * 1024 * 0.25 / _ACTIVATION_PER_IMAGE_MB))
        batch_size = min(_prev_power_of_two(cpu_batch), 8)
        device = "cpu"
        notes.append(f"CPU-only: batch_size capped at {batch_size}")

    # num_workers — each worker copies the COCO train JSON into its address space
    training_baseline_gb = _COCO_TRAIN_JSON_GB + _MODEL_OVERHEAD_GB
    worker_budget = (hw.free_ram_gb - training_baseline_gb) * 0.8
    max_by_ram = max(0, int(worker_budget / _COCO_TRAIN_JSON_GB))
    max_by_cpu = max(0, hw.cpu_count // 2)
    num_workers = min(max_by_ram, max_by_cpu)

    if hw.is_wsl:
        num_workers = min(num_workers, 1)
        if num_workers > 0:
            notes.append("WSL2 detected: capping num_workers=1")

    if num_workers == 0:
        notes.append(
            f"num_workers=0 — only {hw.free_ram_gb:.1f} GB free; "
            f"baseline needs ~{training_baseline_gb:.1f} GB, {worker_budget:.1f} GB left for workers"
        )
    else:
        headroom = hw.free_ram_gb - training_baseline_gb - num_workers * _COCO_TRAIN_JSON_GB
        notes.append(f"num_workers={num_workers} (~{headroom:.1f} GB RAM headroom)")

    # pin_memory — only useful with real CUDA DMA; WSL2 has no DMA path
    pin_memory = hw.has_cuda and not hw.is_wsl
    if hw.has_cuda and hw.is_wsl:
        notes.append("pin_memory=False: WSL2 has no real CUDA DMA path")

    persistent_workers = num_workers > 0
    if persistent_workers:
        notes.append("persistent_workers=True: avoids re-fork + JSON reload each epoch")

    return TrainingConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        device=device,
        notes=notes,
    )


def _prev_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def print_report(hw: HardwareInfo, cfg: TrainingConfig) -> None:
    sep = "-" * 50
    print(sep)
    print(f"  RAM   : {hw.total_ram_gb:.1f} GB total, {hw.free_ram_gb:.1f} GB free")
    if hw.has_cuda:
        print(f"  GPU   : {hw.gpu_name} — {hw.vram_gb:.1f} GB VRAM")
    else:
        print("  GPU   : none (CPU-only)")
    print(f"  CPUs  : {hw.cpu_count}  |  WSL2: {'yes' if hw.is_wsl else 'no'}")
    print(sep)
    print(f"  device             = {cfg.device}")
    print(f"  batch_size         = {cfg.batch_size}")
    print(f"  num_workers        = {cfg.num_workers}")
    print(f"  pin_memory         = {cfg.pin_memory}")
    print(f"  persistent_workers = {cfg.persistent_workers}")
    print(sep)
    for note in cfg.notes:
        print(f"  • {note}")
    print()


_HW_KEYS = {
    "DATALOADER_WORKERS",
    "DATALOADER_PIN_MEMORY",
    "DATALOADER_PERSISTENT_WORKERS",
    "TRAINING_BATCH_SIZE",
    "TRAINING_DEVICE",
}


def export_env(cfg: TrainingConfig, path: str = ".env") -> None:
    """Merge hardware config into path (.env), preserving existing keys."""
    new_values = {
        "DATALOADER_WORKERS": str(cfg.num_workers),
        "DATALOADER_PIN_MEMORY": "1" if cfg.pin_memory else "0",
        "DATALOADER_PERSISTENT_WORKERS": "1" if cfg.persistent_workers else "0",
        "TRAINING_BATCH_SIZE": str(cfg.batch_size),
        "TRAINING_DEVICE": cfg.device,
    }

    existing: list[str] = []
    try:
        with open(path) as f:
            existing = f.readlines()
    except FileNotFoundError:
        pass

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
