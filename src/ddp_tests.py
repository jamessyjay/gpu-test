# src/ddp_tests.py
"""
Distributed Data Parallel (DDP) acceptance test.

This module validates inter-GPU (and inter-node) communication using a simple
all-reduce on a large FP32 tensor. It is intended to catch NCCL/network/config
issues and to provide a basic performance sanity check via total runtime.

Key behavior:
- Initializes a torch.distributed process group with NCCL backend.
- Selects CUDA device based on LOCAL_RANK and performs all-reduce across world.
- Verifies numeric correctness against expected sum.
- Auto-scales iterations and tensor size when CLI values are zero.
- Writes a JSON report from rank0 when requested and logs a concise summary.
- Skips gracefully (exit 0) when no GPUs are detected.

CLI parameters:
- --iters (int): all-reduce iterations. 0 = auto-scale with world size.
- --numel (int): number of tensor elements. 0 = auto-scale with world size.
- --verbose (flag): enable DEBUG logging.
- --report-dir (str, optional): directory to write JSON report (rank0 only).
- --report-name (str, optional): file name for JSON report (rank0 only).
"""
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from logging import basicConfig, getLogger, INFO, DEBUG
from os import environ
from sys import exit as sys_exit
from typing import Optional
from time import time
import json
import os

import torch
import torch.distributed as tdist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from gpu_tests import TinyMLP, ToyDataset

basicConfig(level=INFO)
logger = getLogger(__name__)


FIVE_MINUTES_TIMEOUT = 600


def set_logger_level(level: int = INFO) -> None:
    """
    Set the logger level.
    :param level: int = INFO - logging level
    """
    logger.setLevel(level)
    return logger


def parse_args() -> Namespace:
    """Parse command-line arguments for the DDP test.

    Returns:
        argparse.Namespace: Parsed arguments with fields:
          - iters (int): all-reduce iterations; 0 enables auto-scaling.
          - numel (int): tensor elements; 0 enables auto-scaling.
          - verbose (bool): enable DEBUG logs.
          - report_dir (str|None): directory for JSON report (rank0 only).
          - report_name (str|None): JSON report file name (rank0 only).
          - train_smoke (bool): enable DDP training smoke test.
          - epochs (int): epochs for training smoke.
          - batch_size (int): batch size for training smoke.
          - steps (int): max batches per epoch for training smoke (0 = full epoch).
    """
    ap = ArgumentParser()
    FP32 = 16_777_216  # Torch fp32 ~ 64MB
    ap.add_argument("--iters", type=int, default=0, help="all-reduce iterations (0=auto)")
    ap.add_argument("--numel", type=int, default=0, help="tensor elements (0=auto; fp32 ~ 64MB baseline)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--report-dir", type=str, default=None, help="directory to write JSON report (rank0 only)")
    ap.add_argument("--report-name", type=str, default=None, help="file name for JSON report (rank0 only)")
    ap.add_argument("--train-smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=0)
    return ap.parse_args()


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Read an integer environment variable.

    Args:
        name (str): Environment variable name.
        default (Optional[int]): Value to return if variable is unset.

    Returns:
        Optional[int]: Parsed integer value or default when not set.
    """
    v = environ.get(name)
    if v is None:
        return default
    return int(v)


def _set_timeout(seconds: int = FIVE_MINUTES_TIMEOUT) -> timedelta:
    """Create a timedelta for process group timeout.

    Args:
        seconds (int): Timeout in seconds.

    Returns:
        datetime.timedelta: Timeout value.
    """
    return timedelta(seconds=seconds)


def process_group(args: Namespace) -> int:
    """Run the DDP all-reduce test within an initialized process group.

    Steps:
      1) Initialize NCCL process group and derive rank/world/local_rank.
      2) Set CUDA device by local_rank.
      3) Auto-scale iterations/tensor size if args specify zero values.
      4) Perform all-reduce and measure total time.
      5) Validate result correctness; log and optionally write JSON on rank0.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Exit code (0 on success, non-zero on failure).
    """
    # initialize process group
    BACKEND = "nccl"
    tdist.init_process_group(backend=BACKEND, timeout=_set_timeout())

    # get rank and world size
    rank = tdist.get_rank()  # get rank of current process
    world = tdist.get_world_size()  # get total number of processes
    local_rank = _env_int("LOCAL_RANK", 0)  # get local rank of current process

    # set device
    torch.cuda.set_device(local_rank)  # set CUDA device for current process
    device = torch.device("cuda", local_rank)  # create torch device object

    # auto-scale parameters if requested
    iters = args.iters if args.iters > 0 else max(1, 2 * int(world))
    base_numel = 16_777_216
    numel = args.numel if args.numel > 0 else max(base_numel // max(1, world), 4_194_304)

    # fill tensor with rank+1 to easily verify sum = world*(world+1)/2
    tensor = torch.full((numel,), float(rank + 1), device=device, dtype=torch.float32)

    # synchronize all processes
    torch.cuda.synchronize()
    start = time()
    for _ in range(iters):
        tdist.all_reduce(tensor, op=tdist.ReduceOp.SUM)
        # perform all-reduce operation across all processes to sum up the tensor
    torch.cuda.synchronize()
    total_time = time() - start

    expected = world * (world + 1) / 2.0
    ok = abs(float(tensor[0].item()) - expected) < 1e-3     # ok if within 1e-3 of expected
    if rank == 0:
        logger.info(f"[DDP] world={world} numel={numel} iters={iters} time={total_time:.3f}s ok={ok}")
        # write JSON report if requested
        if args.report_dir:
            os.makedirs(args.report_dir, exist_ok=True)
            name = args.report_name or "ddp_tests_result.json"
            path = os.path.join(args.report_dir, name)
            with open(path, "w") as f:
                json.dump({
                    "world": world,
                    "iters": iters,
                    "numel": numel,
                    "time": total_time,
                    "ok": ok
                }, f)
    exit_code = 0 if ok else 3
    return exit_code


def ddp_training_smoke(args: Namespace) -> int:
    """Run a tiny DDP training loop to validate distributed training health.

    Uses ToyDataset/TinyMLP from gpu_tests with DistributedSampler and DDP.
    Verifies average loss across ranks decreases sufficiently.
    """
    BACKEND = "nccl"
    tdist.init_process_group(backend=BACKEND, timeout=_set_timeout())

    rank = tdist.get_rank()
    world = tdist.get_world_size()
    local_rank = _env_int("LOCAL_RANK", 0)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset = ToyDataset(device=str(device))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=max(1, int(args.batch_size)), sampler=sampler)

    model = TinyMLP(input_features=dataset.d, k_classes=dataset.k).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = max(1, int(args.epochs))
    max_steps = max(0, int(args.steps))

    losses = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        seen = 0
        steps = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            bs = batch_x.size(0)
            epoch_loss += loss.item() * bs
            seen += bs
            steps += 1
            if max_steps and steps >= max_steps:
                break
        avg_loss = epoch_loss / max(1, seen)
        t = torch.tensor([avg_loss], device=device, dtype=torch.float32)
        tdist.all_reduce(t, op=tdist.ReduceOp.SUM)
        t /= float(world)
        losses.append(float(t.item()))

    loss_first = losses[0]
    loss_last = losses[-1]
    ok = loss_last < loss_first * 0.8  # require at least 20% improvement

    if rank == 0:
        logger.info(f"[DDP-TRAIN] world={world} epochs={epochs} batch={args.batch_size} steps={max_steps} ok={ok}")
        if args.report_dir:
            os.makedirs(args.report_dir, exist_ok=True)
            name = args.report_name or "ddp_training_result.json"
            path = os.path.join(args.report_dir, name)
            with open(path, "w") as f:
                json.dump({
                    "world": world,
                    "epochs": epochs,
                    "batch_size": args.batch_size,
                    "steps": max_steps,
                    "loss_start": loss_first,
                    "loss_end": loss_last,
                    "improved": ok
                }, f)

    return 0 if ok else 3



def main() -> None:
    """Program entry point: configure logging, skip when no GPU, orchestrate test.

    Behavior:
      - Parses arguments and sets logging level.
      - If no GPUs are visible, writes a skipped JSON (when requested) and exits 0.
      - Otherwise, executes the DDP process group test and propagates its exit code.
    """
    args = parse_args()
    set_logger_level(DEBUG if args.verbose else INFO)
    # no-GPU handling: skip gracefully
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        logger.info("[DDP] No GPUs detected, skipping DDP test")
        if args.report_dir:
            os.makedirs(args.report_dir, exist_ok=True)
            # choose file name based on mode
            name = args.report_name or ("ddp_training_skipped.json" if args.train_smoke else "ddp_tests_skipped.json")
            path = os.path.join(args.report_dir, name)
            with open(path, "w") as f:
                json.dump({"skipped": True, "reason": "no_gpu"}, f)
        sys_exit(0)
    try:
        if getattr(args, "train_smoke", False):
            exit_code = ddp_training_smoke(args)
        else:
            exit_code = process_group(args)
    finally:
        if tdist.is_initialized():
            tdist.destroy_process_group()
    sys_exit(exit_code)


if __name__ == "__main__":
    main()