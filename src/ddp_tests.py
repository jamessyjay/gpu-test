# src/ddp_tests.py
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from logging import basicConfig, getLogger, INFO, DEBUG
from os import environ
from sys import exit as sys_exit
from typing import Optional
from time import time

import torch
import torch.distributed as tdist

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
    """
    Parse command-line arguments.
    :return: Namespace - parsed arguments
    """
    ap = ArgumentParser()
    FP32 = 16_777_216  # Torch fp32 ~ 64MB
    ap.add_argument("--iters", type=int, default=1, help="all-reduce iterations")
    ap.add_argument("--numel", type=int, default=FP32, help="tensor elements (fp32 ~ 64MB)")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """
    Get an environment variable as an integer.
    :param name: str - name of the environment variable
    :param default: Optional[int] = None - default value if the environment variable is not set
    :return: Optional[int] - value of the environment variable
    """
    v = environ.get(name)
    if v is None:
        return default
    return int(v)


def _set_timeout(seconds: int = FIVE_MINUTES_TIMEOUT) -> timedelta:
    """
    Set the timeout for the process group.
    :param seconds: int = FIVE_MINUTES_TIMEOUT - timeout in seconds
    :return: timedelta - timeout
    """
    return timedelta(seconds=seconds)


def process_group(args: Namespace) -> int:
    """
    Process group.
    :param args: Namespace - parsed arguments
    :return: int - exit code
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

    # fill tensor with rank+1 to easily verify sum = world*(world+1)/2
    tensor = torch.full((args.numel,), float(rank + 1), device=device, dtype=torch.float32)

    # synchronize all processes
    torch.cuda.synchronize()
    start = time()
    for _ in range(args.iters):
        tdist.all_reduce(tensor, op=tdist.ReduceOp.SUM)
        # perform all-reduce operation across all processes to sum up the tensor
    torch.cuda.synchronize()
    total_time = time() - start

    expected = world * (world + 1) / 2.0
    ok = abs(float(tensor[0].item()) - expected) < 1e-3     # ok if within 1e-3 of expected
    if rank == 0:
        logger.info(f"[DDP] world={world} numel={args.numel} iters={args.iters} "
                    f"time={total_time:.3f}s ok={ok}")
    exit_code = 0 if ok else 3
    return exit_code


def main() -> None:
    args = parse_args()
    set_logger_level(DEBUG if args.verbose else INFO)
    try:
        exit_code = process_group(args)
    finally:
        if tdist.is_initialized():
            tdist.destroy_process_group()
    sys_exit(exit_code)


if __name__ == "__main__":
    main()