"""
Unified test runner for GPU acceptance checks.

This module orchestrates execution of two test suites inside the container:
- gpu_tests.py: single-node compute and training smoke tests across all local GPUs
- ddp_tests.py: distributed all-reduce test via torchrun, scaling by number of GPUs

Features:
- Auto-detects number of visible GPUs and selects nproc_per_node accordingly
- Emits JSON reports for each test (delegated to test scripts) and a combined summary
- Prints concise console summary for CI

CLI parameters:
- --quick: bool — run with reduced workload for fast checks
- --verbose: bool — enable DEBUG logging
- --report-dir: str|None — directory to store JSON summary (created if absent)
- --report-name: str — filename of combined JSON summary (default: summary.json)
"""

from argparse import ArgumentParser
from logging import basicConfig, getLogger, INFO, DEBUG
from subprocess import run, PIPE
from typing import Dict, Any
import json
import os
import shlex
import torch

basicConfig(level=INFO)
logger = getLogger(__name__)


def parse_args():
    """Parse CLI arguments for the unified test runner.

    Returns:
        argparse.Namespace: Parsed arguments with fields:
            - quick (bool)
            - verbose (bool)
            - report_dir (str|None)
            - report_name (str)
    """
    ap = ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="quick run")
    ap.add_argument("--verbose", action="store_true", help="verbose output")
    ap.add_argument("--report-dir", type=str, default=None, help="directory to write JSON report")
    ap.add_argument("--report-name", type=str, default="summary.json", help="file name for JSON report")
    return ap.parse_args()


def _ensure_dir(path: str | None):
    """Create directory if path is provided and does not exist.

    Args:
        path (str|None): Target directory path or None to skip.
    """
    if path:
        os.makedirs(path, exist_ok=True)


def _run(cmd: str) -> Dict[str, Any]:
    """Run a shell command and capture outputs.

    Args:
        cmd (str): Command string to execute.

    Returns:
        Dict[str, Any]: Mapping with keys: cmd, returncode, stdout, stderr.
    """
    logger.info(f"[RUN] {cmd}")
    p = run(shlex.split(cmd), stdout=PIPE, stderr=PIPE, text=True)
    return {
        "cmd": cmd,
        "returncode": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
    }


def main():
    """Entry point for unified test execution.

    Logic:
    - Parse args and configure logging
    - Detect GPUs count to decide ddp nproc_per_node
    - Execute gpu_tests.py and, if GPUs present, ddp_tests.py via torchrun
    - Aggregate results, print CI-friendly summary, write combined JSON if requested

    Exits with non-zero code if any executed test failed.
    """
    args = parse_args()
    logger.setLevel(DEBUG if args.verbose else INFO)

    _ensure_dir(args.report_dir)

    have_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    ngpu = torch.cuda.device_count() if have_gpu else 0
    logger.info(f"[ENV] cuda_available={have_gpu} device_count={ngpu}")

    results: Dict[str, Any] = {
        "env": {
            "cuda_available": have_gpu,
            "cuda_device_count": ngpu,
            "torch": torch.__version__,
        },
        "gpu_tests": None,
        "ddp_tests": None,
    }

    # Run single-node compute/train tests (gpu_tests.py)
    gpu_report_path = None
    if args.report_dir:
        gpu_report_path = os.path.join(args.report_dir, "gpu_tests_result.json")
    cmd_gpu = "python src/gpu_tests.py"
    if args.quick:
        cmd_gpu += " --quick"
    if args.verbose:
        cmd_gpu += " --verbose"
    if args.report_dir:
        cmd_gpu += f" --report-dir {shlex.quote(args.report_dir)}"
    res_gpu = _run(cmd_gpu)

    # Run DDP all-reduce test with torchrun if GPUs present
    ddp_report_path = None
    res_ddp = None
    if have_gpu:
        nproc = ngpu
        cmd_ddp = (
            f"torchrun --standalone --nproc_per_node={nproc} src/ddp_tests.py"
        )
        if args.verbose:
            cmd_ddp += " --verbose"
        if args.report_dir:
            cmd_ddp += f" --report-dir {shlex.quote(args.report_dir)}"
        res_ddp = _run(cmd_ddp)
    else:
        # generate skipped report for ddp
        if args.report_dir:
            ddp_report_path = os.path.join(args.report_dir, "ddp_tests_skipped.json")
            with open(ddp_report_path, "w") as f:
                json.dump({"skipped": True, "reason": "no_gpu"}, f)

    # Collect outputs
    results["gpu_tests"] = {
        "returncode": res_gpu["returncode"],
        "stdout": res_gpu["stdout"],
        "stderr": res_gpu["stderr"],
    }
    if res_ddp is not None:
        results["ddp_tests"] = {
            "returncode": res_ddp["returncode"],
            "stdout": res_ddp["stdout"],
            "stderr": res_ddp["stderr"],
        }

    # Print concise console summary
    logger.info("[SUMMARY] gpu_tests rc=%s; ddp_tests rc=%s",
                results["gpu_tests"]["returncode"],
                results["ddp_tests"]["returncode"] if results["ddp_tests"] else "skipped")

    # Write combined JSON summary
    if args.report_dir:
        summary_path = os.path.join(args.report_dir, args.report_name)
        with open(summary_path, "w") as f:
            json.dump(results, f)
        logger.info(f"[SUMMARY] report saved: {summary_path}")

    # Exit code: non-zero if any present test failed
    exit_code = 0
    if results["gpu_tests"]["returncode"] != 0:
        exit_code = results["gpu_tests"]["returncode"]
    if results["ddp_tests"] and results["ddp_tests"]["returncode"] != 0:
        exit_code = results["ddp_tests"]["returncode"]
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
