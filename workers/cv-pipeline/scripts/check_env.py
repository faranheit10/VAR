"""Environment and dependency verification script for 8v8 Turf AI analytics pipeline.

Verifies that PyTorch, OpenCV, Ultralytics, Supervision, Transformers, UMAP,
Scikit-Learn, SciPy, Pydantic, Pytest, and MyPy are all successfully imported
and that CUDA / hardware acceleration or explicit CPU fallback is functional.
"""

from __future__ import annotations

import sys
from typing import Any, Dict


def verify_dependencies() -> Dict[str, str]:
    """Import and record versions of all required runtime dependencies."""
    results: Dict[str, str] = {}

    import torch
    results["torch"] = torch.__version__

    import torchvision
    results["torchvision"] = torchvision.__version__

    import cv2
    results["opencv"] = cv2.__version__

    import ultralytics
    results["ultralytics"] = ultralytics.__version__

    import supervision as sv
    results["supervision"] = sv.__version__

    import transformers
    results["transformers"] = transformers.__version__

    import umap
    results["umap-learn"] = getattr(umap, "__version__", "installed")

    import sklearn
    results["scikit-learn"] = sklearn.__version__

    import scipy
    results["scipy"] = scipy.__version__

    import pydantic
    results["pydantic"] = pydantic.__version__

    import pytest
    results["pytest"] = pytest.__version__

    import mypy.version
    results["mypy"] = mypy.version.__version__

    return results


def check_cuda_and_compute() -> tuple[bool, str, str]:
    """Inspect CUDA hardware availability and run a deterministic tensor operation.

    Returns:
        (cuda_available, status_string, device_name)
    """
    import torch

    cuda_available: bool = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        status_str = "CUDA_AVAILABLE: True"
        device = torch.device("cuda:0")
    else:
        device_name = "CPU (Fallback)"
        status_str = "CUDA_AVAILABLE: False (CPU fallback active)"
        device = torch.device("cpu")

    # Run deterministic sanity tensor computation
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)
    dot_product = float(torch.dot(a, b).item())
    expected_dot = 32.0  # 1*4 + 2*5 + 3*6 = 32
    if abs(dot_product - expected_dot) > 1e-5:
        raise ValueError(
            f"Tensor computation sanity check failed: expected {expected_dot}, got {dot_product}"
        )

    return cuda_available, status_str, device_name


def main() -> int:
    """Run all environment validation checks and return exit code."""
    print("=" * 60)
    print("TURFVISION CV PIPELINE ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version.split()[0]}")
    print("-" * 60)

    try:
        deps = verify_dependencies()
        print("Required Packages Loaded Successfully:")
        for pkg, ver in sorted(deps.items()):
            print(f"  - {pkg:<20}: {ver}")
    except Exception as exc:
        print(f"ERROR: Dependency import failed: {exc}", file=sys.stderr)
        return 1

    print("-" * 60)
    try:
        _, status_str, device_name = check_cuda_and_compute()
        print(status_str)
        print(f"Active Compute Device: {device_name}")
        print("Tensor Sanity Check: PASSED (dot_product = 32.0)")
    except Exception as exc:
        print(f"ERROR: Compute verification failed: {exc}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("ENVIRONMENT STATUS: OK - All checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
