"""Run the Kaggle kernel locally against a simulated Kaggle layout.

A kernel that fails on Kaggle costs a queue slot and returns almost no
diagnostics (the first attempt reported only "ERROR"), so the whole path
is exercised here first: read-only input dir, the bundled code and data,
env-driven dataset path, jax/torch shims, and a real 1-epoch train+val.

    .venv-katacr/bin/python scripts/simulate_kaggle.py [--epochs 1] [--datasize 8]

Passing here means the kernel's logic is sound; it does not prove Kaggle's
GPU image has identical package versions.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
BUNDLE = ROOT / "data" / "kaggle_bundle"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--datasize", type=int, default=8)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    if not (BUNDLE / "kernel.py").exists():
        sys.exit("bundle missing; run: kaggle_train.py package")

    sandbox = Path(tempfile.mkdtemp(prefix="kaggle_sim_"))
    input_dir = sandbox / "input" / "clashcopilot-train"
    working = sandbox / "working"
    working.mkdir(parents=True)
    print(f"simulating Kaggle in {sandbox}")

    # mirror the bundle as the read-only /kaggle/input mount
    shutil.copytree(BUNDLE, input_dir,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Kaggle mounts inputs read-only -- directories too, so code that
    # mkdirs next to its own source fails there. Marking only files
    # read-only let exactly that bug through to a real run.
    paths = sorted(input_dir.rglob("*"), reverse=True)
    for path in paths:
        path.chmod(0o444 if path.is_file() else 0o555)
    input_dir.chmod(0o555)

    kernel = (BUNDLE / "kernel.py").read_text()
    kernel = kernel.replace("epochs={}".format(_epochs_in(kernel)), f"epochs={args.epochs}") \
        if False else kernel  # kernel already carries its configured values
    script = sandbox / "kernel_local.py"
    script.write_text(kernel)

    env = {
        **os.environ,
        "CC_INPUT": str(input_dir),
        "CC_WORKING": str(working),
        "CC_INSTALL": "0",       # use the already-pinned local venv
        "CC_WORKERS": "0",       # MPS + forked workers hangs
        "KAGGLE_SIM": "1",
    }
    result = subprocess.run([sys.executable, str(script)], env=env,
                            capture_output=True, text=True)
    out = (result.stdout + result.stderr).splitlines()
    keep = [ln for ln in out if any(k in ln for k in (
        "preflight", "torch", "val images", "data.yaml", "jax", "TRAINING DONE",
        "Error", "error", "Traceback", "assert", "all ", "Exception"))]
    print("\n".join(keep[-30:]))
    print(f"\nexit code: {result.returncode}")

    weights = list((working / "runs").rglob("*.pt"))
    print(f"checkpoints written: {[w.name for w in weights]}")
    ok = result.returncode == 0 and any(w.name == "last.pt" for w in weights)
    print("\nSIMULATION PASSED" if ok else "\nSIMULATION FAILED")
    if not args.keep:
        for path in sorted(sandbox.rglob("*"), reverse=True):
            try:
                path.chmod(0o755)  # restore write bits so cleanup can remove them
            except OSError:
                pass
        shutil.rmtree(sandbox, ignore_errors=True)
    sys.exit(0 if ok else 1)


def _epochs_in(_: str) -> str:
    return ""


if __name__ == "__main__":
    main()
