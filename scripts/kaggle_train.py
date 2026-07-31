"""Run the detector fine-tune on Kaggle's free GPU.

Local Apple-Silicon training is ~6x slower than a cloud GPU and, at the
scale that fits locally, measurably degrades the model (mAP50 0.863 ->
0.799). Kaggle gives 30 GPU-hours/week free, which is enough for the real
run: a few thousand generated images, two-phase freeze/unfreeze, replay of
the original data.

Kaggle runs code as *kernels* (notebooks/scripts) against *datasets*, so
this script does three things:
  package  -- bundle the training inputs (KataCR code, sprites, 2026
              backgrounds, base weights) into a dataset directory
  push     -- upload the dataset and a training kernel
  fetch    -- poll the kernel and download the trained weights

Auth: KAGGLE_KEY (Bearer token) and KAGGLE_USERNAME in .env. The username
is required because Kaggle addresses everything as "<username>/<slug>";
a token whose users.get scope is denied cannot resolve it automatically.

    set -a && source .env && set +a
    .venv/bin/python scripts/kaggle_train.py package
    .venv/bin/python scripts/kaggle_train.py push
    .venv/bin/python scripts/kaggle_train.py fetch
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
BUNDLE = ROOT / "data" / "kaggle_bundle"
DATASET_SLUG = "clashcopilot-train"
KERNEL_SLUG = "clashcopilot-detector-fine-tune"  # Kaggle derives the slug from the title
API = "https://www.kaggle.com/api/v1"

# What the kernel runs on Kaggle's GPU.
KERNEL_SOURCE = '''
import os, sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ultralytics==8.1.24", "numpy==1.26.4"], check=True)
DATA = "/kaggle/input/{dataset}"
sys.path.insert(0, f"{{DATA}}/KataCR")
os.makedirs("/kaggle/working/runs", exist_ok=True)

# point KataCR at the bundled dataset
import katacr.build_dataset.constant as const
const.path_dataset = __import__("pathlib").Path(f"{{DATA}}/katacr-dataset")

import katacr.yolov8.cfg as kcfg
kcfg.train_datasize = {datasize}
import katacr.yolov8.custom_dataset as cds
cds.train_datasize = {datasize}

from ultralytics.cfg import get_cfg
from katacr.yolov8.train import YOLO_CR

model = YOLO_CR(f"{{DATA}}/detector1_v0.7.13.pt")
cfg = dict(get_cfg(f"{{DATA}}/KataCR/katacr/yolov8/ClashRoyale.yaml"))
cfg.update(data=f"{{DATA}}/KataCR/katacr/yolov8/detector1/data.yaml",
           epochs={epochs}, batch={batch}, imgsz=896, device=0, workers=4,
           amp=True, val=True, plots=False, pretrained=True,
           optimizer="AdamW", lr0=1e-4, lrf=0.2, warmup_epochs=1.0,
           cos_lr=True, freeze={freeze},
           # checkpoint every epoch: Kaggle hard-kills sessions at 9h and a
           # killed run saves no output, so partial progress must be durable
           save_period=1,
           project="/kaggle/working/runs", name="ft", exist_ok=True)
model.train(**cfg)
print("TRAINING DONE")
'''


def auth_headers() -> dict:
    key = os.environ.get("KAGGLE_KEY")
    if not key:
        sys.exit("set KAGGLE_KEY in .env")
    return {"Authorization": f"Bearer {key}", "User-Agent": "kaggle-api",
            "Content-Type": "application/json"}


def kaggle_client():
    """Authenticated client.

    Newer Kaggle tokens (KGAT_ prefix) are access tokens, not the classic
    username+key pair: they authenticate as Bearer and the client picks
    them up only from KAGGLE_API_TOKEN. KAGGLE_USERNAME/KAGGLE_KEY must be
    absent, or the client takes the legacy path and uploads 401.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    if os.environ.get("KAGGLE_API_TOKEN"):
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)
    api = KaggleApi()
    api.authenticate()
    return api


def username() -> str:
    user = os.environ.get("KAGGLE_USERNAME")
    if not user:
        sys.exit("set KAGGLE_USERNAME in .env (this token cannot resolve it: "
                 "users.get is denied). It is the name in your Kaggle profile URL.")
    return user


def package(datasize: int, epochs: int, batch: int, freeze: int) -> None:
    """Assemble everything the kernel needs into one directory."""
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    katacr = ROOT / "vendor" / "KataCR"
    shutil.copytree(katacr / "katacr", BUNDLE / "KataCR" / "katacr",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    ds = ROOT / "data" / "footage" / "katacr-dataset" / "images"
    for sub in ("segment", "part2"):
        if (ds / sub).exists():
            shutil.copytree(ds / sub, BUNDLE / "katacr-dataset" / "images" / sub,
                            ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(katacr / "runs" / "detector1_v0.7.13.pt", BUNDLE / "detector1_v0.7.13.pt")

    (BUNDLE / "dataset-metadata.json").write_text(json.dumps({
        "title": "ClashCopilot training inputs",
        "id": f"{username()}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=1))
    (BUNDLE / "kernel.py").write_text(KERNEL_SOURCE.format(
        dataset=DATASET_SLUG, datasize=datasize, epochs=epochs,
        batch=batch, freeze=freeze))
    (BUNDLE / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username()}/{KERNEL_SLUG}",
        "title": "ClashCopilot detector fine-tune",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{username()}/{DATASET_SLUG}"],
    }, indent=1))

    size = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file())
    print(f"bundle ready: {BUNDLE} ({size/1e6:.0f} MB)")
    print("next: .venv/bin/python scripts/kaggle_train.py push")


def push() -> None:
    """Upload dataset + kernel using the official client (handles multipart)."""
    os.environ.setdefault("KAGGLE_USERNAME", username())
    api = kaggle_client()
    print("creating/updating dataset ...")
    try:
        api.dataset_create_version(str(BUNDLE), "update", dir_mode="zip")
    except Exception:
        api.dataset_create_new(str(BUNDLE), dir_mode="zip", public=False)
    print("pushing kernel ...")
    api.kernels_push(str(BUNDLE))
    print(f"kernel: https://www.kaggle.com/code/{username()}/{KERNEL_SLUG}")


def fetch() -> None:
    os.environ.setdefault("KAGGLE_USERNAME", username())
    api = kaggle_client()
    ref = f"{username()}/{KERNEL_SLUG}"
    status = api.kernels_status(ref)
    print("status:", status)
    if str(status.get("status", "")).lower() in ("complete", "completed"):
        out = ROOT / "data" / "runs" / "kaggle"
        out.mkdir(parents=True, exist_ok=True)
        api.kernels_output(ref, str(out))
        print(f"downloaded outputs -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["package", "push", "fetch"])
    parser.add_argument("--datasize", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--freeze", type=int, default=10)
    args = parser.parse_args()
    if args.action == "package":
        package(args.datasize, args.epochs, args.batch, args.freeze)
    elif args.action == "push":
        push()
    else:
        fetch()


if __name__ == "__main__":
    main()
