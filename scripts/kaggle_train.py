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
"""Fine-tune the KataCR detector. Runs on Kaggle GPU; also runnable locally
against a simulated layout (see scripts/simulate_kaggle.py) so the whole
path is proven before it costs GPU hours."""
import os, sys, types, shutil, subprocess
from pathlib import Path

MARKER = "detector1_v0.7.13.pt"


def find_input():
    """Locate the attached dataset by its marker file.

    Kaggle has mounted datasets at both /kaggle/input/<slug>/ and
    /kaggle/input/datasets/<owner>/<slug>/; searching for a known file is
    robust to either (and to the local simulator).
    """
    override = os.environ.get("CC_INPUT")
    if override:
        return override
    for base in ("/kaggle/input",):
        hits = sorted(Path(base).rglob(MARKER)) if Path(base).exists() else []
        if hits:
            return str(hits[0].parent)
    raise SystemExit(f"could not find {{MARKER}} under /kaggle/input")


DATA = find_input()
WORK = os.environ.get("CC_WORKING", "/kaggle/working")
Path(WORK).mkdir(parents=True, exist_ok=True)
print("input root:", DATA, flush=True)

# --- preflight: fail loudly and early rather than deep inside training ---
need = [f"{{DATA}}/KataCR/katacr", f"{{DATA}}/katacr-dataset/images/segment",
        f"{{DATA}}/katacr-dataset/images/part2", f"{{DATA}}/{{MARKER}}"]
missing = [p for p in need if not Path(p).exists()]
assert not missing, f"missing inputs: {{missing}}"
nbg = len(list(Path(f"{{DATA}}/katacr-dataset/images/segment/backgrounds").glob("*.jpg")))
print(f"preflight ok | backgrounds={{nbg}}", flush=True)

# Kaggle kernels have no internet unless the account is phone-verified, so
# nothing can be pip-installed. The pinned ultralytics (KataCR subclasses
# internals that moved after 8.1.x) is vendored as source and prepended to
# sys.path, ahead of whatever version Kaggle preinstalls.
sys.path.insert(0, f"{{DATA}}/pylibs")
import ultralytics
assert ultralytics.__version__.startswith("8.1"), (
    f"wrong ultralytics: {{ultralytics.__version__}} (vendored copy not picked up)")
print("ultralytics", ultralytics.__version__, "from", ultralytics.__file__, flush=True)

# KataCR's dataset path is read at import time from this env var
os.environ["KATACR_DATASET"] = f"{{DATA}}/katacr-dataset"
sys.path.insert(0, f"{{DATA}}/KataCR")

# jax is imported only for plotting helpers in the training import chain.
# Installing it risks a numpy conflict with torch, so satisfy the import
# with a stub when it is absent.
try:
    import jax  # noqa: F401
except Exception:
    stub = types.ModuleType("jax"); stub.numpy = types.ModuleType("jax.numpy")
    stub.jit = lambda f=None, **k: (f if f else (lambda g: g))
    sys.modules["jax"] = stub; sys.modules["jax.numpy"] = stub.numpy
    print("jax not present -> stubbed", flush=True)

# torch >= 2.6 defaults torch.load(weights_only=True), which cannot load
# these checkpoints (they pickle KataCR's custom model class).
import numpy as _np
import torch
# the torch<->numpy bridge is what actually breaks on a version mismatch,
# and it otherwise fails deep inside the dataloader
torch.from_numpy(_np.zeros((2, 2), dtype="float32"))
print("bridge ok | numpy", _np.__version__, flush=True)
_orig_load = torch.load
def _load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _load
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), flush=True)

# Rewrite the data config into a writable dir: /kaggle/input is read-only
# and the bundled yaml/annotation list carry authoring-machine paths.
part2 = f"{{DATA}}/katacr-dataset/images/part2"
ann = Path(WORK) / "yolo_annotation.txt"
imgs = sorted(str(p) for p in Path(part2).rglob("*.jpg"))
assert imgs, "no validation images found"
ann.write_text("\\n".join(imgs))
src_yaml = Path(f"{{DATA}}/KataCR/katacr/yolov8/detector1/data.yaml").read_text().split("\\n")
out = []
for line in src_yaml:
    if line.startswith("path:"): out.append(f"path: {{part2}}")
    elif line.startswith("val:"): out.append(f"val: {{ann}}")
    else: out.append(line)
data_yaml = Path(WORK) / "data.yaml"
data_yaml.write_text("\\n".join(out))
print(f"val images={{len(imgs)}} | data.yaml -> {{data_yaml}}", flush=True)

import katacr.yolov8.cfg as kcfg
kcfg.train_datasize = {datasize}
import katacr.yolov8.custom_dataset as cds
cds.train_datasize = {datasize}

from ultralytics.cfg import get_cfg
from katacr.yolov8.train import YOLO_CR

model = YOLO_CR(f"{{DATA}}/{{MARKER}}")
cfg = dict(get_cfg(f"{{DATA}}/KataCR/katacr/yolov8/ClashRoyale.yaml"))
cfg.update(data=str(data_yaml),
           epochs={epochs}, batch={batch}, imgsz=896,
           device=(0 if torch.cuda.is_available() else "cpu"),
           workers=int(os.environ.get("CC_WORKERS", "4")),
           amp=torch.cuda.is_available(), val=True, plots=False, pretrained=True,
           optimizer="AdamW", lr0=1e-4, lrf=0.2, warmup_epochs=1.0,
           cos_lr=True, freeze={freeze},
           # checkpoint every epoch: Kaggle hard-kills sessions at 9h and a
           # killed run saves no output, so partial progress must be durable
           save_period=1,
           project=f"{{WORK}}/runs", name="ft", exist_ok=True)
model.train(**cfg)
print("TRAINING DONE", flush=True)
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
    # Kernels have no internet, and .whl files do not survive Kaggle's
    # dataset processing -- so ship the pinned packages as plain source
    # trees and put them on sys.path. Both are pure Python.
    site = next((ROOT / ".venv-katacr" / "lib").glob("python3.*")) / "site-packages"
    libs = BUNDLE / "pylibs"
    libs.mkdir()
    for pkg in ("ultralytics", "thop"):
        src = site / pkg
        if src.exists():
            shutil.copytree(src, libs / pkg,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print("vendored:", [p.name for p in libs.iterdir()])

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
    # A kernel started before the dataset version finishes processing runs
    # against the PREVIOUS version -- which silently omits whatever the new
    # run depends on. Three failed runs traced to exactly this.
    import time

    for _ in range(60):
        status = str(api.dataset_status(f"{username()}/{DATASET_SLUG}")).lower()
        print("dataset status:", status, flush=True)
        if "ready" in status:
            break
        time.sleep(20)
    else:
        sys.exit("dataset never became ready; not pushing kernel")

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
