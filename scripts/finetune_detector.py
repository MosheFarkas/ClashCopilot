"""Fine-tune a KataCR detector on 2026-domain synthetic data.

Why: KataCR's weights were trained on 2024 arena backgrounds, and a large
part of their error on current footage is that the arena art changed
(seasonal skins, new tower skins, decorations). Their generator composites
labeled sprites onto background images, so adding backgrounds extracted
from current footage (scripts/extract_backgrounds.py, median over moving
objects) produces correctly-labeled 2026-domain training data for free.

This starts from their released weights and fine-tunes at a low learning
rate, so it adapts to the new backgrounds without discarding what the
model already knows.

Must run under .venv-katacr with vendor/KataCR on PYTHONPATH:
    .venv-katacr/bin/python scripts/finetune_detector.py --detector 1
        [--epochs 3] [--batch 4] [--imgsz 896] [--device mps] [--datasize 800]
"""

import argparse
import sys
from pathlib import Path

KATACR = Path(__file__).parent.parent / "vendor" / "KataCR"
sys.path.insert(0, str(KATACR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--datasize", type=int, default=800,
                        help="generated images per epoch (overrides their 20000)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    # Their dataset length is a module constant; shrink it so an epoch is
    # minutes rather than hours on a laptop GPU.
    import katacr.yolov8.cfg as kcfg

    kcfg.train_datasize = args.datasize
    import katacr.yolov8.custom_dataset as cds

    cds.train_datasize = args.datasize

    from ultralytics.cfg import get_cfg

    from katacr.yolov8.train import YOLO_CR

    weights = KATACR / "runs" / f"detector{args.detector}_v0.7.13.pt"
    model = YOLO_CR(str(weights))

    cfg = dict(get_cfg(str(KATACR / "katacr/yolov8/ClashRoyale.yaml")))
    cfg.update(
        data=str(KATACR / f"katacr/yolov8/detector{args.detector}/data.yaml"),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=0,          # MPS + fork workers is a known hang
        amp=False,          # AMP is unusable on MPS
        val=False,          # their validator needs the real-frame val set
        plots=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr,
        lrf=0.5,
        warmup_epochs=0.5,
        cos_lr=True,
        project=str(Path(__file__).parent.parent / "data" / "runs"),
        name=args.name or f"ft_detector{args.detector}",
        exist_ok=True,
        save_period=1,
    )
    print(f"fine-tuning detector{args.detector} from {weights.name}: "
          f"{args.epochs} epochs x {args.datasize} generated images, "
          f"batch {args.batch}, {args.device}")
    model.train(**cfg)


if __name__ == "__main__":
    main()
