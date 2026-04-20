# -*- coding: utf-8 -*-
import argparse
import os
import sys
import traceback
from pathlib import Path

from ultralytics import YOLO


def norm_path(p: Path) -> str:
    s = str(p.resolve())
    if s.startswith("\\\\?\\"):
        return s[4:]
    return s


def find_latest_best_pt(bundle: Path):
    runs = bundle / "runs" / "detect"
    if not runs.is_dir():
        return None
    best_path = None
    best_mtime = -1.0
    for sub in runs.iterdir():
        if not sub.is_dir():
            continue
        cand = sub / "weights" / "best.pt"
        if cand.is_file():
            try:
                mt = cand.stat().st_mtime
            except OSError:
                continue
            if mt > best_mtime:
                best_mtime = mt
                best_path = cand
    return best_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Built-in CPU trainer for YOLOv8")
    parser.add_argument("--bundle", required=True, help="train_<ts> bundle directory")
    parser.add_argument("--model", required=True, help="model filename inside bundle")
    parser.add_argument("--epochs", type=int, required=True, help="training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    model_path = bundle / args.model
    data_yaml = bundle / "data.yaml"

    if not bundle.is_dir():
        print(f"[错误] 训练包目录不存在: {bundle}", flush=True)
        return 2
    if not model_path.is_file():
        print(f"[错误] 权重不存在: {model_path}", flush=True)
        return 3
    if not data_yaml.is_file():
        print(f"[错误] data.yaml 不存在: {data_yaml}", flush=True)
        return 4

    os.chdir(norm_path(bundle))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    print("[训练] 使用自带 CPU 训练器", flush=True)
    print(f"[训练] bundle={bundle}", flush=True)
    print(f"[训练] data={data_yaml}", flush=True)
    print(f"[训练] model={model_path}", flush=True)
    print(f"[训练] epochs={args.epochs} imgsz={args.imgsz} device=cpu", flush=True)

    try:
        model = YOLO(norm_path(model_path))
        model.train(
            data=norm_path(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            workers=0,
            device="cpu",
        )
    except BaseException as exc:
        print(f"[训练] 异常: {type(exc).__module__}.{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 10

    print("[训练] 训练完成。", flush=True)
    best = find_latest_best_pt(bundle)
    if best is None:
        print("[训练] 未找到 runs/detect/*/weights/best.pt", flush=True)
        return 11
    print(f"[训练] best.pt={best}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
