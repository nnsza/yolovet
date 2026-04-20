# -*- coding: utf-8 -*-
import argparse
import shutil
import sys
import traceback
from pathlib import Path

from ultralytics import YOLO


def norm_path(p: Path) -> str:
    s = str(p.resolve())
    if s.startswith("\\\\?\\"):
        return s[4:]
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="Built-in ONNX exporter for YOLOv8")
    parser.add_argument("--weights", required=True, help="best.pt path")
    parser.add_argument("--dest", required=True, help="best.onnx output path")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    dest = Path(args.dest).resolve()
    if not weights.is_file():
        print(f"[导出] 权重不存在: {weights}", flush=True)
        return 2
    dest.parent.mkdir(parents=True, exist_ok=True)

    print("[导出] 使用自带 ONNX 转码器", flush=True)
    print(f"[导出] weights={weights}", flush=True)
    print(f"[导出] dest={dest}", flush=True)

    try:
        model = YOLO(norm_path(weights))
        export_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            nms=False,
            batch=1,
            simplify=True,
            device="cpu",
        )
        if isinstance(export_path, (list, tuple)):
            export_path = export_path[0]
        export_file = Path(str(export_path)).resolve()
        if not export_file.is_file():
            print(f"[导出] 未找到导出文件: {export_path}", flush=True)
            return 3
        shutil.copy2(export_file, dest)
        print(f"[导出] 已复制 ONNX 至 {dest}", flush=True)
        return 0
    except BaseException as exc:
        print(f"[导出] ONNX 导出失败: {type(exc).__module__}.{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 10


if __name__ == "__main__":
    sys.exit(main())
