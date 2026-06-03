#!/usr/bin/env python3
"""
Download pretrained torchvision weights for offline use on the cluster.

Run LOCALLY (internet required) before uploading to the cluster:
  python scripts/download_weights.py --out pretrained_weights/

Weights are saved flat: pretrained_weights/*.pth
Upload pretrained_weights/ directly to $DATA/pretrained_weights/ on the cluster.
"""

import argparse
import os
import pathlib
import shutil
import sys
import tempfile


BACKBONES = [
    ("resnet18",           "resnet18",           "ResNet18_Weights.DEFAULT"),
    ("resnet50",           "resnet50",           "ResNet50_Weights.DEFAULT"),
    ("mobilenet_v3_small", "mobilenet_v3_small", "MobileNet_V3_Small_Weights.DEFAULT"),
    ("efficientnet_b0",    "efficientnet_b0",    "EfficientNet_B0_Weights.DEFAULT"),
]


def download_all(out_dir: pathlib.Path):
    import torchvision.models as tv_models

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TORCH_HOME"] = tmp
        checkpoints_dir = pathlib.Path(tmp) / "hub" / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {len(BACKBONES)} backbones...\n")

        failed = []
        for name, factory_fn, weights_attr in BACKBONES:
            print(f"  [{name}] ... ", end="", flush=True)
            try:
                factory = getattr(tv_models, factory_fn)
                weights_cls, attr = weights_attr.split(".")
                weights = getattr(getattr(tv_models, weights_cls), attr)
                factory(weights=weights)
                print("OK")
            except Exception as exc:
                print(f"FAILED: {exc}")
                failed.append(name)

        # Flatten: move .pth files directly into out_dir
        for pth in checkpoints_dir.glob("*.pth"):
            shutil.move(str(pth), str(out_dir / pth.name))

    print(f"\nDone. Weights at: {out_dir}")
    if failed:
        print(f"FAILED backbones ({len(failed)}): {failed}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download pretrained weights for cluster use")
    parser.add_argument("--out", default="pretrained_weights",
                        help="Output directory - .pth files land here directly")
    args = parser.parse_args()
    download_all(pathlib.Path(args.out).resolve())


if __name__ == "__main__":
    main()
