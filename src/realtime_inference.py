"""
Real-time ASL sign language inference from the laptop camera.

Usage:
    python -m src.realtime_inference --checkpoint checkpoints/<exp>/best.pt --model mobilenet_v3_small

Controls:
    q  — quit
    s  — save screenshot
"""

import argparse
import pathlib
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow running as a module (python -m src.realtime_inference) or as a script
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.augmentation import build_eval_transform
from src.model import get_model

CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "nothing",
]

# Guide-box fraction of the shorter frame dimension
ROI_FRACTION = 0.55


def load_checkpoint(model_name: str, checkpoint_path: str, device: str) -> torch.nn.Module:
    model = get_model(model_name, num_classes=len(CLASSES), freeze_backbone=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # last.pt is a full dict; best.pt is a raw state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_roi(frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop a centered square ROI from the frame. Returns (crop, (x, y, w, h))."""
    h, w = frame.shape[:2]
    sz = int(min(h, w) * ROI_FRACTION)
    cx, cy = w // 2, h // 2
    x0 = max(cx - sz // 2, 0)
    y0 = max(cy - sz // 2, 0)
    x1 = min(x0 + sz, w)
    y1 = min(y0 + sz, h)
    return frame[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def draw_overlay(
    frame: np.ndarray,
    roi_box: Tuple[int, int, int, int],
    top_results: List[Tuple[str, float]],
    fps: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, rw, rh = roi_box

    # Guide box around the ROI
    label_str, conf = top_results[0]
    box_color = (0, 220, 80) if conf >= 0.70 else (0, 200, 255) if conf >= 0.40 else (0, 80, 255)
    cv2.rectangle(frame, (x0, y0), (x0 + rw, y0 + rh), box_color, 2)
    cv2.putText(frame, "Place hand here", (x0, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 1, cv2.LINE_AA)

    # Top-prediction panel (top-left)
    panel_x, panel_y, panel_w, panel_h = 8, 8, 270, 155
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Big prediction label + confidence
    cv2.putText(frame, f"{label_str}", (panel_x + 12, panel_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, box_color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"{conf * 100:.1f}%", (panel_x + 90, panel_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (230, 230, 230), 2, cv2.LINE_AA)

    # Confidence bar
    bar_x, bar_y = panel_x + 12, panel_y + 65
    bar_w, bar_h = panel_w - 24, 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * conf), bar_y + bar_h), box_color, -1)

    # Top-2 and Top-3
    for i, (lbl, c) in enumerate(top_results[1:3], start=1):
        cv2.putText(frame, f"  {lbl}: {c * 100:.1f}%",
                    (panel_x + 12, panel_y + 88 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1, cv2.LINE_AA)

    # FPS (bottom-right)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 100, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)

    # Controls hint (bottom-left)
    cv2.putText(frame, "q: quit   s: screenshot", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1, cv2.LINE_AA)

    return frame


def run(checkpoint: str, model_name: str, camera: int, top_k: int, device: str) -> None:
    print(f"Loading '{model_name}' from {checkpoint} on {device} ...")
    model = load_checkpoint(model_name, checkpoint, device)
    transform = build_eval_transform(image_size=224, use_imagenet_norm=True)
    print("Ready. Camera opening ...")

    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera index {camera}. Try --camera 1 or another index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize latency

    prev_time = time.perf_counter()
    screenshot_idx = 0
    top_results: List[Tuple[str, float]] = [(CLASSES[0], 0.0)] * top_k

    print("Press  q  to quit,  s  to save a screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed — camera disconnected?")
            break

        roi_crop, roi_box = get_roi(frame)

        # BGR crop → RGB PIL → eval transform → batch
        rgb_crop = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_crop)

        with torch.no_grad():
            tensor = transform(pil_img).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        indices = probs.topk(top_k).indices.cpu().tolist()
        top_results = [(CLASSES[i], float(probs[i])) for i in indices]

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame = draw_overlay(frame, roi_box, top_results, fps)
        cv2.imshow("ASL Real-Time Inference  (q=quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"screenshot_{screenshot_idx:03d}.jpg"
            cv2.imwrite(path, frame)
            print(f"Screenshot saved: {path}")
            screenshot_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def _find_checkpoints() -> List[pathlib.Path]:
    ckpt_dir = pathlib.Path("checkpoints")
    return sorted(ckpt_dir.rglob("best.pt")) if ckpt_dir.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time ASL sign language inference from laptop camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.realtime_inference --checkpoint checkpoints/transfer_mobilenet/best.pt
  python -m src.realtime_inference --checkpoint checkpoints/exp/best.pt --model resnet18
  python -m src.realtime_inference --checkpoint checkpoints/exp/best.pt --camera 1
        """,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a best.pt checkpoint (saved model state_dict)")
    parser.add_argument("--model", default="mobilenet_v3_small",
                        choices=[
                            "baseline", "deep", "deep_regularized",
                            "deep_batchnorm", "deep_batchnorm_regularized",
                            "resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0",
                        ],
                        help="Model architecture — must match the checkpoint (default: mobilenet_v3_small)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top predictions shown in the overlay (default: 3)")
    parser.add_argument("--device", default=None,
                        help="Compute device: cpu or cuda (auto-detected if omitted)")
    args = parser.parse_args()

    ckpt = pathlib.Path(args.checkpoint)
    if not ckpt.exists():
        candidates = _find_checkpoints()
        if candidates:
            print("Checkpoint not found. Available best.pt files:")
            for c in candidates:
                print(f"  {c}")
        sys.exit(f"\nCheckpoint not found: {args.checkpoint}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run(str(ckpt), args.model, args.camera, args.top_k, device)


if __name__ == "__main__":
    main()
