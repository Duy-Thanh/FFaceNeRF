import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import persistence
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────
# Argument Parser
# ─────────────────────────────
parser = argparse.ArgumentParser(description='Evaluate FFaceNeRF model.')
parser.add_argument('--network', required=True, help='Path to trained .pth file')
parser.add_argument('--num_classes', type=int, default=17, help='Number of segmentation classes')
parser.add_argument('--save_results', action='store_true', help='Save prediction masks to ./results')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────
# Metric Computation Utilities
# ─────────────────────────────
def compute_class_iou(preds, targets, num_classes):
    intersection = torch.zeros(num_classes).to(preds.device)
    union = torch.zeros(num_classes).to(preds.device)
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        intersection[cls] = (pred_mask & target_mask).sum()
        union[cls] = (pred_mask | target_mask).sum()
    iou = intersection / (union + 1e-6)
    return iou.cpu().numpy()

def compute_accuracy(preds, targets):
    correct = (preds == targets).float().sum()
    total = torch.numel(targets)
    return (correct / total).item()

def compute_mean_accuracy(preds, targets, num_classes):
    acc = 0.0
    valid_classes = 0
    for cls in range(num_classes):
        target_mask = targets == cls
        if target_mask.sum() == 0:
            continue
        pred_mask = preds == cls
        acc += (pred_mask & target_mask).sum().float() / (target_mask.sum().float() + 1e-6)
        valid_classes += 1
    return acc / max(valid_classes, 1)

# ─────────────────────────────
# Visualization Helper
# ─────────────────────────────
def visualize(pred, label, idx):
    os.makedirs('./results', exist_ok=True)
    pred_img = (pred.squeeze().cpu().numpy() * 12).astype(np.uint8)
    label_img = (label.squeeze().cpu().numpy() * 12).astype(np.uint8)
    Image.fromarray(pred_img).save(f'./results/pred_{idx:04d}.png')
    Image.fromarray(label_img).save(f'./results/label_{idx:04d}.png')

# ─────────────────────────────
# Main Evaluation
# ─────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Loading network: {args.network}")
    G = torch.load(f"./{args.network}", map_location=device).to(device).eval().requires_grad_(False)

    # Configure rendering parameters
    G.rendering_kwargs['depth_resolution'] = 96
    G.rendering_kwargs['depth_resolution_importance'] = 96

    fov_deg = 18.837
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)

    eval_list = [10, 13, 14, 15, 16, 26, 29, 33, 35, 45, 50, 59, 60, 61, 62, 63, 64, 65, 68, 69, 70,
                 71, 72, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 92, 93, 97, 98, 99]

    total_miou, total_acc, total_macc = [], [], []
    class_ious = []

    for img_num in tqdm(eval_list, desc="Evaluating"):
        ws_path = f'./data/ws/ws{img_num:04d}.pt'
        cam_path = f'./data/camera_params/c{img_num:04d}.pt'
        label_path = f'./data/labels_base/label{img_num:04d}.pt'

        if not (os.path.exists(ws_path) and os.path.exists(cam_path) and os.path.exists(label_path)):
            print(f"[WARN] Missing data for image {img_num:04d}, skipping.")
            continue

        ws = torch.load(ws_path, map_location=device)
        camera_params = torch.load(cam_path, map_location=device)
        label = torch.load(label_path).to(device)

        with torch.no_grad():
            outputs = G.synthesis(ws, camera_params)["image_seg"]
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            miou_vec = compute_class_iou(preds, label, args.num_classes)
            total_miou.append(np.nanmean(miou_vec))
            total_acc.append(compute_accuracy(preds, label))
            total_macc.append(compute_mean_accuracy(preds, label, args.num_classes).item())
            class_ious.append(miou_vec)

            if args.save_results:
                visualize(preds, label, img_num)

        torch.cuda.empty_cache()

    # ─────────────────────────────
    # Compute Final Metrics
    # ─────────────────────────────
    avg_miou = np.mean(total_miou)
    avg_acc = np.mean(total_acc)
    avg_macc = np.mean(total_macc)
    iou_per_class = np.nanmean(np.stack(class_ious), axis=0)

    print("\n─────────────────────────────")
    print(f"✅ Evaluation Summary:")
    print(f"Average mIoU: {avg_miou:.4f}")
    print(f"Average Pixel Accuracy: {avg_acc:.4f}")
    print(f"Average Mean Accuracy: {avg_macc:.4f}")
    print("─────────────────────────────\n")

    print("Per-Class IoU:")
    for i, val in enumerate(iou_per_class):
        print(f"  Class {i:02d}: {val:.4f}")

    # ─────────────────────────────
    # Save Results to CSV
    # ─────────────────────────────
    import pandas as pd
    df = pd.DataFrame({
        'ImageID': eval_list[:len(total_miou)],
        'mIoU': total_miou,
        'PixelAcc': total_acc,
        'MeanAcc': total_macc
    })
    df.to_csv('evaluation_results.csv', index=False)
    print("\n📄 Results saved to evaluation_results.csv")
    if args.save_results:
        print("🖼️ Prediction masks saved to ./results/")
