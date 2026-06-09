"""
Render GradCAM overlays for sampled train and val images from a trained model.
Edit CONFIG, then run:  python3 gradcam_viz.py
"""

import random
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets
from torch.utils.data import Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from model import build_model, GradCAM
from dataset import get_transform, IMG_MEAN, IMG_STD

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_ROOT       = "/root/dataset/LumbarSpinalStenosis/LumbarSpinalStenosis"
MODEL_PATH      = "best_model.pth"
N_TRAIN         = 8        # images sampled from train folder
N_VAL           = 8        # images sampled from test folder
OUT_DIR         = "gradcam_viz"
EXCLUDE_CLASSES = {"Thecal Sac"}
IMG_SIZE        = 224
SEED            = 42
# ---------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def unnormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).numpy()
    return np.clip(img * np.array(IMG_STD) + np.array(IMG_MEAN), 0, 1)


def cam_overlay(img_np, cam_hw):
    h, w = img_np.shape[:2]
    cam_up = np.array(PILImage.fromarray((cam_hw * 255).astype(np.uint8)).resize(
        (w, h), PILImage.BILINEAR)) / 255.0
    cam_rgb = plt.cm.jet(cam_up)[:, :, :3]
    overlay = np.clip(0.5 * img_np + 0.5 * cam_rgb, 0, 1)
    return cam_up, overlay


def load_split(folder, n):
    """Return a Subset (with transforms) of n randomly sampled non-excluded images."""
    ds_raw = datasets.ImageFolder(str(folder))
    kept   = [i for i, (_, lbl) in enumerate(ds_raw.samples)
              if ds_raw.classes[lbl] not in EXCLUDE_CLASSES]
    chosen = random.sample(kept, min(n, len(kept)))
    ds_tf  = datasets.ImageFolder(str(folder), transform=get_transform(IMG_SIZE))
    return Subset(ds_tf, chosen)


def run_gradcam(ds, model, cam_fn, class_names, device, split_tag, out_dir):
    """
    Run GradCAM on every sample in ds.
    Saves one 3-panel PNG per sample and returns a list of row data for the summary.
    """
    rows = []
    for i in range(len(ds)):
        img_tensor, label = ds[i]
        cam_hw = cam_fn(img_tensor)
        img_np = unnormalize(img_tensor)
        cam_up, overlay = cam_overlay(img_np, cam_hw)

        with torch.no_grad():
            pred = model(img_tensor.unsqueeze(0).to(device)).argmax(1).item()

        true_name = class_names[label]
        pred_name = class_names[pred]
        correct   = label == pred

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np);             axes[0].set_title("Input");    axes[0].axis("off")
        axes[1].imshow(cam_up, cmap="jet"); axes[1].set_title("GradCAM"); axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title(f"true: {true_name}  pred: {pred_name}  {'OK' if correct else 'WRONG'}")
        axes[2].axis("off")
        plt.suptitle(f"{split_tag} sample {i}", fontsize=9, y=1.01)
        plt.tight_layout()
        fname = f"{out_dir}/{split_tag}_{i:04d}_{true_name.replace(' ', '_')}.png"
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)

        rows.append((split_tag, img_np, cam_up, overlay, true_name, pred_name, correct))
        status = "OK   " if correct else "WRONG"
        print(f"  {split_tag:5s} {i:3d} | {status} | true: {true_name:20s} | pred: {pred_name}")

    return rows


def save_summary(rows, out_dir):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)

    for r, (split_tag, img_np, cam_up, overlay, true_name, pred_name, correct) in enumerate(rows):
        label_str = f"[{split_tag}]\ntrue: {true_name}\npred: {pred_name}\n{'OK' if correct else 'WRONG'}"
        axes[r][0].imshow(img_np)
        axes[r][0].set_ylabel(label_str, fontsize=7, labelpad=4)
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        axes[r][1].imshow(cam_up, cmap="jet"); axes[r][1].axis("off")
        axes[r][2].imshow(overlay);            axes[r][2].axis("off")

    axes[0][0].set_title("Input",   fontsize=10)
    axes[0][1].set_title("GradCAM", fontsize=10)
    axes[0][2].set_title("Overlay", fontsize=10)

    plt.tight_layout()
    path = f"{out_dir}/summary.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary grid  → {path}")


def main():
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    root = Path(DATA_ROOT)
    train_ds = load_split(root / "train", N_TRAIN)
    val_ds   = load_split(root / "test",  N_VAL)

    # class_names from train folder, alphabetical, excluding unwanted classes
    raw_classes = datasets.ImageFolder(str(root / "train")).classes
    class_names = [c for c in raw_classes if c not in EXCLUDE_CLASSES]
    print(f"Classes: {class_names}\n")

    model = build_model(len(class_names), pretrained=False).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {MODEL_PATH}\n")

    Path(OUT_DIR).mkdir(exist_ok=True)
    cam_fn = GradCAM(model)

    print(f"Train samples ({N_TRAIN}):")
    train_rows = run_gradcam(train_ds, model, cam_fn, class_names, device, "train", OUT_DIR)

    print(f"\nVal samples ({N_VAL}):")
    val_rows = run_gradcam(val_ds, model, cam_fn, class_names, device, "val", OUT_DIR)

    save_summary(train_rows + val_rows, OUT_DIR)
    print(f"Individual PNGs → {OUT_DIR}/")


if __name__ == "__main__":
    main()
