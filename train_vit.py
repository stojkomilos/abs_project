"""
Lumbar Spine MRI — ViT-Base/16 training entry point.
Run independently:  python3 train_vit.py

ViT does not support GradCAM (no spatial feature maps).
For visualisation use gradcam_viz.py with the ResNet-50 model instead.
"""

import copy
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from dataset import make_loaders
from model import build_vit

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
DATA_ROOT         = "/root/dataset/LumbarSpinalStenosis/LumbarSpinalStenosis"
EPOCHS            = 20
BATCH             = 128
LR                = 2e-4          # lower LR for fine-tuning pretrained ViT
IMG_SIZE          = 224
MAX_TRAIN         = None
MAX_VAL           = 128
PRETRAINED        = True          # essential — ViT trains poorly from scratch on 5k images
OUT_PATH          = "best_model_vit.pth"
INTERMEDIARY_PATH = "best_intermediary_vit.pth"
EXCLUDE_CLASSES   = {"Thecal Sac"}
GRAD_CLIP         = 1.0
CKPT_INTERVAL     = 900           # seconds between intermediary saves (15 min)
WANDB_PROJECT     = "lumbar-spine-mri"
WANDB_ENABLED     = True
# ---------------------------------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    profile
except NameError:
    def profile(f): return f


@profile
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_sum = correct = total = 0
    gnorm_sum = 0.0
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
        gnorm_sum += gnorm
        print(f"  ep {epoch:3d} step {step:4d} | loss {loss.item():.4f} | grad_norm {gnorm:.4f}")
    return loss_sum / total, correct / total, gnorm_sum / (step + 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total


def main():
    no_val = MAX_VAL is not None and MAX_VAL <= 0

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device  : {device}\n")

    train_loader, val_loader, train_ds, class_names, class_weights = make_loaders(
        data_root=DATA_ROOT,
        batch_size=BATCH,
        exclude_classes=EXCLUDE_CLASSES,
        max_train=MAX_TRAIN,
        max_val=MAX_VAL,
        img_size=IMG_SIZE,
        seed=SEED,
    )

    model     = build_vit(len(class_names), pretrained=PRETRAINED).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # AdamW is the standard optimiser for ViT fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if WANDB_ENABLED:
        try:
            wandb.init(
                project=WANDB_PROJECT,
                config=dict(
                    model="vit_base_patch16_224", epochs=EPOCHS, batch=BATCH,
                    lr=LR, img_size=IMG_SIZE, pretrained=PRETRAINED,
                    grad_clip=GRAD_CLIP, weight_decay=1e-2,
                    max_train=MAX_TRAIN, max_val=MAX_VAL,
                ),
            )
            import webbrowser
            webbrowser.open(wandb.run.url)
            print(f"[wandb] run url: {wandb.run.url}")
        except Exception as e:
            print(f"[wandb] init failed ({e}) — continuing without wandb")
            globals()["WANDB_ENABLED"] = False

    history = {"tr_loss": [], "tr_acc": [], "tr_gnorm": [], "vl_loss": [], "vl_acc": []}

    if no_val:
        print(f"\n{'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'s':>5}")
        print("-" * 28)
    else:
        print(f"\n{'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'VlLoss':>7}  {'VlAcc':>6}  {'s':>5}")
        print("-" * 44)

    best_val_acc, best_weights = training_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        history, no_val, device)

    if WANDB_ENABLED:
        wandb.finish()

    torch.save(best_weights, OUT_PATH)
    if no_val:
        print(f"\nSaved final weights to {OUT_PATH}")
    else:
        print(f"\nBest val acc : {best_val_acc:.4f}  — saved to {OUT_PATH}")

    epochs_x = range(1, len(history["tr_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(epochs_x, history["tr_loss"], label="Train")
    if not no_val:
        axes[0].plot(epochs_x, history["vl_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs_x, history["tr_acc"], label="Train")
    if not no_val:
        axes[1].plot(epochs_x, history["vl_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[2].plot(epochs_x, history["tr_gnorm"], label="Train")
    axes[2].set_title("Grad Norm (mean/epoch)"); axes[2].set_xlabel("Epoch"); axes[2].legend()
    plt.tight_layout()
    curve_path = OUT_PATH.replace(".pth", "_curves.png")
    plt.savefig(curve_path)
    print(f"Learning curves saved to {curve_path}")


@profile
def training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler,
                  history, no_val, device):
    best_val_acc           = 0.0
    best_weights           = None
    best_intermediary_loss = float('inf')
    last_ckpt_time         = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_gnorm = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        history["tr_loss"].append(tr_loss)
        history["tr_acc"].append(tr_acc)
        history["tr_gnorm"].append(tr_gnorm)

        log = {"epoch": epoch, "train/loss": tr_loss,
               "train/acc": tr_acc, "train/gnorm": tr_gnorm}

        if no_val:
            print(f"{epoch:3d}  {tr_loss:7.4f}  {tr_acc:6.4f}  {time.time()-t0:5.1f}s")
        else:
            vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
            history["vl_loss"].append(vl_loss)
            history["vl_acc"].append(vl_acc)
            log["val/loss"] = vl_loss
            log["val/acc"]  = vl_acc
            marker = " *" if vl_acc > best_val_acc else ""
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_weights = copy.deepcopy(model.state_dict())
            print(f"{epoch:3d}  {tr_loss:7.4f}  {tr_acc:6.4f}  {vl_loss:7.4f}  {vl_acc:6.4f}  "
                  f"{time.time()-t0:5.1f}s{marker}")

            if time.time() - last_ckpt_time >= CKPT_INTERVAL and vl_loss < best_intermediary_loss:
                best_intermediary_loss = vl_loss
                last_ckpt_time = time.time()
                torch.save(model.state_dict(), INTERMEDIARY_PATH)
                print(f"  [ckpt] intermediary checkpoint saved to {INTERMEDIARY_PATH}"
                      f" (val_loss={vl_loss:.4f}, ep={epoch})")

        if WANDB_ENABLED:
            wandb.log(log)

    return best_val_acc, best_weights


if __name__ == "__main__":
    main()
