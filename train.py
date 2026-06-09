"""
Lumbar Spine MRI — training entry point.
Press F5 / run this file. Edit the globals below to change behaviour.
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

from dataset import make_loaders
from model import build_model, save_gradcam

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
DATA_ROOT       = "/root/dataset/LumbarSpinalStenosis/LumbarSpinalStenosis"
EPOCHS          = 2
BATCH           = 2
LR              = 1e-2
IMG_SIZE        = 224
MAX_TRAIN       = 2       # None = full dataset
MAX_VAL         = 0       # None = full val; 0 or -1 = skip val entirely
PRETRAINED      = False
OUT_PATH        = "best_model.pth"
EXCLUDE_CLASSES = {"Thecal Sac"}
GRADCAM_OUT     = "gradcam"
# ---------------------------------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# line_profiler: no-op when not running under kernprof
try:
    profile
except NameError:
    def profile(f): return f


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


@profile
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_sum = correct = total = 0
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        gnorm = grad_norm(model)
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += imgs.size(0)
        print(f"  ep {epoch:3d} step {step:4d} | loss {loss.item():.4f} | grad_norm {gnorm:.4f}")
    return loss_sum / total, correct / total


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

    model     = build_model(len(class_names), pretrained=PRETRAINED).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"tr_loss": [], "tr_acc": [], "vl_loss": [], "vl_acc": []}

    if no_val:
        print(f"\n{'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'s':>5}")
        print("-" * 28)
    else:
        print(f"\n{'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'VlLoss':>7}  {'VlAcc':>6}  {'s':>5}")
        print("-" * 44)

    best_val_acc, best_weights = training_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        history, no_val, device)

    torch.save(best_weights, OUT_PATH)
    if no_val:
        print(f"\nSaved final weights to {OUT_PATH}")
    else:
        print(f"\nBest val acc : {best_val_acc:.4f}  — saved to {OUT_PATH}")

    # Learning curves
    epochs_x = range(1, EPOCHS + 1)
    n_plots = 1 if no_val else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if no_val:
        axes = [axes]
    axes[0].plot(epochs_x, history["tr_loss"], label="Train")
    if not no_val:
        axes[0].plot(epochs_x, history["vl_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    if not no_val:
        axes[1].plot(epochs_x, history["tr_acc"], label="Train")
        axes[1].plot(epochs_x, history["vl_acc"], label="Val")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    curve_path = OUT_PATH.replace(".pth", "_curves.png")
    plt.savefig(curve_path)
    print(f"Learning curves saved to {curve_path}")

    # GradCAM
    model.load_state_dict(best_weights)
    save_gradcam(model, train_ds, class_names, GRADCAM_OUT, device)


@profile
def training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler,
                  history, no_val, device):
    best_val_acc = 0.0
    best_weights = None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        history["tr_loss"].append(tr_loss)
        history["tr_acc"].append(tr_acc)

        if no_val:
            print(f"{epoch:3d}  {tr_loss:7.4f}  {tr_acc:6.4f}  {time.time()-t0:5.1f}s")
            best_weights = copy.deepcopy(model.state_dict())
        else:
            vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
            history["vl_loss"].append(vl_loss)
            history["vl_acc"].append(vl_acc)
            marker = " *" if vl_acc > best_val_acc else ""
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_weights = copy.deepcopy(model.state_dict())
            print(f"{epoch:3d}  {tr_loss:7.4f}  {tr_acc:6.4f}  {vl_loss:7.4f}  {vl_acc:6.4f}  "
                  f"{time.time()-t0:5.1f}s{marker}")
    return best_val_acc, best_weights


if __name__ == "__main__":
    main()
