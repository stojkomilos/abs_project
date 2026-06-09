"""
Lumbar Spine MRI — training entry point.
Change MODEL_TYPE below to switch between models, then run:  python3 train.py
"""

import copy
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from dataset import make_loaders, IMG_MEAN, IMG_STD
from model import build_model, build_vit, save_gradcam, GradCAM

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
MODEL_TYPE = "vit"   # "resnet50" | "vit"

DATA_ROOT        = "/root/dataset/LumbarSpinalStenosis/LumbarSpinalStenosis"
EPOCHS           = 20
BATCH            = 128
IMG_SIZE         = 224
MAX_TRAIN        = None          # None = full dataset
MAX_VAL          = 128           # None = full val; 0 or -1 = skip val entirely
EXCLUDE_CLASSES  = {"Thecal Sac"}
GRAD_CLIP        = 1.0
CKPT_INTERVAL    = 900           # seconds between intermediary saves (15 min)
GRADCAM_INTERVAL = 2             # epochs between GradCAM logs (ResNet only)
N_GRADCAM_VIZ    = 2             # samples per split per GradCAM log
WANDB_PROJECT    = "lumbar-spine-mri"
WANDB_ENABLED    = True

assert MODEL_TYPE in ["resnet50", "vit"]

# Per-model hyperparams — auto-selected from MODEL_TYPE, do not edit directly
_MODEL_CFG = {
    "resnet50": dict(lr=5e-4, weight_decay=1e-3, pretrained=False,
                     out_path="best_model_resnet50.pth",
                     intermediary_path="best_intermediary_resnet50.pth"),
    "vit":      dict(lr=1e-4, weight_decay=1e-2, pretrained=True,
                     out_path="best_model_vit.pth",
                     intermediary_path="best_intermediary_vit.pth"),
}
LR               = _MODEL_CFG[MODEL_TYPE]["lr"]
WEIGHT_DECAY     = _MODEL_CFG[MODEL_TYPE]["weight_decay"]
PRETRAINED       = _MODEL_CFG[MODEL_TYPE]["pretrained"]
OUT_PATH         = _MODEL_CFG[MODEL_TYPE]["out_path"]
INTERMEDIARY_PATH = _MODEL_CFG[MODEL_TYPE]["intermediary_path"]
# ---------------------------------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    profile
except NameError:
    def profile(f): return f


# ---------------------------------------------------------------------------
# GradCAM helpers (ResNet only)
# ---------------------------------------------------------------------------

def _unnormalize(img_tensor):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * np.array(IMG_STD) + np.array(IMG_MEAN), 0, 1)


def _cam_overlay(img_np, cam_hw):
    h, w = img_np.shape[:2]
    cam_up = np.array(PILImage.fromarray(
        (cam_hw * 255).astype(np.uint8)).resize((w, h), PILImage.BILINEAR)) / 255.0
    cam_rgb = plt.cm.jet(cam_up)[:, :, :3]
    return cam_up, np.clip(0.5 * img_np + 0.5 * cam_rgb, 0, 1)


def _gradcam_log(model, cam_fn, train_ds, val_ds, class_names, device, epoch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    log_dict = {}
    for split_tag, ds in [("train", train_ds)] + ([("val", val_ds)] if val_ds else []):
        indices = random.sample(range(len(ds)), min(N_GRADCAM_VIZ, len(ds)))
        for j, idx in enumerate(indices):
            img_tensor, label = ds[idx]
            cam_hw  = cam_fn(img_tensor)
            img_np  = _unnormalize(img_tensor)
            cam_up, overlay = _cam_overlay(img_np, cam_hw)
            with torch.no_grad():
                pred = model(img_tensor.unsqueeze(0).to(device)).argmax(1).item()
            true_name, pred_name = class_names[label], class_names[pred]
            correct = label == pred

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np);             axes[0].set_title("Input");    axes[0].axis("off")
            axes[1].imshow(cam_up, cmap="jet"); axes[1].set_title("GradCAM"); axes[1].axis("off")
            axes[2].imshow(overlay)
            axes[2].set_title(f"true: {true_name}  pred: {pred_name}  {'OK' if correct else 'WRONG'}")
            axes[2].axis("off")
            plt.suptitle(f"ep {epoch} | {split_tag} sample {j}", fontsize=9)
            plt.tight_layout()
            log_dict[f"gradcam/{split_tag}_{j}"] = wandb.Image(
                fig, caption=f"true:{true_name} pred:{pred_name} {'OK' if correct else 'WRONG'}")
            plt.close(fig)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    log_dict["gradcam/time_s"] = elapsed
    print(f"  [gradcam] logged {N_GRADCAM_VIZ} train + {N_GRADCAM_VIZ if val_ds else 0} val "
          f"images in {elapsed:.2f}s")
    return log_dict


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@profile
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_sum = correct = total = 0
    gnorm_sum = 0.0
    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
        optimizer.step()
        loss_sum  += loss.item() * imgs.size(0)
        correct   += (out.argmax(1) == labels).sum().item()
        total     += imgs.size(0)
        gnorm_sum += gnorm
        print(f"  ep {epoch:3d} step {step:4d} | loss {loss.item():.4f} | grad_norm {gnorm:.4f}")
    return loss_sum / total, correct / total, gnorm_sum / (step + 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
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
    print(f"Device     : {device}")
    print(f"Model type : {MODEL_TYPE}\n")

    train_loader, val_loader, train_ds, class_names, class_weights = make_loaders(
        data_root=DATA_ROOT, batch_size=BATCH, exclude_classes=EXCLUDE_CLASSES,
        max_train=MAX_TRAIN, max_val=MAX_VAL, img_size=IMG_SIZE, seed=SEED,
    )

    if MODEL_TYPE == "resnet50":
        model     = build_model(len(class_names), pretrained=PRETRAINED).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        cam_fn    = GradCAM(model)
    else:
        model     = build_vit(len(class_names), pretrained=PRETRAINED).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        cam_fn    = None   # ViT does not support GradCAM

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    val_ds    = val_loader.dataset if val_loader is not None else None

    if WANDB_ENABLED:
        try:
            wandb.init(
                project=WANDB_PROJECT,
                job_type=MODEL_TYPE,
                tags=[MODEL_TYPE],
                config=dict(
                    model=MODEL_TYPE, epochs=EPOCHS, batch=BATCH, lr=LR,
                    img_size=IMG_SIZE, pretrained=PRETRAINED, grad_clip=GRAD_CLIP,
                    weight_decay=WEIGHT_DECAY, max_train=MAX_TRAIN, max_val=MAX_VAL,
                ),
            )
        except Exception as e:
            print(f"[wandb] init failed ({e}) — continuing without wandb")
            globals()["WANDB_ENABLED"] = False

        if WANDB_ENABLED:
            try:
                wandb.run.name = f"{MODEL_TYPE}-{wandb.run.name}"
                wandb.run.save()
            except Exception:
                pass  # rename failed — logging still works
            import webbrowser
            webbrowser.open(wandb.run.url)
            print(f"[wandb] {wandb.run.name}  {wandb.run.url}")

    history = {"tr_loss": [], "tr_acc": [], "tr_gnorm": [], "vl_loss": [], "vl_acc": []}

    if no_val:
        print(f"\n{'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'s':>5}")
        print("-" * 28)
    else:
        print(f"\n{'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'VlLoss':>7}  {'VlAcc':>6}  {'s':>5}")
        print("-" * 44)

    best_val_acc, best_weights = training_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        history, no_val, device, cam_fn, train_ds, val_ds, class_names)

    if WANDB_ENABLED:
        wandb.finish()

    torch.save(best_weights, OUT_PATH)
    print(f"\nBest val acc : {best_val_acc:.4f}  — saved to {OUT_PATH}" if not no_val
          else f"\nSaved final weights to {OUT_PATH}")

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

    return

    # GradCAM full run (currently disabled — use gradcam_viz.py instead)
    model.load_state_dict(best_weights)
    save_gradcam(model, train_ds, class_names, "gradcam", device)


@profile
def training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler,
                  history, no_val, device, cam_fn, train_ds, val_ds, class_names):
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

        if cam_fn is not None and epoch % GRADCAM_INTERVAL == 0:
            log.update(_gradcam_log(model, cam_fn, train_ds, val_ds, class_names, device, epoch))

        if WANDB_ENABLED:
            wandb.log(log)

    return best_val_acc, best_weights


if __name__ == "__main__":
    main()
