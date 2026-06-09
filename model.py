import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


class GradCAM:
    """Grad-CAM for ResNet: hooks layer4's last conv output."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._feats = None
        self._grads = None
        target_layer = model.layer4[-1].conv3 if hasattr(model.layer4[-1], "conv3") else model.layer4[-1].conv2
        target_layer.register_forward_hook(self._save_feats)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_feats(self, _, __, output):
        self._feats = output.detach()

    def _save_grads(self, _, __, grad_output):
        self._grads = grad_output[0].detach()

    def __call__(self, img_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Returns HxW heatmap in [0,1]."""
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0).to(next(self.model.parameters()).device)
        logits = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._feats).sum(dim=1).squeeze(0)
        cam = torch.relu(cam).cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def save_gradcam(model: nn.Module, dataset, class_names: list, out_dir: str, device):
    """Run GradCAM on every sample in dataset and save overlay PNGs."""
    from PIL import Image as PILImage
    Path(out_dir).mkdir(exist_ok=True)
    cam_fn = GradCAM(model)
    inv_mean = np.array(IMG_MEAN)
    inv_std  = np.array(IMG_STD)

    for i in range(len(dataset)):
        img_tensor, label = dataset[i]
        cam = cam_fn(img_tensor, class_idx=None)

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = img_np * inv_std + inv_mean
        img_np = np.clip(img_np, 0, 1)

        cam_pil = PILImage.fromarray((cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((img_np.shape[1], img_np.shape[0]), PILImage.BILINEAR)
        cam_resized = np.array(cam_pil) / 255.0
        cam_rgb = plt.cm.jet(cam_resized)[:, :, :3]

        overlay = np.clip(0.5 * img_np + 0.5 * cam_rgb, 0, 1)

        pred_idx = cam_fn.model(img_tensor.unsqueeze(0).to(device)).argmax(1).item()
        true_name = class_names[label]
        pred_name = class_names[pred_idx]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np);      axes[0].set_title("Input");   axes[0].axis("off")
        axes[1].imshow(cam_resized, cmap="jet"); axes[1].set_title("GradCAM"); axes[1].axis("off")
        axes[2].imshow(overlay);     axes[2].set_title(f"true:{true_name}\npred:{pred_name}"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/sample_{i:04d}_{true_name.replace(' ', '_')}.png", dpi=100)
        plt.close(fig)

    print(f"GradCAM saved to {out_dir}/ ({len(dataset)} images)")
