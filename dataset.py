from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


def get_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


def _filter_indices(dataset, exclude_classes: set) -> list:
    return [i for i, (_, label) in enumerate(dataset.samples)
            if dataset.classes[label] not in exclude_classes]


def _cap_balanced(indices, targets_array, class_to_idx, class_names, max_n) -> np.ndarray:
    n_per_class = max(1, max_n // len(class_names))
    chosen = []
    for cls in class_names:
        cidx = class_to_idx[cls]
        cls_samples = [i for i in indices if targets_array[i] == cidx]
        chosen.extend(cls_samples[:n_per_class])
    return np.array(chosen)


def make_loaders(
    data_root: str,
    batch_size: int,
    exclude_classes: set,
    max_train: int | None,
    max_val: int | None,
    val_split: float = 0.15,
    img_size: int = 224,
    seed: int = 42,
):
    """
    Returns (train_loader, val_loader, train_ds, class_names, class_weights).
    val_loader is None when max_val <= 0.
    """
    no_val = max_val is not None and max_val <= 0
    root = Path(data_root)
    tf = get_transform(img_size)

    src_train = datasets.ImageFolder(str(root / "train"))
    src_test  = datasets.ImageFolder(str(root / "test"))

    class_names = [c for c in src_train.classes if c not in exclude_classes]

    kept_train = np.array(_filter_indices(src_train, exclude_classes))
    tgt_train  = np.array(src_train.targets)[kept_train]

    if no_val:
        idx_train = kept_train
        idx_val_from_train = np.array([], dtype=int)
    else:
        idx_train, idx_val_from_train = train_test_split(
            kept_train, test_size=val_split, stratify=tgt_train, random_state=seed)

    if max_train is not None and max_train > 0 and max_train < len(idx_train):
        idx_train = _cap_balanced(idx_train, src_train.targets, src_train.class_to_idx, class_names, max_train)

    kept_test = np.array(_filter_indices(src_test, exclude_classes))

    if not no_val:
        if max_val is not None and max_val > 0:
            idx_val_from_train_capped = _cap_balanced(
                idx_val_from_train, src_train.targets, src_train.class_to_idx, class_names, max_val // 2)
            idx_val_from_test_capped = _cap_balanced(
                kept_test, src_test.targets, src_test.class_to_idx, class_names, max_val // 2)
        else:
            idx_val_from_train_capped = idx_val_from_train
            idx_val_from_test_capped  = kept_test

    pin = torch.cuda.is_available()

    train_ds = Subset(datasets.ImageFolder(str(root / "train"), transform=tf), idx_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(idx_train)), shuffle=True, num_workers=8, pin_memory=pin)

    val_loader = None
    if not no_val:
        val_ds = ConcatDataset([
            Subset(datasets.ImageFolder(str(root / "train"), transform=tf), idx_val_from_train_capped),
            Subset(datasets.ImageFolder(str(root / "test"),  transform=tf), idx_val_from_test_capped),
        ])
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=pin)

    train_targets = np.array(src_train.targets)[idx_train]
    train_counts  = {c: int(np.sum(train_targets == src_train.class_to_idx[c])) for c in class_names}
    counts        = np.array(list(train_counts.values()), dtype=float)
    class_weights = torch.tensor(counts.max() / counts, dtype=torch.float)

    print("=" * 50)
    print(f"CLASSES    : {class_names}")
    print(f"TRAIN TOTAL: {len(idx_train)}")
    for c, n in train_counts.items():
        print(f"  {c}: {n}")
    if no_val:
        print("VAL        : DISABLED")
    else:
        n_val_train = len(idx_val_from_train_capped)
        n_val_test  = len(idx_val_from_test_capped)
        vt  = np.array(src_train.targets)[idx_val_from_train_capped]
        vtt = np.array(src_test.targets)[idx_val_from_test_capped]
        val_counts = {c: int(np.sum(vt == src_train.class_to_idx[c])) +
                         int(np.sum(vtt == src_test.class_to_idx[c])) for c in class_names}
        print(f"VAL TOTAL  : {n_val_train + n_val_test}  (train split: {n_val_train}, test folder: {n_val_test})")
        for c, n in val_counts.items():
            print(f"  {c}: {n}")
    print("=" * 50)

    return train_loader, val_loader, train_ds, class_names, class_weights
