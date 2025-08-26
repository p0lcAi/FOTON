"""
data_setup.py

Drop-in data utilities for MNIST/CIFAR10/CIFAR100:
- get_data_visually(...): backwards-compatible wrapper returning
  (train_loader, test_loader, class_names, class_to_idx, train_data, test_data, input_dim, num_classes).
- Standard normalizations and optional visualization with correct unnormalization.

"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Type, Union
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# -----------------------------
# Canonical normalization stats
# -----------------------------
_DATASET_STATS: Dict[str, Dict[str, Tuple[float, ...]]] = {
    "MNIST":    {"mean": (0.1307,), "std": (0.3081,)},
    "CIFAR10":  {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)},
    "CIFAR100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
}


@dataclass
class _Meta:
    name: str
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    classes: List[str]
    class_to_idx: Dict[str, int]
    input_dim: int
    num_classes: int


# -----------------------------
# Public API
# -----------------------------

def get_data_visually(
    dataset_class: Union[str, Type[datasets.VisionDataset]],
    batch_size: int = 32,
    to_print: bool = False,
    colored: bool = False,
    visually: bool = False,
    *,
    root: str = "data",
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last_train: bool = False,
    download: bool = True,
    seed: Optional[int] = 42,
):
    """
    Data loader builder for MNIST/CIFAR10/CIFAR100.

    Returns:
        train_dataloader, test_dataloader,
        class_names, class_to_idx,
        train_data, test_data,
        n (flattened input dim), m (#classes)
    """
    train_data, test_data, meta = _get_datasets_and_meta(
        dataset_class, root=root, download=download
    )

    if to_print:
        print(
            f"[{meta.name}] train={len(train_data)} samples | test={len(test_data)} samples "
            f"| input_dim={meta.input_dim} | classes={meta.num_classes}"
        )

    if visually:
        _plot_some_images(
            dataset=train_data,
            class_names=meta.classes,
            mean=meta.mean,
            std=meta.std,
            n=16,
            seed=seed,
            colored=colored,
        )

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    n = meta.input_dim
    m = meta.num_classes
    return (
        train_dataloader,
        test_dataloader,
        meta.classes,
        meta.class_to_idx,
        train_data,
        test_data,
        n,
        m,
    )


# -----------------------------
# Internals
# -----------------------------

def _normalize_dataset_arg(
    dataset: Union[str, Type[datasets.VisionDataset]]
) -> Tuple[Type[datasets.VisionDataset], str]:
    if isinstance(dataset, str):
        name = dataset.upper()
        if name not in _DATASET_STATS:
            raise ValueError(f"Unknown dataset name: {dataset} (expected one of {list(_DATASET_STATS)})")
        ds_class = getattr(datasets, name)
        return ds_class, name
    name = dataset.__name__.upper()
    if name not in _DATASET_STATS:
        raise ValueError(f"Unsupported dataset class: {dataset} (expected MNIST/CIFAR10/CIFAR100)")
    return dataset, name


def _get_datasets_and_meta(
    dataset: Union[str, Type[datasets.VisionDataset]],
    *,
    root: str,
    download: bool,
) -> Tuple[datasets.VisionDataset, datasets.VisionDataset, _Meta]:
    ds_class, ds_name = _normalize_dataset_arg(dataset)
    mean = _DATASET_STATS[ds_name]["mean"]
    std = _DATASET_STATS[ds_name]["std"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = ds_class(root=root, train=True, download=download, transform=transform)
    test_ds  = ds_class(root=root, train=False, download=download, transform=transform)

    # infer dims from one sample
    x0, _ = train_ds[0]  # [C,H,W], normalized tensor
    input_dim = int(torch.numel(x0))
    classes: List[str] = list(getattr(train_ds, "classes"))
    class_to_idx: Dict[str, int] = dict(getattr(train_ds, "class_to_idx"))

    meta = _Meta(
        name=ds_name,
        mean=mean,
        std=std,
        classes=classes,
        class_to_idx=class_to_idx,
        input_dim=input_dim,
        num_classes=len(classes),
    )
    return train_ds, test_ds, meta


def _plot_some_images(
    dataset: datasets.VisionDataset,
    class_names: Iterable[str],
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    n: int = 16,
    seed: Optional[int] = 42,
    colored: bool = False,
) -> None:
    """Visualize n random samples with correct unnormalization."""
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    else:
        g = None

    n = max(1, n)
    rows = cols = int(n ** 0.5)
    if rows * cols < n:
        cols += 1
        while rows * cols < n:
            rows += 1

    fig = plt.figure(figsize=(cols * 2.5, rows * 2.5))
    class_names = list(class_names)

    for i in range(1, rows * cols + 1):
        if i > n:
            break
        idx = torch.randint(0, len(dataset), (1,), generator=g).item()
        img_t, label = dataset[idx]  # [C,H,W] tensor
        img_np = _to_numpy_image(img_t, mean=mean, std=std)  # HxW or HxWxC

        ax = fig.add_subplot(rows, cols, i)
        if img_np.ndim == 3:
            if colored:
                ax.imshow(img_np)
            else:
                ax.imshow(img_np.mean(axis=2), cmap="gray")
        else:
            ax.imshow(img_np, cmap="gray")
        ax.set_title(class_names[label])
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def _to_numpy_image(
    x: torch.Tensor,
    mean: Optional[Tuple[float, ...]],
    std: Optional[Tuple[float, ...]],
):
    """Convert normalized tensor [C,H,W] -> numpy image (unnormalized + clipped to [0,1])."""
    import numpy as np  # local import to keep plotting optional

    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W] tensor, got {tuple(x.shape)}")

    x = x.detach().cpu().float()
    c, _, _ = x.shape

    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=x.dtype).view(c, 1, 1)
        std_t  = torch.tensor(std, dtype=x.dtype).view(c, 1, 1)
        x = x * std_t + mean_t

    if c == 1:
        return x.squeeze(0).clamp(0, 1).numpy()
    elif c == 3:
        return x.permute(1, 2, 0).clamp(0, 1).numpy()
    else:
        # Fallback for uncommon channel counts
        return x.mean(dim=0).clamp(0, 1).numpy()


__all__ = ["get_data_visually"]
