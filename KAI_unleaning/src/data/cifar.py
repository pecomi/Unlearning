import torch
import torchvision
from torch.utils.data import Subset

from .base import BaseDataset

# Improves model performance (https://github.com/weiaicunzai/pytorch-cifar100)
CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def _split_train_val(indices, val_ratio: float, generator: torch.Generator):
    """Split a list of indices into train and validation subsets."""
    if len(indices) == 0:
        return [], []

    shuffled = torch.tensor(indices)
    shuffled = shuffled[torch.randperm(len(shuffled), generator=generator)].tolist()
    val_size = min(int(len(shuffled) * val_ratio), len(shuffled))
    val_indices = shuffled[:val_size]
    train_indices = shuffled[val_size:]
    return train_indices, val_indices


def _class_balanced_train_val_indices(
    targets: torch.Tensor,
    num_classes: int,
    val_ratio: float,
    generator: torch.Generator,
):
    """Create one fixed class-balanced validation split before forget/retain splitting."""
    train_indices, val_indices = [], []
    for cls in range(num_classes):
        cls_idx = torch.where(targets == cls)[0]
        cls_idx = cls_idx[torch.randperm(len(cls_idx), generator=generator)].tolist()
        cls_train, cls_val = _split_train_val(cls_idx, val_ratio, generator)
        train_indices.extend(cls_train)
        val_indices.extend(cls_val)
    return train_indices, val_indices


class CIFAR10Dataset(BaseDataset):

    name = "cifar10"
    num_classes = 10

    def load(self) -> None:
        """Load CIFAR-10 train/test datasets onto self."""
        transform_train = torchvision.transforms.Compose([
            ### augmentation
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=CIFAR_MEAN,
                std=CIFAR_STD
            )
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=CIFAR_MEAN,
                std=CIFAR_STD
            )
        ])

        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=self.download,
            transform=transform_train
        )

        self.eval_train_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=False,
            transform=transform_test
        )

        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=self.download,
            transform=transform_test
        )

    def create_splits(self, forget_ratio: float, val_ratio: float,
                     batch_size: int, num_workers: int, seed: int,
                     pin_memory: bool = True, split_strategy: str = "random",
                     forget_labels: list = None) -> dict:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        targets = torch.tensor(self.train_dataset.targets)
        train_pool_indices, val_indices = _class_balanced_train_val_indices(
            targets,
            self.num_classes,
            val_ratio,
            g,
        )

        if split_strategy == "label_based":
            if forget_labels is None:
                raise ValueError("forget_labels must be specified for label_based strategy")

            # Ensure forget_labels is a list (handle both int and list inputs)
            if isinstance(forget_labels, int):
                forget_labels = [forget_labels]
            else:
                try:
                    forget_labels = list(forget_labels)
                except Exception:
                    raise ValueError(f"forget_labels must be an int or list-like object, got {type(forget_labels)}")

            forget_label_set = set(forget_labels)
            forget_indices = [
                idx for idx in train_pool_indices
                if targets[idx].item() in forget_label_set
            ]
            retain_indices = [
                idx for idx in train_pool_indices
                if targets[idx].item() not in forget_label_set
            ]
            forget_val_indices = [
                idx for idx in val_indices
                if targets[idx].item() in forget_label_set
            ]
            retain_val_indices = [
                idx for idx in val_indices
                if targets[idx].item() not in forget_label_set
            ]

            # DEBUG: Verify forget set contains only specified labels
            actual_forget_labels = [self.train_dataset.targets[i] for i in forget_indices]
            unique_forget_labels = sorted(set(actual_forget_labels))
            print(f"[DEBUG] Label-based split:")
            print(f"  Requested forget_labels: {forget_labels}")
            print(f"  Actual labels in forget set: {unique_forget_labels}")
            print(f"  Forget set size: {len(forget_indices)}")
            print(f"  Forget val set size: {len(forget_val_indices)}")
            print(f"  Retain val set size: {len(retain_val_indices)}")
            print(f"  Retain set size: {len(retain_indices)}")

        elif split_strategy == "all_labels":
            # Per-class forget split on the training pool; validation stays one shared set.
            forget_indices, retain_indices = [], []
            for cls in range(self.num_classes):
                cls_idx = torch.tensor([idx for idx in train_pool_indices if targets[idx].item() == cls])
                cls_idx = cls_idx[torch.randperm(len(cls_idx), generator=g)].tolist()
                n_forget = int(len(cls_idx) * forget_ratio)
                forget_indices.extend(cls_idx[:n_forget])
                retain_indices.extend(cls_idx[n_forget:])
            forget_val_indices, retain_val_indices = [], []

        else:  # Default: random strategy
            forget_size = int(len(train_pool_indices) * forget_ratio)
            indices = torch.tensor(train_pool_indices)
            indices = indices[torch.randperm(len(indices), generator=g)].tolist()
            forget_indices = indices[:forget_size]
            retain_indices = indices[forget_size:]
            forget_val_indices, retain_val_indices = [], []

        # Create subsets
        forget_set = Subset(self.train_dataset, forget_indices)
        forget_val_set = Subset(self.eval_train_dataset, forget_val_indices) if forget_val_indices else None
        retain_set = Subset(self.train_dataset, retain_indices)
        retain_val_set = Subset(self.eval_train_dataset, retain_val_indices) if retain_val_indices else None
        val_set = Subset(self.eval_train_dataset, val_indices)

        train_indices = forget_indices + retain_indices
        full_train_set = Subset(self.train_dataset, train_indices)

        # Create dataloaders
        return {
            "train_loader": self._create_dataloader(full_train_set, batch_size, True, num_workers, pin_memory=pin_memory),
            "forget_loader": self._create_dataloader(forget_set, batch_size, False, num_workers, pin_memory=pin_memory),
            "forget_val_loader": self._create_dataloader(forget_val_set, batch_size, False, num_workers, pin_memory=pin_memory) if forget_val_set is not None else None,
            "retain_loader": self._create_dataloader(retain_set, batch_size, True, num_workers, pin_memory=pin_memory),
            "retain_val_loader": self._create_dataloader(retain_val_set, batch_size, False, num_workers, pin_memory=pin_memory) if retain_val_set is not None else None,
            "val_loader": self._create_dataloader(val_set, batch_size, False, num_workers, pin_memory=pin_memory),
            "test_loader": self._create_dataloader(self.test_dataset, batch_size, False, num_workers, pin_memory=pin_memory),
        }


class CIFAR100Dataset(BaseDataset):

    name = "cifar100"
    num_classes = 100

    def load(self) -> None:
        """Load CIFAR-100 train/test datasets onto self."""
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=CIFAR_MEAN,
                std=CIFAR_STD
            )
        ])

        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.root,
            train=True,
            download=self.download,
            transform=transform
        )

        self.eval_train_dataset = torchvision.datasets.CIFAR100(
            root=self.root,
            train=True,
            download=False,
            transform=transform
        )

        self.test_dataset = torchvision.datasets.CIFAR100(
            root=self.root,
            train=False,
            download=self.download,
            transform=transform
        )

    def create_splits(self, forget_ratio: float, val_ratio: float,
                     batch_size: int, num_workers: int, seed: int,
                     pin_memory: bool = True, split_strategy: str = "random",
                     forget_labels: list = None) -> dict:
        
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        targets = torch.tensor(self.train_dataset.targets)
        train_pool_indices, val_indices = _class_balanced_train_val_indices(
            targets,
            self.num_classes,
            val_ratio,
            g,
        )

        if split_strategy == "label_based":
            if forget_labels is None:
                raise ValueError("forget_labels must be specified for label_based strategy")

            # Ensure forget_labels is a list (handle both int and list inputs)
            if isinstance(forget_labels, int):
                forget_labels = [forget_labels]
            elif not isinstance(forget_labels, (list, tuple)):
                raise ValueError(f"forget_labels must be an int or list/tuple, got {type(forget_labels)}")

            forget_label_set = set(forget_labels)
            forget_indices = [
                idx for idx in train_pool_indices
                if targets[idx].item() in forget_label_set
            ]
            retain_indices = [
                idx for idx in train_pool_indices
                if targets[idx].item() not in forget_label_set
            ]
            forget_val_indices = [
                idx for idx in val_indices
                if targets[idx].item() in forget_label_set
            ]
            retain_val_indices = [
                idx for idx in val_indices
                if targets[idx].item() not in forget_label_set
            ]

        elif split_strategy == "all_labels":
            # Per-class forget split on the training pool; validation stays one shared set.
            forget_indices, retain_indices = [], []
            for cls in range(self.num_classes):
                cls_idx = torch.tensor([idx for idx in train_pool_indices if targets[idx].item() == cls])
                cls_idx = cls_idx[torch.randperm(len(cls_idx), generator=g)].tolist()
                n_forget = int(len(cls_idx) * forget_ratio)
                forget_indices.extend(cls_idx[:n_forget])
                retain_indices.extend(cls_idx[n_forget:])
            forget_val_indices, retain_val_indices = [], []

        else:  # Default: random strategy
            forget_size = int(len(train_pool_indices) * forget_ratio)
            indices = torch.tensor(train_pool_indices)
            indices = indices[torch.randperm(len(indices), generator=g)].tolist()
            forget_indices = indices[:forget_size]
            retain_indices = indices[forget_size:]
            forget_val_indices, retain_val_indices = [], []

        # Create subsets
        forget_set = Subset(self.train_dataset, forget_indices)
        forget_val_set = Subset(self.eval_train_dataset, forget_val_indices) if forget_val_indices else None
        retain_set = Subset(self.train_dataset, retain_indices)
        retain_val_set = Subset(self.eval_train_dataset, retain_val_indices) if retain_val_indices else None
        val_set = Subset(self.eval_train_dataset, val_indices)
        
        train_indices = forget_indices + retain_indices
        full_train_set = Subset(self.train_dataset, train_indices)

        return {
            "train_loader": self._create_dataloader(full_train_set, batch_size, True, num_workers, pin_memory=pin_memory),
            "forget_loader": self._create_dataloader(forget_set, batch_size, False, num_workers, pin_memory=pin_memory),
            "forget_val_loader": self._create_dataloader(forget_val_set, batch_size, False, num_workers, pin_memory=pin_memory) if forget_val_set is not None else None,
            "retain_loader": self._create_dataloader(retain_set, batch_size, True, num_workers, pin_memory=pin_memory),
            "retain_val_loader": self._create_dataloader(retain_val_set, batch_size, False, num_workers, pin_memory=pin_memory) if retain_val_set is not None else None,
            "val_loader": self._create_dataloader(val_set, batch_size, False, num_workers, pin_memory=pin_memory),
            "test_loader": self._create_dataloader(self.test_dataset, batch_size, False, num_workers, pin_memory=pin_memory),
        }
