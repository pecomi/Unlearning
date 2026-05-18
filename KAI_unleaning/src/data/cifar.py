import torch
import torchvision
from torch.utils.data import Subset

from .base import BaseDataset

# Improves model performance (https://github.com/weiaicunzai/pytorch-cifar100)
CIFAR_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


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
        total_samples = len(self.train_dataset)

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

            # Get targets from dataset
            targets = torch.tensor([self.train_dataset.targets[i] for i in range(len(self.train_dataset))])

            # Create boolean masks
            forget_mask = torch.isin(targets, torch.tensor(forget_labels))
            retain_mask = ~forget_mask

            # Get indices
            forget_indices = torch.where(forget_mask)[0].tolist()
            retain_indices = torch.where(retain_mask)[0].tolist()

            # DEBUG: Verify forget set contains only specified labels
            actual_forget_labels = [self.train_dataset.targets[i] for i in forget_indices]
            unique_forget_labels = sorted(set(actual_forget_labels))
            print(f"[DEBUG] Label-based split:")
            print(f"  Requested forget_labels: {forget_labels}")
            print(f"  Actual labels in forget set: {unique_forget_labels}")
            print(f"  Forget set size: {len(forget_indices)}")
            print(f"  Retain set size (before val split): {len(retain_indices)}")

            # Sample validation set from retain set
            val_size = min(int(len(retain_indices) * val_ratio), len(retain_indices))
            retain_indices = torch.tensor(retain_indices)
            retain_indices = retain_indices[torch.randperm(len(retain_indices))].tolist()
            val_indices = retain_indices[:val_size]
            retain_indices = retain_indices[val_size:]

            print(f"  Val set size: {len(val_indices)}")
            print(f"  Retain set size (after val split): {len(retain_indices)}")

        elif split_strategy == "all_labels":
            # Per-class: forget_ratio for forget, then val_ratio of remainder for val
            targets = torch.tensor(self.train_dataset.targets)
            g = torch.Generator().manual_seed(seed)

            forget_indices, val_indices, retain_indices = [], [], []
            for cls in range(self.num_classes):
                cls_idx = torch.where(targets == cls)[0]
                cls_idx = cls_idx[torch.randperm(len(cls_idx), generator=g)].tolist()
                n_forget = int(len(cls_idx) * forget_ratio)
                n_val = int((len(cls_idx) - n_forget) * val_ratio)
                forget_indices.extend(cls_idx[:n_forget])
                val_indices.extend(cls_idx[n_forget:n_forget + n_val])
                retain_indices.extend(cls_idx[n_forget + n_val:])

        else:  # Default: random strategy
            # Calculate split sizes
            forget_size = int(total_samples * forget_ratio)
            val_size = int(total_samples * val_ratio)
            retain_size = total_samples - forget_size - val_size

            # Create indices
            indices = torch.randperm(total_samples).tolist()

            forget_indices = indices[:forget_size]
            val_indices = indices[forget_size:forget_size + val_size]
            retain_indices = indices[forget_size + val_size:]

        # Create subsets
        forget_set = Subset(self.train_dataset, forget_indices)
        val_set = Subset(self.train_dataset, val_indices)
        retain_set = Subset(self.train_dataset, retain_indices)

        train_indices = forget_indices + retain_indices
        full_train_set = Subset(self.train_dataset, train_indices)

        # Create dataloaders
        return {
            "train_loader": self._create_dataloader(full_train_set, batch_size, True, num_workers, pin_memory=pin_memory),
            "forget_loader": self._create_dataloader(forget_set, batch_size, False, num_workers, pin_memory=pin_memory),
            "retain_loader": self._create_dataloader(retain_set, batch_size, True, num_workers, pin_memory=pin_memory),
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
        total_samples = len(self.train_dataset)

        if split_strategy == "label_based":
            if forget_labels is None:
                raise ValueError("forget_labels must be specified for label_based strategy")

            # Ensure forget_labels is a list (handle both int and list inputs)
            if isinstance(forget_labels, int):
                forget_labels = [forget_labels]
            elif not isinstance(forget_labels, (list, tuple)):
                raise ValueError(f"forget_labels must be an int or list/tuple, got {type(forget_labels)}")

            # Get targets from dataset
            targets = torch.tensor([self.train_dataset.targets[i] for i in range(len(self.train_dataset))])

            # Create boolean masks
            forget_mask = torch.isin(targets, torch.tensor(forget_labels))
            retain_mask = ~forget_mask

            # Get indices
            forget_indices = torch.where(forget_mask)[0].tolist()
            retain_indices = torch.where(retain_mask)[0].tolist()

            # Sample validation set from retain set
            val_size = min(int(len(retain_indices) * val_ratio), len(retain_indices))
            retain_indices = torch.tensor(retain_indices)
            retain_indices = retain_indices[torch.randperm(len(retain_indices))].tolist()
            val_indices = retain_indices[:val_size]
            retain_indices = retain_indices[val_size:]

        elif split_strategy == "all_labels":
            # Per-class: forget_ratio for forget, then val_ratio of remainder for val
            targets = torch.tensor(self.train_dataset.targets)
            g = torch.Generator().manual_seed(seed)

            forget_indices, val_indices, retain_indices = [], [], []
            for cls in range(self.num_classes):
                cls_idx = torch.where(targets == cls)[0]
                cls_idx = cls_idx[torch.randperm(len(cls_idx), generator=g)].tolist()
                n_forget = int(len(cls_idx) * forget_ratio)
                n_val = int((len(cls_idx) - n_forget) * val_ratio)
                forget_indices.extend(cls_idx[:n_forget])
                val_indices.extend(cls_idx[n_forget:n_forget + n_val])
                retain_indices.extend(cls_idx[n_forget + n_val:])

        else:  # Default: random strategy
            # Calculate split sizes
            forget_size = int(total_samples * forget_ratio)
            val_size = int(total_samples * val_ratio)
            retain_size = total_samples - forget_size - val_size

            indices = torch.randperm(total_samples).tolist()

            forget_indices = indices[:forget_size]
            val_indices = indices[forget_size:forget_size + val_size]
            retain_indices = indices[forget_size + val_size:]

        # Create subsets
        forget_set = Subset(self.train_dataset, forget_indices)
        val_set = Subset(self.train_dataset, val_indices)
        retain_set = Subset(self.train_dataset, retain_indices)
        
        train_indices = forget_indices + retain_indices
        full_train_set = Subset(self.train_dataset, train_indices)

        return {
            "train_loader": self._create_dataloader(full_train_set, batch_size, True, num_workers, pin_memory=pin_memory),
            "forget_loader": self._create_dataloader(forget_set, batch_size, False, num_workers, pin_memory=pin_memory),
            "retain_loader": self._create_dataloader(retain_set, batch_size, True, num_workers, pin_memory=pin_memory),
            "val_loader": self._create_dataloader(val_set, batch_size, False, num_workers, pin_memory=pin_memory),
            "test_loader": self._create_dataloader(self.test_dataset, batch_size, False, num_workers, pin_memory=pin_memory),
        }
