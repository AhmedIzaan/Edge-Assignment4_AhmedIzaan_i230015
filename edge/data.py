"""CIFAR data with a deterministic, class-balanced validation split."""
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from .preprocess import NORMALIZATION


def preprocessing(dataset, resolution, augment=False):
    operations = []
    if augment:
        operations.extend([transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip()])
    if resolution != 32:
        operations.append(transforms.Resize((resolution, resolution), antialias=True))
    return transforms.Compose([*operations, transforms.ToTensor(),
                               transforms.Normalize(*NORMALIZATION[dataset])])


class CIFARView(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        image = Image.fromarray(self.dataset.data[index])
        return self.transform(image), self.dataset.targets[index]


def load_cifar(dataset, root):
    cls = {"cifar100": datasets.CIFAR100, "cifar10": datasets.CIFAR10}[dataset]
    try:
        training = cls(root, train=True, download=False)
        testing = cls(root, train=False, download=False)
    except RuntimeError:
        training = cls(root, train=True, download=True)
        testing = cls(root, train=False, download=False)
    return training, testing


def split_indices(labels, seed, train_limit=0):
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    maximum = min(len(group) - len(group) // 10 for group in groups.values()) * len(groups)
    if train_limit < 0 or train_limit > maximum or train_limit % len(groups):
        raise ValueError(f"Training limit must be a multiple of {len(groups)} from 0 to {maximum}.")
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in sorted(groups):
        indices = rng.permutation(groups[label])
        count = len(indices) // 10
        val.extend(indices[:count])
        train.extend(indices[count:count + train_limit // len(groups)]
                     if train_limit else indices[count:])
    return train, val


def make_loaders(spec, args, training, testing, limit, *, skip_test=False):
    dataset, resolution = spec["dataset"], spec["resolution"]
    train, val = split_indices(training.targets, args.split_seed, limit)
    train_view = CIFARView(training, train, preprocessing(dataset, resolution, True))
    evaluation = preprocessing(dataset, resolution)
    val_view = CIFARView(training, val, evaluation)
    test_view = CIFARView(testing, range(len(testing)), evaluation)
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=args.device.startswith("cuda"),
                   persistent_workers=args.workers > 0)
    return (
        DataLoader(train_view, shuffle=True, generator=torch.Generator().manual_seed(args.seed),
                   drop_last=len(train_view) % args.batch_size == 1, **options),
        DataLoader(val_view, **options),
        None if skip_test else DataLoader(test_view, **options),
    )
