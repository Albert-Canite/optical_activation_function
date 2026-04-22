import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

from utils.cutout_augmentation import Cutout


DATASET_SPECS = {
    "cifar10": {
        "num_classes": 10,
        "in_channels": 3,
        "image_size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "mnist": {
        "num_classes": 10,
        "in_channels": 1,
        "image_size": 28,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
}


def get_dataset_spec(dataset_name):
    name = dataset_name.lower()
    if name not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from: {', '.join(DATASET_SPECS)}")
    return DATASET_SPECS[name]


def _build_transforms(dataset_name, augment=True, cutout=True):
    spec = get_dataset_spec(dataset_name)
    name = dataset_name.lower()

    train_transforms = []
    if augment:
        if name == "cifar10":
            train_transforms.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        elif name == "mnist":
            train_transforms.extend([
                transforms.RandomCrop(28, padding=2),
            ])

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(spec["mean"], spec["std"]),
    ])
    if cutout and name == "cifar10":
        train_transforms.append(Cutout(n_holes=1, length=16))

    eval_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(spec["mean"], spec["std"]),
    ]
    return transforms.Compose(train_transforms), transforms.Compose(eval_transforms)


def _dataset_class(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return datasets.CIFAR10
    if name == "mnist":
        return datasets.MNIST
    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def read_dataset(
    dataset_name="cifar10",
    batch_size=128,
    valid_size=0.2,
    calibration_size=0.05,
    num_workers=0,
    data_dir="dataset",
    download=True,
    augment=True,
    cutout=True,
    seed=42,
):
    dataset_name = dataset_name.lower()
    dataset_cls = _dataset_class(dataset_name)
    train_transform, eval_transform = _build_transforms(dataset_name, augment=augment, cutout=cutout)

    train_data = dataset_cls(data_dir, train=True, download=download, transform=train_transform)
    valid_data = dataset_cls(data_dir, train=True, download=download, transform=eval_transform)
    test_data = dataset_cls(data_dir, train=False, download=download, transform=eval_transform)

    num_train = len(train_data)
    indices = np.arange(num_train)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_valid = int(np.floor(valid_size * num_train))
    split_cal = int(np.floor(calibration_size * num_train))

    valid_idx = indices[:split_valid]
    cal_idx = indices[split_valid:split_valid + split_cal]
    train_idx = indices[split_valid + split_cal:]

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(valid_idx),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    cal_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(cal_idx),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, valid_loader, test_loader, cal_loader
