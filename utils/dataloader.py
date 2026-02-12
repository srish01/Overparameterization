import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
import random

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_transform(dataset, ds_normalization=False):
    print(f"[TRAIN TRANSFORM] Input normlization is set to {ds_normalization}")

    if ds_normalization is True:
        if dataset == "mnist":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.1307,), (0.3081,)),
                                            transforms.Lambda(lambda img: img.reshape(-1))
                                            ])
        else:
            transform = transforms.Compose([transforms.RandomCrop(28, padding=0),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4916, 0.4824, 0.4467),
                                                                 (0.0299, 0.0295, 0.0318))
                                            ])
        return transform


    else:
        if dataset == "mnist":
            transform = transforms.Compose([transforms.ToTensor()]) #,transforms.Lambda(lambda img: img.reshape(-1))])
        else:
            transform = transforms.Compose([transforms.RandomCrop(28, padding=0),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            ])
        return transform
    
def get_test_transform(dataset, ds_normalization=False):

    """
    Transofrmations for test data for mnist and cifar10
    """
    print(f"[TEST TRANSFORM] Input normlization is set to {ds_normalization}")
    if ds_normalization is True:
        if dataset == "mnist":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.1307,), (0.3081,)),
                                            transforms.Lambda(lambda img: img.reshape(-1))
                                            ])
        else:
            transform = transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4916, 0.4824, 0.4467),
                                                                 (0.0299, 0.0295, 0.0318))
                                            ])
        return transform
    else:
        if dataset == "mnist":
            transform = transforms.Compose([transforms.ToTensor()]) #,transforms.Lambda(lambda img: img.reshape(-1))])
        else:
            transform = transforms.Compose([
                                            transforms.Resize(28),
                                            transforms.ToTensor(),
                                            ])
        return transform




def get_data_loaders(
    dataset_name: str,
    batch_size: int = 128,
    train_subset: float = 1.0,
    test_subset: float = 1.0,
    ds_normalization: bool = False,
    seed: int = 42,
    model_name: str = "mlp",
    num_workers: int = 2
):
    """
    Returns train, val, test DataLoaders for MNIST or CIFAR10.

    Args:
        dataset_name: "mnist" or "cifar10"
        batch_size: batch size for loaders
        train_subset: fraction of training data to use (0 < train_subset <= 1)
        ds_normalization: whether to apply dataset-specific normalization

    Returns:
        train_loader, val_loader, test_loader
    """
    
    dataset_name = dataset_name.lower()

    # ---------- RNG generator ----------
    g = torch.Generator()
    g.manual_seed(seed)

    # ---------- Transforms ----------  
    train_transform = get_train_transform(dataset_name, ds_normalization)   # SG: Removed model_name argument since train transform is same for all models
    test_transform = get_test_transform(dataset_name, ds_normalization)     # SG: Removed model_name argument since test transform is same for all models

    # ---------- Load full dataset ----------
    if dataset_name == "mnist":
        full_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=test_transform
        )

    elif dataset_name == "cifar10":
        full_train = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError("Dataset must be 'mnist' or 'cifar10'.")

    # ---------- Optional train subset ----------
    if train_subset < 1.0:
        subset_size = int(len(full_train) * train_subset)
        full_train, _ = random_split(
            full_train,
            [subset_size, len(full_train) - subset_size],
            generator=g
        )

    if test_subset < 1.0:
        subset_size = int(len(test_dataset) * test_subset)
        test_dataset, _ = random_split(
            test_dataset,
            [subset_size, len(test_dataset) - subset_size],
            generator=g
        )

    # ---------- Train / Validation split ----------
    val_size = int(len(full_train) * 0.2)
    train_size = len(full_train) - val_size

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=g
    )

    # ---------- DataLoaders ----------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
