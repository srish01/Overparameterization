import json
import os
from statistics import mean

from torch import std
from utils.models import *
import argparse
import torch.optim as optim
import torch.nn.functional as F
from secml.ml.classifiers import CClassifierPyTorch
from secml.array import CArray
from secml.data import CDataset

MODEL_REGISTRY = {
    "resnet": ScalableResNet,
    "cnn": ScalableCNN,
    "mlp": ScalableMLP,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolation and overparameterization experiments"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["search", "train", "run_attacks"],
        default="train",
        # required=True,
        help="Run hyperparameter search or perform training"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=MODEL_REGISTRY.keys(),
        required=True,
        help="Model architecture"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "mnist"],
        required=True,
        help="Dataset name"
    )

    parser.add_argument(
        "--capacity",
        type=int,
        nargs="+",
        help="Model capacity (single int or list for sweep)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--hp_epochs",
        type=int,
        default=30,
        help="Number of epochs for hyperparameter search"
    )

    parser.add_argument(
        "--tr_epochs",
        type=int,
        default=100,
        help="Number of epochs for training"
    )

    parser.add_argument(
        "--train_subset",
        type=float,
        default=1.0,
        help="Fraction of training data to use"
    )

    parser.add_argument(
        "--test_subset",
        type=float,
        default=1.0,
        help="Fraction of test data to use"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--attack",
        type=str,
        # required=True,
        choices=["clean", "pgdl2", "pgdlinf", "autoattack"]
    )

    parser.add_argument(
        "--compute_lip", 
        action="store_true"
    )


    return parser.parse_args()

def checkpoint_exists(path):
    return os.path.isfile(path)

def build_optimizer(model, optimizer_name, hyperparams):
    if optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=hyperparams["lr"],
            momentum=hyperparams.get("momentum", 0.0),
            weight_decay=hyperparams.get("weight_decay", 0.0),
            nesterov=hyperparams.get("nesterov", False)
        )

    elif optimizer_name.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams.get("weight_decay", 0.0)
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def search_run(
    model,
    train_loader,
    val_loader,
    device,
    optimizer_name,
    hyperparams,
    epochs
):
    model.to(device)

    optimizer = build_optimizer(model, optimizer_name, hyperparams)

    best_val_acc = 0.0

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        val_metrics = model.evaluate(val_loader, device)
        best_val_acc = max(best_val_acc, val_metrics["accuracy"])

    return {"best_val_accuracy": best_val_acc}

def is_final_checkpoint(path: str, device) -> bool:
    if not os.path.exists(path):
        return False
    try:
        ckpt = torch.load(path, map_location=device)
        return ckpt.get("is_final", False) is True
    except Exception:
        return False

def _make_model_dataset_key(dataset, model_name):
    return f"{dataset}::{model_name}"

def load_fixed_hyperparams(dataset, model_name, registry_path="best_hyperparams.json"):
    if not os.path.exists(registry_path):
        return None

    with open(registry_path, "r") as f:
        registry = json.load(f)

    return registry.get(_make_model_dataset_key(dataset, model_name), None)

def save_fixed_hyperparams(
    dataset,
    model_name,
    best_config,
    capacities,
    registry_path="best_hyperparams.json"
):
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    registry[_make_model_dataset_key(dataset, model_name)] = {
        **best_config,
        "searched_capacities": capacities
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

def run_hyperparam_search(
    model_class,
    capacities,
    train_loader,
    val_loader,
    input_size,
    in_channels,
    device,
    hp_epochs
):
    candidates = [
        ("sgd", {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4}),
        ("sgd", {"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4}),
        ("adam", {"lr": 1e-3, "weight_decay": 1e-4}),
    ]

    best_val_acc = -float("inf")
    best_cfg = None

    for opt_name, hparams in candidates:
        print(f"Testing hyperparams: {opt_name} | {hparams}")

        for cap in capacities:
            print(f"  Capacity={cap}")

            model = model_class(
                capacity=cap,
                num_classes=10,
                input_size = input_size,
                in_channels = in_channels
            ).to(device)

            print(f"Searching model: {cap} with {model.num_parameters()} params")

            optimizer = build_optimizer(model, opt_name, hparams)

            local_best_acc = 0.0

            for epoch in range(hp_epochs):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                    loss.backward()
                    optimizer.step()

                val_metrics = model.evaluate(val_loader, device)
                local_best_acc = max(local_best_acc, val_metrics["accuracy"])

            print(f"    best val acc={local_best_acc:.4f}")

            if local_best_acc > best_val_acc:
                best_val_acc = local_best_acc
                best_cfg = {
                    "optimizer": opt_name,
                    "hyperparams": hparams
                }

    print("✓ Selected hyperparameters based on best val accuracy")
    print(best_cfg)
    return best_cfg

def _make_result_key(dataset, model_name, capacity):
    return f"{dataset}::{model_name}::cap{capacity}"

def collect_full_test_set(test_loader, device):
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
    x_all = torch.cat(xs).to(device)
    y_all = torch.cat(ys).to(device)
    return x_all, y_all

def save_experiment_result(
    dataset,
    model_name,
    capacity,
    result_dict,
    results_path="results.json"
):
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    key = _make_result_key(dataset, model_name, capacity)
    registry[key] = result_dict

    with open(results_path, "w") as f:
        json.dump(registry, f, indent=2)


def jacobian_spectral_norm(model, x, n_iters=10):
    """
    Estimates ||J_f(x)||_2 using power iteration
    x: single batch (requires grad)
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    batch_size = x.size(0)
    num_classes = model(x).size(1)

    # random vector in output space
    v = torch.randn(batch_size, num_classes, device=x.device)
    v = v / (v.norm(dim=1, keepdim=True) + 1e-12)

    for _ in range(n_iters):
        # J^T v
        logits = model(x)
        JTv = torch.autograd.grad(
            outputs=logits,
            inputs=x,
            grad_outputs=v,
            retain_graph=True,
            create_graph=True
        )[0]

        # J (J^T v)
        JJTv = torch.autograd.grad(
            outputs=JTv,
            inputs=logits,
            grad_outputs=torch.ones_like(JTv),
            retain_graph=True
        )[0]

        v = JJTv
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)

    # Rayleigh quotient approximation
    logits = model(x)
    JTv = torch.autograd.grad(logits, x, v, retain_graph=False)[0]
    spec_norm = JTv.view(batch_size, -1).norm(p=2, dim=1)

    return spec_norm.mean().item()


def get_normalized_bounds(args):
    if args.dataset == "mnist":
        mean = torch.tensor((0.1307,), device=args.device).view(1,1,1,1)
        std  = torch.tensor((0.3081,), device=args.device).view(1,1,1,1)
    else:
        mean = torch.tensor((0.4916,0.4824,0.4467), device=args.device).view(1,3,1,1)
        std  = torch.tensor((0.0299,0.0295,0.0318), device=args.device).view(1,3,1,1)

    # lower = (0 - mean) / std
    # upper = (1 - mean) / std

    lower = float(((0 - mean) / std).min().item())
    upper = float(((1 - mean) / std).max().item())

    return lower, upper



def wrap_model_secml(model, input_shape, batch_size):
    return CClassifierPyTorch(
        model=model,
        input_shape=input_shape,
        preprocess=None,
        batch_size=batch_size,
        softmax_outputs=True,
    )

def get_pgd_attack_hyperparams(dataset_name):

    lb, ub = 0, 1  # Bounds of the attack space. Can be set to `None` for
    # unbounded
    y_target = None  # None if `error-generic` or a class label for `error-specific`

    # Should be chosen depending on the optimization problem
    if dataset_name == 'mnist':
            solver_params = {
                'eta': 0.5,
                'eta_min': 0.1,
                'eta_max': None,
                'max_iter': 100,
                'eps': 1e-3
            }
    elif dataset_name == 'cifar10':
            solver_params = {
                'eta': 0.1,
                'eta_min': 1e-4,
                'eta_max': None,
                'max_iter': 300,
                'eps': 1e-4
            }


    return solver_params, lb, ub, y_target


def pytorch_ds_to_secml_ds(ds_loader, batch_size):
    """
    Get a pytorch dataset loader and return a CDataset (the secml data
    structure for datasets)

    Input:
    ds_loader: dict 
        dataset.train_loader or dataset.test_loader
    batch_size: int
        batchsize for secml framework
    """
    secml_ds = None
    img_size = None
    for img, y in ds_loader:
        
        if img_size is None:
            # check the image size
            first_img = img[0,:]
            img_size = first_img.reshape(-1)
            img_size = img_size.numpy().size
            # img_size = first_img.numpy().size

        # the pytorch images have 3 dimensions whereas secml work with
        # flattened array.
        secml_img = CArray(img.reshape(batch_size,img_size).numpy())
        secml_y =  CArray(y.numpy())

        current_sample = CDataset(secml_img, secml_y)
        if secml_ds is None:
            secml_ds =  current_sample
        else:
            secml_ds = secml_ds.append(current_sample)

    return secml_ds



