import torch
import torch.nn as nn
from itertools import product
import argparse
from utils.attacks import eval_autoattack_l2, eval_secml_pgd, eval_foolbox_pgd_linf
from utils.dataloader import *
from utils.models import *
from tqdm import tqdm
from utils.utils import *

from copy import deepcopy
import math

from secml.ml.peval.metrics import CMetricAccuracy

"""
This is the main entry point for training and evaluating models. It supports three modes of operation:
1. Hyperparameter search: runs a grid search over optimizers and learning rates to find the best configuration for each (dataset, model) pair. Saves results to fixed_hyperparams.json. 
2. Training: trains models for each capacity using the best hyperparameters found in the search phase. Saves checkpoints and training results to train_results.json.
3. Running attacks: evaluates trained models against specified adversarial attacks (PGD-L2, PGD-L∞, AutoAttack) and saves results to results/sec_eval_{dataset}_{model}_{attack}_results.json. This mode also gets clean accuracy (ε=0) for reference.

"""

HYPERPARAM_GRID = {
    "sgd": {
        "lr": [0.1, 0.05, 0.01],
        "momentum": [0.9],
        "weight_decay": [5e-4, 1e-4]
    },
    "adam": {
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-4, 0.0]
    }
}

MODEL_REGISTRY = {
    "resnet": ScalableResNet,
    "cnn": ScalableCNN,
    "mlp": ScalableMLP,
}

EPS_REGISTRY = {
        "pgdl2":{
            ("mnist", "mlp"):   np.linspace(0.1, 3.0, 5),
            ("mnist", "cnn"):   np.linspace(0.01, 1.0, 5),
            ("cifar10", "resnet"): np.linspace(0.005, 0.3, 5)
        },
        "pgdlinf": {
            ("mnist", "mlp"):   np.linspace(0.01, 0.15, 5),
            ("mnist", "cnn"):   np.linspace(0.01, 0.07, 5),
            ("cifar10", "resnet"): np.linspace(0.0001, 0.01, 5)
        },
        "autoattack": {
            ("mnist", "mlp"):   np.linspace(0.1, 2.0, 5),
            ("mnist", "cnn"):   np.linspace(0.01, 1.0, 5),
            ("cifar10", "resnet"): np.linspace(0.005, 0.25, 5)
        }
}

    
def train_network(
    model,
    input_size,
    in_channels,
    model_kwargs: dict,
    train_loader,
    val_loader,
    device,
    optimizer_name: str,
    hyperparams: dict,
    max_epochs: int,
    criterion=nn.CrossEntropyLoss(),
    save_path: str = None
):
    
    """
    Trains a model using parameters from the hyperparameter grid search 

    Args:
        model_class: ScalableResNet / ScalableCNN / ScalableMLP
        model_kwargs: arguments to instantiate the model
        train_loader, val_loader: CIFAR-10 loaders
        device: torch.device
        max_epochs: upper bound on training epochs
        optimizer_choices: subset of ["sgd", "adam"]
        save_dir: where to save interpolated model

    Returns:
        dict with interpolation results
    """
    # -------------------------------------------------
    # Checkpoint exists → skip training
    # -------------------------------------------------
    if save_path is not None and is_final_checkpoint(save_path, device):
        print(f"[SKIP] Final checkpoint exists: {save_path}")
        return {
            "status": "skipped",
            "checkpoint": save_path
        }

    model = model_class(input_size = inp_size, in_channels = in_channels,**model_kwargs).to(device)
    optimizer = build_optimizer(model, optimizer_name=optimizer_name, hyperparams=hyperparams)
    
    best_val_error = math.inf
    best_epoch = -1
    best_metrics = None
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            train_acc = correct / total
            pbar.set_postfix(
                train_loss=f"{loss_sum / len(train_loader):.4f}",
                train_acc=f"{train_acc:.4f}"
            )

        train_loss = loss_sum / len(train_loader)
        train_error = 1.0 - train_acc

        # --------------------
        # Validation
        # --------------------
        val_metrics = model.evaluate(val_loader, device, criterion)
        val_acc = val_metrics["accuracy"]
        val_error = val_metrics["error"]

        print(
            f"Epoch {epoch:04d} | "
            f"Train acc {train_acc:.4f}, error {train_error:.4f} | "
            f"Val acc {val_acc:.4f}, error {val_error:.4f}"
        )

        # --------------------
        # Best model tracking
        # --------------------
        if val_error < best_val_error:
            best_val_error = val_error
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            best_metrics = {
                "epoch": best_epoch,
                "train_acc": train_acc,
                "train_error": train_error,
                "val_acc": val_acc,
                "val_error": val_error,
                "num_parameters": model.num_parameters()
            }
            if save_path is not None:
                print(f"Saving checkpoint at {save_path}")
                model.save(save_path)

        # --------------------
        # Optional early stop at interpolation
        # --------------------
        if val_error == 0.0:
            print("✔ Interpolation achieved (zero validation error)")
            break
    # --------------------
    # Restore best model and save it with is_final flag
    # --------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path is not None:
            model.save(save_path, is_final=True)

    return {
        "status": "finished",
        "best_epoch": best_epoch,
        "best_val_error": best_val_error,
        "best_metrics": best_metrics,
        "checkpoint": save_path
    }


@torch.no_grad()
def eval_clean(model, x, y):
    model.eval()
    preds = model(x).argmax(dim=1)
    acc = (preds == y).float().mean().item()
    return {
        "accuracy": acc,
        "error_rate": 1.0 - acc
    }

if __name__ == "__main__":
    args = parse_args()

    set_global_seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    if args.dataset == "mnist":
        inp_size, in_channels, num_classes = 28, 1, 10
        capacities = range(1, 11)
        if args.model =='mlp':
            tr_epochs = 500
        elif args.model == 'cnn':
            tr_epochs = 200
    elif args.dataset == "cifar10":
        inp_size, in_channels, num_classes = 32, 3, 10
        capacities = [1, 2, 4, 6, 8, 10, 16, 20, 24, 26] # 22, 28
        # capacities = [22]
        if args.model =='resnet':
            tr_epochs = 1000
    
    # tr_epochs = args.tr_epochs
    model_class = MODEL_REGISTRY[args.model]

    ds_normalization = False

    # for cap in args.capacity:
    print("=" * 80)
    
    # ------------------------------
    # Data loaders
    # ------------------------------
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        seed=args.seed,
        model_name = args.model,
        ds_normalization=ds_normalization
    )
        
    # ------------------------------
    # Load or search hyperparameters
    # ------------------------------
    if args.mode == "search":
        print(f"HP SEARCH: model={args.model}, dataset={args.dataset}")

        small_train, small_val, _ = get_data_loaders(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            train_subset=0.2,
            seed=args.seed,
            model_name = args.model
        )
            
        best_cfg = run_hyperparam_search(
            model_class = model_class,
            capacities = args.capacity,
            train_loader = small_train,
            val_loader = small_val,
            hp_epochs = args.hp_epochs,
            input_size = inp_size,
            in_channels = in_channels,
            device = device
        )

        save_fixed_hyperparams(
            args.dataset,
            args.model,
            best_cfg,
            args.capacity
        )
        
    elif args.mode == "train":
        print(f"TRAINING: model={args.model}, dataset={args.dataset}")
        
        fixed_cfg = load_fixed_hyperparams(args.dataset, args.model)
        assert fixed_cfg is not None, "Run hyperparameter search first."

        for i, cap in enumerate(args.capacity):
            ckpt_path = f"checkpoints/{args.dataset}_{model_class.__name__}_{cap}.pt"

            if checkpoint_exists(ckpt_path):
                print(f"[SKIP] Checkpoint exists for capacity={cap}")
                continue
            
            model = model_class(capacity=args.capacity[i], num_classes=10)
            print(f"Training model: {cap} with {model.num_parameters()} params")
            print(f"for {tr_epochs} epochs")
            
            result = train_network(
                model = model,
                input_size = inp_size,
                in_channels = in_channels,
                model_kwargs={"capacity": cap, "num_classes": 10},
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                optimizer_name=fixed_cfg["optimizer"],
                hyperparams=fixed_cfg["hyperparams"],
                max_epochs=tr_epochs,
                save_path=ckpt_path
            )

            train_result_payload = {
                "capacity": cap,
                "num_parameters": result["best_metrics"]["num_parameters"],
                "best_epoch": result["best_epoch"],
                "train_acc": result["best_metrics"]["train_acc"],
                "train_error": result["best_metrics"]["train_error"],
                "val_acc": result["best_metrics"]["val_acc"],
                "val_error": result["best_metrics"]["val_error"],
                "checkpoint": result["checkpoint"]
            }
            

            save_experiment_result(
                dataset=args.dataset,
                model_name=args.model,
                capacity=cap,
                result_dict=train_result_payload,
                results_path="train_results.json"
            )

    elif args.mode == "run_attacks":
        print(f"RUN ATTACKS: model={args.model}, dataset={args.dataset} for {args.attack}")
        assert args.attack is not None, "Specify an attack to run in this mode."
        assert args.test_subset is not None, "Specify test subset fraction for evaluation - NOT ADVISED TO USE FULL TEST SET FOR ADV EVAL (long runtime)"
        output_path = f"results/sec_eval_{args.dataset}_{args.model}_{args.attack}_results.json"

        x_test, y_test = collect_full_test_set(test_loader, device)

        results = {
            "dataset": args.dataset,
            "model": args.model,
            "attack": args.attack,
            "results": {}
        }

        
        # print("\nRunning test evaluation from saved checkpoints")

        for i, cap in enumerate(capacities):

            print(f"[INFO] Evaluating cap={cap}")

            model = model_class(capacity=cap, num_classes=num_classes,
                          input_size=inp_size, in_channels=in_channels).to(device)

            ckpt_path = f"checkpoints/{args.dataset}_{model_class.__name__}_{cap}.pt"
            print(f"\t Loading checkpoint from {ckpt_path}")
            print(f"\t {model.num_parameters()} params")
            model.load(ckpt_path, cap=cap, map_location=device)
            model.eval()

            if not checkpoint_exists(ckpt_path):
                print(f"[SKIP] No checkpoint for capacity={cap}")
                continue
            cap_key = f"cap_{cap}"
            results["results"][cap_key] = {}

            # ---------------- CLEAN ----------------
            if args.attack == "clean":
                stats = eval_clean(model, x_test, y_test)
                results["results"][cap_key]["clean"] = stats
                continue

            # ---------------- ADVERSARIAL ----------------
            elif args.attack == "autoattack":
                epsilons = EPS_REGISTRY[args.attack][(args.dataset, args.model)]
                    
                stats = eval_autoattack_l2(
                    args,
                    model=model,
                    test_loader=test_loader,
                    epsilons=epsilons,
                    device=device,
                    batch_size=args.batch_size
                )
                for eps, acc in zip(epsilons, stats.values()):
                    results["results"][cap_key][f"eps_{eps}"] = acc

            elif args.attack == "pgdl2":
                epsilons = EPS_REGISTRY[args.attack][(args.dataset, args.model)]
                norm="l2"
                metric = CMetricAccuracy()
                print(f"[INFO] Running PGD-L2 for epsilons = {epsilons}")
                solver_params, lb, ub, y_target = get_pgd_attack_hyperparams(args.dataset)
                # lower, upper = get_normalized_bounds(args)
                stats = eval_secml_pgd(
                    args,
                    model=model,
                    num_classes=num_classes,
                    eps_list=epsilons,         # pass full list
                    lower=lb,
                    upper=ub,
                    norm=norm,
                    y_target=y_target,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    solver_params=solver_params,
                    save_adv_ds=False
                )

                # stats.save_data(f"results/sec_eval_{args.dataset}_{args.model}_{cap{cap}}.gz")
                res = stats.sec_eval_data
                epsilons = res.param_values
                y_true = res.Y
                att_pred = res.Y_pred

                for i, eps in enumerate(epsilons):
                    results["results"][cap_key][f"eps_{eps}"] = metric.performance_score(y_true=y_true, y_pred=att_pred[i])


            elif args.attack == "pgdlinf":
                epsilons = EPS_REGISTRY[args.attack][(args.dataset, args.model)]
                print(f"[INFO] Running Foolbox PGD-L∞ for epsilons = {epsilons}")

                stats = eval_foolbox_pgd_linf(
                    model=model,
                    test_loader=test_loader,
                    epsilons=epsilons,
                    device=device,
                    steps=100 if args.dataset == "mnist" else 150
                )

                for eps, acc in stats.items():
                    results["results"][cap_key][f"eps_{eps}"] = acc

        # ---------------- Save ----------------
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[✓] Results saved to {output_path}")


