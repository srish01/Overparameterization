import argparse
import os
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
import numpy as np

# -------------------------------
# IMPORT MODELS + SEcML ATTACK
# -------------------------------
from utils.models import *
from utils.utils import *
from utils.attacks import eval_secml_pgd_l2   # adjust import if needed
from utils.dataloader import *


# -------------------------------
# ARGUMENTS
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["cnn", "mlp"], required=True)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--digit", type=int, default=4)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--output_dir", type=str, default="./adv_examples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# -------------------------------
# LOAD MNIST (single digit)
# -------------------------------
def load_mnist_digit_loader(digit, num_samples, batch_size):
    transform = T.ToTensor()
    dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    indices = [i for i, (_, y) in enumerate(dataset) if y == digit][:num_samples]
    subset = Subset(dataset, indices)

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return loader


# -------------------------------
# LOAD MODEL
# -------------------------------
def load_model(args, capacity, device):

    inp_size, in_channels, num_classes = 28, 1, 10

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(
        capacity=capacity,
        input_size=inp_size,
        in_channels=in_channels,
        num_classes=num_classes,
    ).to(device)

    checkpoint_path = f"{args.checkpoints_dir}/{args.dataset}_{model_cls.__name__}_{capacity}.pt"
    print(f"Loading model from: {checkpoint_path}")
    model.load(checkpoint_path, cap=capacity, map_location=device)

    model.eval()
    return model, num_classes


# -------------------------------
# MAIN
# -------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # capacities
    small_cap = 1
    large_cap = 10

    EPS_REGISTRY = {
        ("mnist", "mlp"):   np.linspace(0.1, 3.0, 5),
        ("mnist", "cnn"):   np.linspace(0.01, 1.5, 5),
        ("cifar10", "resnet"): np.linspace(0.1, 6.0, 5)
        }

    train_loader, _, _ = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        train_subset=0.1,
        seed=args.seed,
        model_name=args.model,
        ds_normalization=False
    )

    test_loader = load_mnist_digit_loader(
        args.digit, args.num_samples, args.batch_size
    )
    # x_clean_all, y_clean_all = collect_full_test_set(test_loader, device)

    solver_params, norm, lb, ub, y_target = get_pgd_attack_hyperparams(args.dataset)
            
    # eps_list = EPS_REGISTRY[(args.dataset, args.model)]
    eps_list = [args.epsilon]
    # Load clean samples ONCE (order preserved)
    x_clean, y_clean = next(iter(test_loader))
    x_clean = x_clean.to(device)
    y_clean = y_clean.to(device)

    for cap in [small_cap, large_cap]:

        model, num_classes = load_model(args, cap, device)

        stats = eval_secml_pgd_l2(
            args=args,
            model=model,
            num_classes=num_classes,
            eps_list=eps_list,
            lower=lb,
            upper=ub,
            y_target=9,
            train_loader=train_loader,
            test_loader=test_loader,
            solver_params=solver_params,
            save_adv_ds = True,
            steps=100,
        )

        # -------------------------------
        # EXTRACT ADVERSARIAL SAMPLES
        # -------------------------------
        adv_ds = stats.sec_eval_data.adv_ds[0]

        x_adv = torch.from_numpy(adv_ds.X.tondarray()).float().to(device)
        x_adv = x_adv.view(-1, 1, 28, 28)

        y_attack = adv_ds.Y.tondarray().astype(int)

        assert x_adv.size(0) == x_clean.size(0), \
            "Mismatch between clean and adversarial sample counts"


        # corresponding clean samples (CORRECT alignment)
        # x_clean = x_clean_all[idx_adv]
        # y_clean = y_clean_all[idx_adv]


        # -------------------------------
        # MODEL PREDICTIONS ON ADV IMAGES
        # -------------------------------
        with torch.no_grad():
            logits_adv = model(x_adv)
            y_pred_adv = logits_adv.argmax(dim=1)

        # -------------------------------
        # SAVE IMAGES + LABELS
        # -------------------------------
        for i in range(x_adv.size(0)):
            save_image(
                x_clean[i],
                os.path.join(
                    args.output_dir,
                    f"clean_digit{args.digit}_{i}.png",
                ),
            )

            save_image(
                x_adv[i],
                os.path.join(
                    args.output_dir,
                    (
                        f"adv_img_model_{args.model}"
                        f"_cap_{cap}"
                        f"_eps{args.epsilon}"
                        f"_true{y_clean[i].item()}"
                        f"_pred{y_pred_adv[i].item()}"
                        f"_idx{i}.png"
                    ),
                ),
            )

        print("Saved adversarial examples to:", args.output_dir)


if __name__ == "__main__":
    main()
