import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.ticker import FuncFormatter

# -----------------------
# I/O utilities
# -----------------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# -----------------------
# Tick Helpers
# -----------------------

def select_log_ticks(values, min_ticks=3, max_ticks=7):
    """
    Select reasonable ticks for log-scale x-axis.
    Always includes first, middle, last.
    """
    values = np.array(values)

    if len(values) <= max_ticks:
        return values.tolist()

    first = values[0]
    last = values[-1]
    middle = values[len(values) // 2]

    ticks = np.unique([first, middle, last])

    return ticks.tolist(), middle


def sci_notation_1dec(x, _):
    """
    Format ticks as 1 decimal scientific notation.
    Example: 3300 -> 3.3×10³
    """
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(x)))
    mant = x / (10 ** exp)
    return rf"${mant:.1f}\times10^{{{exp}}}$"

# -----------------------
# AUC
# -----------------------

def compute_auc(eps, err):
    return np.trapz(err, eps)


# -----------------------
# Data extraction
# -----------------------

def extract_train_base_performance(train_results, dataset, model):
    """
    Returns lists sorted by number of parameters:
        params, train_error, cap_list
    """
    records = []

    for key, v in train_results.items():
        if not key.startswith(f"{dataset}::{model}::cap"):
            continue

        records.append({
            "cap": v["capacity"],
            "params": v["num_parameters"],
            "train_error": v["train_error"],
        })

    records = sorted(records, key=lambda x: x["params"])

    params = [r["params"] for r in records]
    train_error = [r["train_error"] for r in records]
    caps = [r["cap"] for r in records]

    return params, train_error, caps


def extract_test_base_performance(test_results, caps):
    """
    Align test error rates to the given capacity order.
    """
    test_error = []

    for cap in caps:
        cap_key = f"cap_{cap}"
        test_error.append(
            test_results["results"][cap_key]["clean"]["error_rate"]
        )

    return test_error


# -----------------------
# Plotting
# -----------------------

def plot_base_performance(
    dataset,
    model,
    train_results_path,
    test_results_path,
    save_dir="plots",
    regime_boundary=None
):
    train_results = load_json(train_results_path)
    test_results = load_json(test_results_path)

    params, train_error, caps = extract_train_base_performance(
        train_results, dataset, model
    )
    test_error = extract_test_base_performance(test_results, caps)

    plt.figure(figsize=(7, 4.5))

    plt.semilogx(params, train_error, marker="o", label="train")
    plt.semilogx(params, test_error, marker="o", label="test")

    xticks, _ = select_log_ticks(params)
    # regime_boundary = interp
    # Always safe: add line only if value is provided
    if regime_boundary is not None:
        interp = params[regime_boundary-1]
        plt.axvline(
            interp,
            linestyle="--",
            color="black",
            linewidth=1.5
        )

    plt.xlabel("Number of parameters", fontsize=16)
    plt.ylabel("Error rate", fontsize=14)
    plt.title(f"Performance on benign samples {dataset.upper()}: {model.upper()}", fontsize=18)

    # Explicit ticks + rotation
    
    plt.xticks(xticks, fontsize=14)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(sci_notation_1dec))


    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    Path(save_dir).mkdir(exist_ok=True)
    save_path = Path(save_dir) / f"{dataset}_{model}_base_performance.png"

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"[SAVED] {save_path}")


def plot_sec(
    dataset,
    model,
    attack,
    results_dir="results",
    save_dir="plots",
):
    """
    Plots Security Evaluation Curves (SEC):
    x-axis: epsilon
    y-axis: error rate
    One curve per capacity
    """

    if attack == "pgdl2":
        title = "PGD-L2"
    elif attack == "autoattack":
        title = "AutoAttack"
    elif attack == "pgdlinf":
        title = "PGD-L∞"
    else:
        title = attack.upper()

    # ---------- Load clean results (ε = 0) ----------
    clean_file = f"{results_dir}/sec_eval_{dataset}_{model}_clean_results.json"
    with open(clean_file, "r") as f:
        clean_data = json.load(f)["results"]

    # ---------- Load attack results ----------
    attack_file = f"{results_dir}/sec_eval_{dataset}_{model}_{attack}_results.json"
    with open(attack_file, "r") as f:
        attack_data = json.load(f)["results"]

    # ---------- Sort capacities ----------
    caps = sorted(clean_data.keys(), key=lambda x: int(x.split("_")[1]))

    plt.figure(figsize=(8, 6))

    for idx, cap in enumerate(caps, start=1):
        # ε = 0
        eps = [0.0]
        err = [clean_data[cap]["clean"]["error_rate"]]

        # attacked eps
        attack_eps = sorted(
            float(k.replace("eps_", ""))
            for k in attack_data[cap].keys()
        )

        for e in attack_eps:
            eps.append(e)
            err.append(1- (attack_data[cap][f"eps_{e}"]))

        eps = np.array(eps)
        err = np.array(err)

        auc = compute_auc(eps, err)

        plt.plot(
            eps,
            err,
            marker="o",
            linewidth=2,
            label=f"M{idx} (AUC={auc:.3f})"
        )

    # ---------- Styling ----------
    plt.xlabel("Perturbation Budget $\epsilon$", fontsize=16)
    # plt.ylabel("Error rate", fontsize=16)
    plt.title(
        f"SEC for {dataset.upper()}-{model.upper()}: {title}",
        fontsize=18,
    )

    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    Path(save_dir).mkdir(exist_ok=True)
    save_path = Path(save_dir) / f"sec_{dataset}_{model}_{attack}.png"

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    print(f"[SAVED] {save_path}")

# -----------------------
# Lipschitz plotting
# -----------------------


def plot_lipschitz(
    dataset: str,
    model: str,
    results_dir: str = "results",
    regime_boundary: int | None = None,
    save_dir: str = "plots"
):
    """
    Plot Lipschitz upper bound vs model capacity.
    """
    lipschitz_path = (
        Path(results_dir)
        / f"lipschitz_{dataset}_{model}.json"
    )

    if not lipschitz_path.exists():
        raise FileNotFoundError(f"Missing Lipschitz file: {lipschitz_path}")

    with open(lipschitz_path, "r") as f:
        data = json.load(f)

    # Sort by capacity (keys are strings)
    capacities = sorted(int(k) for k in data.keys())
    lipschitz_vals = [
        data[str(c)]["lipschitz_upper_bound"] for c in capacities
    ]

    plt.figure(figsize=(6, 4))
    plt.plot(capacities, lipschitz_vals, marker="o")
    plt.yscale("log")

    plt.xlabel("Model capacity")
    plt.ylabel("Lipschitz upper bound (log scale)")
    plt.title(f"Lipschitz scaling — {dataset.upper()} - {model.upper()}")

    if regime_boundary is not None:
        plt.axvline(
            regime_boundary,
            linestyle="--",
            color="gray",
            label="Regime boundary"
        )
        plt.legend()
    
    Path(save_dir).mkdir(exist_ok=True)
    save_path = Path(save_dir) / f"lipschitz_{dataset}_{model}.png"

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    print(f"[SAVED] {save_path}")

# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument(
        "--plot_type",
        required=True,
        choices=["base", "sec", "lipschitz"]
    )
    parser.add_argument(
        "--attack",
        type=str,
        choices=["pgdl2", "autoattack", "pgdlinf"]
    )
    parser.add_argument(
        "--train_results",
        default="train_results.json"
    )
    parser.add_argument(
        "--results_dir",
        default="results"
    )
    parser.add_argument(
        "--regime_boundary",
        type=int,
        default=None,
        help="Optional x-value for vertical dashed line"
    )

    args = parser.parse_args()

    test_results_path = (
        Path(args.results_dir)
        / f"sec_eval_{args.dataset}_{args.model}_clean_results.json"
    )

    if args.plot_type == "base":
        plot_base_performance(
            dataset=args.dataset,
            model=args.model,
            train_results_path=args.train_results,
            test_results_path=test_results_path,
            regime_boundary=args.regime_boundary
        )
    elif args.plot_type == "sec":
        plot_sec(
            dataset=args.dataset,
            model=args.model,
            attack=args.attack
        )
    elif args.plot_type == "lipschitz":
        plot_lipschitz(
            dataset=args.dataset,
            model=args.model,
            results_dir="results",
            regime_boundary=args.regime_boundary
    )


if __name__ == "__main__":
    main()
