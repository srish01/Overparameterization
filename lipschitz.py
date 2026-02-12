import argparse
import os
import json
import torch

from utils.models import *

from lipestime.src.lipestime import LipUpperBound

# -----------------------------
# Import your model definitions
# -----------------------------



MODEL_REGISTRY = {
    "cnn": ScalableCNN,
    "mlp": ScalableMLP,
    "resnet": ScalableResNet,
}

def compute_lipschitz_upper_bound(model, in_channels, input_size):
    """
    Uses LiPeSTiME to compute an upper bound.
    """
    estimator = LipUpperBound(model=model)
    C, H, W = in_channels, input_size, input_size

    _ = estimator.propagate(torch.randn(1, C, H, W))
    final_lip = None
    lip=1.0
    
    for node in estimator.graph.nodes:
        if hasattr(node, 'lip'):
            final_lip = node.lip
            print(
                f"Node: {node.name}, "
                f"op: {node.op}, "
                f"target: {node.target}, "
                f"lip: {final_lip:.3e}"
            )

    assert final_lip is not None, "No Lipschitz value found in graph."
    
    return float(final_lip)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "mnist":
        inp_size, in_channels, num_classes = 28, 1, 10
        capacities = range(1, 11)
    elif args.dataset == "cifar10":
        inp_size, in_channels, num_classes = 32, 3, 10
        capacities = [1, 2, 4, 6, 8, 10, 16, 20] #, 24, 28]

    # capacities = args.capacities
    results = {}

    for cap in capacities:

        model_cls = MODEL_REGISTRY[args.model]
        model = model_cls(capacity=cap, num_classes=num_classes,
                          input_size=inp_size, in_channels=in_channels)
        ckpt_path = f"checkpoints/{args.dataset}_{model_cls.__name__}_{cap}.pt"

        if not os.path.exists(ckpt_path):
            print(f"[WARN] Missing checkpoint: {ckpt_path}")
            continue

        print(f"▶ Processing capacity {cap}")

        # model.load(ckpt_path, cap=cap, map_location=device)
        model.load(ckpt_path, cap=cap)
        model.eval()

        L = compute_lipschitz_upper_bound(model, in_channels, inp_size)

        results[cap] = {
            "lipschitz_upper_bound": L
        }

        print(f"  Lipschitz upper bound: {L:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    out_file = os.path.join(
        args.output_dir,
        f"lipschitz_{args.dataset}_{args.model}.json"
    )

    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✓ Results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mnist", "cifar10"])
    parser.add_argument("--model", type=str, required=True,
                        choices=["cnn", "mlp", "resnet"])
    # parser.add_argument("--capacities", type=int, nargs="+", required=True,
    #                     help="List of model capacities (e.g. 16 32 64 ...)")
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()
    main(args)
