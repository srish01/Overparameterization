import torch
import torch.nn.functional as F
import numpy as np
from secml.adv.attacks.evasion import CAttackEvasionPGDLS
from secml.adv.seceval import CSecEval, CSecEvalData
from secml.array import CArray
from utils.utils import *

import foolbox as fb

# def run_autoattack_l2(attacker, x, y, is_mlp=False):
#     """
#     attacker : AutoAttack instance already created for (model, eps)
#     model    : original model (MLP/CNN/ResNet)
#     x        : batch from loader
#     """

#     # ---- AutoAttack generates adversarial IMAGES ----
#     x_adv = attacker.run_standard_evaluation(x, y, bs=len(x))

#     return x_adv

from autoattack import AutoAttack



def eval_autoattack_l2(args, model, test_loader, epsilons, device, batch_size):
    
    model.eval()
    results = {}

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0).to(device)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0).to(device)


    for eps in epsilons:

        attacker = AutoAttack(
            model,
            norm="L2",
            eps=eps,
            version="standard",
            seed=args.seed,
            device=device
        )

        print(f"---- Running AutoAttack with eps={eps} ----")
        x_adv, y_adv = attacker.run_standard_evaluation(x_test, y_test, bs=batch_size, return_labels=True)
        x_adv = x_adv.to(next(model.parameters()).device)
        

        # check if y_adv is same as preds
        with torch.no_grad():
            preds = model(x_adv).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

        results[eps] = acc
        print(f"     Accuracy @ eps={eps}: {acc:.4f}")
    
    return results
   

def eval_secml_pgd(
    args,
    model,
    num_classes,
    eps_list,
    lower,
    upper,
    norm,
    y_target,
    train_loader,
    test_loader,
    solver_params,
    save_adv_ds,
    steps=100
):
    model.eval()

    x, y = collect_full_test_set(test_loader, args.device)

    # Torch → numpy
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()


    tr_dataset = pytorch_ds_to_secml_ds(train_loader, args.batch_size)
    ts_dataset = pytorch_ds_to_secml_ds(test_loader, args.batch_size)

    clf = wrap_model_secml(
        model,
        input_shape=x_np.shape[1:],
        batch_size=args.batch_size,
    )

    clf._trained = True
    clf._classes = CArray.arange(num_classes)   # MNIST / CIFAR-10
    clf._n_features = int(np.prod(x_np.shape[1:]))

    attack = CAttackEvasionPGDLS(
        classifier=clf,
        double_init_ds=tr_dataset,
        double_init=True,
        distance=norm,
        dmax=0,
        lb=lower,
        ub=upper,
        y_target=y_target,
        # lb= CArray(lower.detach().cpu().numpy().reshape(-1)),
        # ub=CArray(upper.detach().cpu().numpy().reshape(-1)),
        solver_params=solver_params,
    )

    sec_eval = CSecEval(
        attack=attack,
        param_name="dmax",
        param_values=eps_list,
        save_adv_ds=save_adv_ds,
    )

    sec_eval.run_sec_eval(ts_dataset)
    
    return sec_eval




def eval_foolbox_pgd_linf(
    model,
    test_loader,
    epsilons,
    device,
    steps=40,
    rel_stepsize=0.01
):
    """
    Foolbox Linf PGD evaluation.
    Inputs assumed in [0,1].
    """

    fmodel = fb.PyTorchModel(
        model,
        bounds=(0.0, 1.0),
        device=device
    )

    attack = fb.attacks.LinfProjectedGradientDescentAttack(
        steps=steps,
        rel_stepsize=rel_stepsize,
        random_start=True
    )

    acc_per_eps = {eps: [] for eps in epsilons}

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        _, _, success = attack(
            fmodel,
            x,
            y,
            epsilons=epsilons
        )

        # success: [len(eps), batch]
        for i, eps in enumerate(epsilons):
            acc = 1.0 - success[i].float().mean().item()
            acc_per_eps[eps].append(acc)

    return {
        eps: sum(v) / len(v)
        for eps, v in acc_per_eps.items()
    }


    