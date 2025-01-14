"""
gaussian2swissroll.py

An example script that trains CPAT, BProj, and SCONES to learn an optimal-like
coupling between a 2D Gaussian source and a 2D Swiss Roll target, but now with:
  - multiple hidden_dims
  - multiple lambda (lmbda) values
  - multiple SCONES sampling learning rates
We also rename "sinkhorn" to "wasserstein" for the distance metric.

All results are saved in a JSON file at the end.
"""

import argparse
import json
import os

import numpy as np
import torch
from ot import sliced_wasserstein_distance
from tqdm import trange

# Local imports (make sure these match your actual file structure)
from bproj import init_bproj, train_bproj
from config import Config
from cpat import init_cpat, train_cpat
from scones import SCONES
from score import init_score, train_score


def main():
    parser = argparse.ArgumentParser(description="Gaussian -> Swiss Roll experiment.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing models if True."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu or cuda)."
    )

    # Score hyperparams
    parser.add_argument(
        "--score_lr", type=float, default=1e-6, help="Learning rate for score model."
    )
    parser.add_argument(
        "--score_iters",
        type=int,
        default=1000,
        help="Number of score training iterations.",
    )
    parser.add_argument(
        "--score_bs", type=int, default=500, help="Batch size for score training."
    )
    parser.add_argument(
        "--score_noise_init",
        type=float,
        default=3.0,
        help="Initial noise std for score training.",
    )
    parser.add_argument(
        "--score_noise_final",
        type=float,
        default=0.01,
        help="Final noise std for score training.",
    )
    parser.add_argument(
        "--score_n_classes",
        type=int,
        default=10,
        help="Number of different noise levels used during annealing.",
    )
    parser.add_argument(
        "--score_steps_per_class", type=int, default=300, help="Steps per noise level."
    )

    parser.add_argument(
        "--scones_sampling_lrs",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 2],
        help="List of step sizes for Langevin dynamics.",
    )

    parser.add_argument(
        "--scones_iters", type=int, default=1000, help="Number of SCONES iterations."
    )
    parser.add_argument(
        "--scones_bs",
        type=int,
        default=1000,
        help="Number of independent samples to generate during scones sampling.",
    )
    parser.add_argument(
        "--scones_samples_per_source",
        type=int,
        default=1,
        help="For each source sample, how many target samples to generate conditioned on that source",
    )

    parser.add_argument(
        "--lmbdas",
        type=float,
        nargs="+",
        default=[2.0],
        help="List of lambda regularization parameters to try.",
    )

    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="List of hidden layer dimensions for CPAT and BProj",
    )
    parser.add_argument(
        "--score_hidden_dim",
        type=int,
        default=2048,
        help="Hidden layer dimension for score model.",
    )

    # Other experimental controls
    parser.add_argument(
        "--seed", type=int, default=2039, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of independent runs per setting."
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=500,
        help="Number of samples to draw for measuring performance.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print more logs if True."
    )

    args = parser.parse_args()

    results = {}

    for subdir in [
        "pretrained",
        "pretrained/cpat",
        "pretrained/bproj",
        "pretrained/score",
    ]:
        os.makedirs(subdir, exist_ok=True)

    for hidden_dim in args.hidden_dims:
        results[hidden_dim] = {}
        for lam in args.lmbdas:
            results[hidden_dim][lam] = {}
            for scones_lr in args.scones_sampling_lrs:
                results[hidden_dim][lam][scones_lr] = {"runs": []}

                # For reproducibility across runs
                for run_idx in trange(
                    args.runs,
                    desc=f"hid={hidden_dim}, lam={lam}, scones_lr={scones_lr}",
                ):
                    # Build our configuration
                    cnf = Config(
                        name=f"Gaussian2Swiss-hdim={hidden_dim}_lam={lam}",
                        source="gaussian",
                        target="swiss-roll",
                        l=lam,
                        device=args.device,
                        score_lr=args.score_lr,
                        score_iters=args.score_iters,
                        score_bs=args.score_bs,
                        score_noise_init=args.score_noise_init,
                        score_noise_final=args.score_noise_final,
                        score_n_classes=args.score_n_classes,
                        score_steps_per_class=args.score_steps_per_class,
                        score_sampling_lr=scones_lr,
                        scones_iters=args.scones_iters,
                        scones_bs=args.scones_bs,
                        scones_samples_per_source=args.scones_samples_per_source,
                        bproj_hidden_layer_dim=hidden_dim,
                        cpat_hidden_layer_dim=hidden_dim,
                        score_hidden_dim=args.score_hidden_dim,
                        seed=args.seed,
                    )

                    torch.manual_seed(cnf.seed)
                    np.random.seed(cnf.seed)

                    cpat = init_cpat(cnf)

                    cpat_path = os.path.join("pretrained", "cpat", cnf.name)
                    bproj_path = os.path.join("pretrained", "bproj", cnf.name)

                    if (not args.overwrite) and os.path.exists(
                        os.path.join(cpat_path, "cpat.pt")
                    ):
                        if args.verbose:
                            print(f"Loading pretrained CPAT from {cpat_path}")
                        cpat.load(os.path.join(cpat_path, "cpat.pt"))
                    else:
                        train_cpat(cpat, cnf, verbose=args.verbose)

                    bproj = init_bproj(cpat, cnf)
                    if (not args.overwrite) and os.path.exists(
                        os.path.join(bproj_path, "bproj.pt")
                    ):
                        if args.verbose:
                            print(f"Loading pretrained BProj from {bproj_path}")
                        bproj.load(os.path.join(bproj_path, "bproj.pt"))
                    else:
                        train_bproj(bproj, cnf, verbose=args.verbose)

                    score = init_score(cnf)

                    score_path = os.path.join(
                        "pretrained", "score", f"Swiss-Roll_{lam}"
                    )

                    if (not args.overwrite) and os.path.exists(
                        os.path.join(score_path, "score.pt")
                    ):
                        if args.verbose:
                            print(f"Loading pretrained score from {score_path}")
                        score.load(os.path.join(score_path, "score.pt"))
                    else:
                        train_score(score, cnf, verbose=args.verbose)

                    scones = SCONES(cpat, score, bproj, cnf)

                    Xs = cnf.source_dist.rvs(size=(args.test_samples,))
                    Xs_th = torch.FloatTensor(Xs).to(cnf.device)

                    bproj_Xs_th = bproj.projector(Xs_th)
                    scones_Xs_th = scones.sample(
                        Xs_th, verbose=args.verbose, source_init=True
                    )

                    if scones_Xs_th.ndim == 3 and scones_Xs_th.shape[1] > 1:
                        scones_Xs_th = scones_Xs_th.mean(dim=1)
                    elif scones_Xs_th.ndim == 3:
                        scones_Xs_th = scones_Xs_th[:, 0, :]

                    Y = cnf.target_dist.rvs(size=(args.test_samples,))

                    bproj_wass = sliced_wasserstein_distance(
                        Y,
                        bproj_Xs_th.detach().cpu().numpy(),
                        seed=cnf.seed,
                    )
                    scones_wass = sliced_wasserstein_distance(
                        Y,
                        scones_Xs_th,
                        seed=cnf.seed,
                    )
                    if np.isnan(bproj_wass):
                        bproj_wass = np.inf
                    if np.isnan(scones_wass):
                        scones_wass = np.inf

                    print("BProj Wasserstein distance:", bproj_wass)
                    print("SCONES Wasserstein distance:", scones_wass)

                    results[hidden_dim][lam][scones_lr]["runs"].append(
                        {
                            "run_idx": run_idx,
                            "bproj_wasserstein": bproj_wass,
                            "scones_wasserstein": scones_wass,
                        }
                    )

                bproj_mean = np.mean(
                    [
                        r["bproj_wasserstein"]
                        for r in results[hidden_dim][lam][scones_lr]["runs"]
                    ]
                )
                scones_mean = np.mean(
                    [
                        r["scones_wasserstein"]
                        for r in results[hidden_dim][lam][scones_lr]["runs"]
                    ]
                )
                results[hidden_dim][lam][scones_lr]["bproj_mean_wasserstein"] = float(
                    bproj_mean
                )
                results[hidden_dim][lam][scones_lr]["scones_mean_wasserstein"] = float(
                    scones_mean
                )

    results_dict = {}
    for hd, lam_dict in results.items():
        results_dict[str(hd)] = {}
        for lam, scones_dict in lam_dict.items():
            results_dict[str(hd)][str(lam)] = {}
            for scones_lr, val in scones_dict.items():
                run_entries = []
                for run_data in val["runs"]:
                    run_entries.append(
                        {
                            "run_idx": run_data["run_idx"],
                            "bproj_wasserstein": float(run_data["bproj_wasserstein"]),
                            "scones_wasserstein": float(run_data["scones_wasserstein"]),
                        }
                    )
                new_val = {
                    "runs": run_entries,
                    "bproj_mean_wasserstein": val["bproj_mean_wasserstein"],
                    "scones_mean_wasserstein": val["scones_mean_wasserstein"],
                }
                results_dict[str(hd)][str(lam)][str(scones_lr)] = new_val

    dims_str = "-".join([str(x) for x in args.hidden_dims])
    lam_str = "-".join([str(x) for x in args.lmbdas])
    eps_str = "-".join([str(x) for x in args.scones_sampling_lrs])
    out_filename = f"Results_hdims={dims_str}_lmbdas={lam_str}_epsilons={eps_str}.json"
    with open(out_filename, "w") as f_out:
        json.dump(results_dict, f_out, indent=2)

    print(f"\nDone! Results saved to '{out_filename}'.")


if __name__ == "__main__":
    main()
