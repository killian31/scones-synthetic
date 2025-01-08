import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.stats
import torch
from tqdm import trange

from bproj import BaryProj, init_bproj, train_bproj
from config import GaussianConfig
from cpat import Compatibility, init_cpat, train_cpat
from datasets import Gaussian
from scones import GaussianSCONES
from score import GaussianScore
from sinkhorn import bw_uvp, sample_stats, sinkhorn, sq_bw_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing models"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[2],
        help="Dimensions to run",
    )
    parser.add_argument(
        "--lmbdas",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 9.0, 10.0, 15.0, 20.0],
        help="List of regularization parameters to test",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="List of hidden layer dimensions for CPAT and BProj",
    )
    parser.add_argument(
        "--seed", type=int, default=2039, help="Seed for reproducibility"
    )
    parser.add_argument(
        "--cpat_bs", type=int, default=500, help="Batch size for CPAT optimization"
    )
    parser.add_argument(
        "--cpat_iters", type=int, default=5000, help="Number of CPAT iterations"
    )
    parser.add_argument(
        "--cpat_lr",
        type=float,
        default=1e-5,
        help="Learning rate for CPAT (will be multiplied by dimension)",
    )
    parser.add_argument(
        "--bproj_bs", type=int, default=500, help="Batch size for BProj optimization"
    )
    parser.add_argument(
        "--bproj_iters", type=int, default=5000, help="Number of BProj iterations"
    )
    parser.add_argument(
        "--bproj_lr", type=float, default=1e-5, help="Learning rate for BProj"
    )
    parser.add_argument(
        "--scones_iters", type=int, default=1000, help="Number of SCONES iterations"
    )
    parser.add_argument(
        "--scones_sampling_lr",
        type=float,
        default=1e-3,
        help="Learning rate for SCONES sampling",
    )
    parser.add_argument(
        "--scones_bs", type=int, default=1000, help="Batch size for SCONES sampling"
    )
    parser.add_argument(
        "--cov_samples",
        type=int,
        default=10000,
        help="Number of samples for covariance",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose output during training"
    )
    args = parser.parse_args()

    results = {str(d): {} for d in args.dims}

    OVERWRITE = args.overwrite

    for d in args.dims:
        for lmbda_val in args.lmbdas:
            for hidden_dim in args.hidden_dims:

                key = f"lambda={lmbda_val}_hdim={hidden_dim}"
                if key not in results[str(d)]:
                    results[str(d)][key] = {"runs": []}

                for i in trange(3, desc=f"d={d}, λ={lmbda_val}, hdim={hidden_dim}"):
                    cnf = GaussianConfig(
                        name=(
                            f"l={lmbda_val}_d={d}_cpatdim={hidden_dim}_"
                            f"bprojdim={hidden_dim}_k=0"
                        ),
                        source_cov=f"data/d={d}/{i}/source_cov.npy",
                        target_cov=f"data/d={d}/{i}/target_cov.npy",
                        scale_huh=False,
                        cpat_bs=args.cpat_bs,
                        cpat_iters=args.cpat_iters,
                        cpat_lr=args.cpat_lr * d,
                        bproj_bs=args.bproj_bs,
                        bproj_iters=args.bproj_iters,
                        bproj_lr=args.bproj_lr,
                        scones_iters=args.scones_iters,
                        scones_sampling_lr=args.scones_sampling_lr,
                        scones_bs=args.scones_bs,
                        cov_samples=args.cov_samples,
                        device=args.device,
                        l=lmbda_val,
                        seed=args.seed,
                        cpat_hidden_layer_dim=hidden_dim,
                        bproj_hidden_layer_dim=hidden_dim,
                    )

                    torch.manual_seed(cnf.seed)
                    np.random.seed(cnf.seed)

                    cpat = init_cpat(cnf)

                    cpat_model_dir = os.path.join("pretrained", "cpat", cnf.name)
                    if (not OVERWRITE) and os.path.exists(cpat_model_dir):
                        cpat.load(os.path.join(cpat_model_dir, "cpat.pt"))
                    else:
                        train_cpat(cpat, cnf, verbose=args.verbose)

                    bproj = init_bproj(cpat, cnf)

                    bproj_model_dir = os.path.join("pretrained", "bproj", cnf.name)
                    if (not OVERWRITE) and os.path.exists(bproj_model_dir):
                        bproj.load(os.path.join(bproj_model_dir, "bproj.pt"))
                    else:
                        train_bproj(bproj, cnf, verbose=args.verbose)

                    prior = GaussianScore(cnf.target_dist, cnf)
                    scones = GaussianSCONES(cpat, prior, bproj, cnf)

                    Xs = cnf.source_dist.rvs(size=(cnf.cov_samples,))
                    Xs_th = torch.FloatTensor(Xs).to(cnf.device)

                    bproj_cov = bproj.covariance(Xs_th)
                    scones_cov = scones.covariance(Xs_th, verbose=args.verbose)

                    bproj_bw_uvp = bw_uvp(
                        bproj_cov, cnf.source_cov, cnf.target_cov, cnf.l
                    )
                    scones_bw_uvp = bw_uvp(
                        scones_cov, cnf.source_cov, cnf.target_cov, cnf.l
                    )

                    results[str(d)][key]["runs"].append(
                        {
                            "run_idx": i,
                            "d": d,
                            "lambda": lmbda_val,
                            "hdim": hidden_dim,
                            "bproj-bw-uvp": float(bproj_bw_uvp),
                            "scones-bw-uvp": float(scones_bw_uvp),
                        }
                    )

                bproj_avg_bw_uvp = np.mean(
                    [run["bproj-bw-uvp"] for run in results[str(d)][key]["runs"]]
                )
                scones_avg_bw_uvp = np.mean(
                    [run["scones-bw-uvp"] for run in results[str(d)][key]["runs"]]
                )
                results[str(d)][key]["bproj-mean-bw-uvp"] = float(bproj_avg_bw_uvp)
                results[str(d)][key]["scones-mean-bw-uvp"] = float(scones_avg_bw_uvp)

    dims_str = "-".join([str(x) for x in args.dims])
    lmbdas_str = "-".join([str(x) for x in args.lmbdas])
    hdims_str = "-".join([str(x) for x in args.hidden_dims])
    out_filename = f"Results_d={dims_str}_lmbdas={lmbdas_str}_hdims={hdims_str}.json"

    with open(out_filename, "w+") as f_out:
        json.dump(results, f_out, indent=2)

    print(f"\nDone! Results saved to '{out_filename}'.")
