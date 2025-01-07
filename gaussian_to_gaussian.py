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
        default=[2, 16, 64, 128, 256],
        help="Dimensions to run",
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
        default=0.00001,
        help="Learning rate for CPAT (divided by dimension)",
    )
    parser.add_argument(
        "--bproj_bs", type=int, default=500, help="Batch size for BProj optimization"
    )
    parser.add_argument(
        "--bproj_iters", type=int, default=5000, help="Number of BProj iterations"
    )
    parser.add_argument(
        "--bproj_lr", type=float, default=0.00001, help="Learning rate for BProj"
    )
    parser.add_argument(
        "--scones_iters", type=int, default=1000, help="Number of SCONES iterations"
    )
    parser.add_argument(
        "--scones_sampling_lr",
        type=float,
        default=0.001,
        help="Learning rate for SCONES sampling",
    )
    parser.add_argument(
        "--scones_samples_per_source",
        type=int,
        default=10,
        help="Scones samples per source",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose output during training"
    )
    args = parser.parse_args()
    OVERWRITE = args.overwrite
    dims = args.dims
    results = {str(d): {"runs": []} for d in dims}
    for d in dims[::-1]:
        for i in trange(3):
            cnf = GaussianConfig(
                name=f"l={2 * d}_d={d}_k=0",
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
                device=args.device,
                l=d * 2,  # regularization parameter
                seed=args.seed,
            )

            torch.manual_seed(cnf.seed)
            np.random.seed(cnf.seed)

            cpat = init_cpat(cnf)

            if (not OVERWRITE) and os.path.exists(
                os.path.join("pretrained/cpat", cnf.name)
            ):
                cpat.load(os.path.join("pretrained/cpat", cnf.name, "cpat.pt"))
            else:
                train_cpat(cpat, cnf, verbose=args.verbose)

            bproj = init_bproj(cpat, cnf)

            if (not OVERWRITE) and os.path.exists(
                os.path.join("pretrained/bproj", cnf.name)
            ):
                bproj.load(os.path.join("pretrained/bproj", cnf.name, "bproj.pt"))
            else:
                train_bproj(bproj, cnf, verbose=args.verbose)

            prior = GaussianScore(cnf.target_dist, cnf)
            scones = GaussianSCONES(cpat, prior, bproj, cnf)

            Xs = cnf.source_dist.rvs(size=(cnf.cov_samples,))
            Xs_th = torch.FloatTensor(Xs).to(cnf.device)

            mean = np.zeros((cnf.source_dim + cnf.target_dim,))

            bproj_cov = bproj.covariance(Xs_th)
            scones_cov = scones.covariance(Xs_th, verbose=args.verbose)

            bproj_bw_uvp = bw_uvp(bproj_cov, cnf.source_cov, cnf.target_cov, cnf.l)
            scones_bw_uvp = bw_uvp(scones_cov, cnf.source_cov, cnf.target_cov, cnf.l)

            results[str(d)]["runs"].append(
                {"d": d, "bproj-bw-uvp": bproj_bw_uvp, "scones-bw-uvp": scones_bw_uvp}
            )
        bproj_avg_bw_uvp = np.mean(
            [run["bproj-bw-uvp"] for run in results[str(d)]["runs"]]
        )
        scones_avg_bw_uvp = np.mean(
            [run["scones-bw-uvp"] for run in results[str(d)]["runs"]]
        )
        print(f"BPROJ average BW-UVP at d={d}: {bproj_avg_bw_uvp}")
        print(f"SCONES average BW-UVP at d={d}: {scones_avg_bw_uvp}")
        results[str(d)]["bproj-mean-bw-uvp"] = bproj_avg_bw_uvp
        results[str(d)]["scones-mean-bw-uvp"] = scones_avg_bw_uvp
    with open(f"Results_{'_'.join([str(x) for x in dims])}.json", "w+") as f_out:
        f_out.write(json.dumps(results))
