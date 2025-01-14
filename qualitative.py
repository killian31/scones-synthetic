import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from ot import sliced_wasserstein_distance
from tqdm import tqdm

from bproj import init_bproj, train_bproj
from config import Config
from cpat import init_cpat, train_cpat
from scones import SCONES
from score import init_score, train_score

hidden_dims = 2048
os.makedirs("qualitative", exist_ok=True)
bproj_dists = []
scones_dists = []
times = []
pbar = tqdm(total=4)

steps_to_run = [100, 300, 600, 1000]

for steps in steps_to_run:
    cnf = Config(
        name=f"Gaussian2Swiss-hdim={hidden_dims}_lam=2.0",
        source="gaussian",
        target="swiss-roll",
        l=2,
        score_lr=0.000001,
        score_iters=1000,
        score_bs=500,
        score_noise_init=3,
        score_noise_final=0.01,
        scones_iters=1000,
        scones_bs=1000,
        device="cpu",
        score_n_classes=10,
        score_steps_per_class=steps,
        score_sampling_lr=1e-6,
        scones_samples_per_source=1,
        seed=2039,
        bproj_hidden_layer_dim=hidden_dims,
        cpat_hidden_layer_dim=hidden_dims,
        score_hidden_dim=2048,
    )
    torch.manual_seed(cnf.seed)
    np.random.seed(cnf.seed)

    cpat = init_cpat(cnf)

    # If TRUE, ignore any existing pretrained models and overwrite them.
    OVERWRITE = False

    # Create directories for saving pretrained models if they do not already exist
    touch_path = lambda p: os.makedirs(p) if not os.path.exists(p) else None
    for path in ["", "cpat", "bproj", "ncsn"]:
        touch_path("pretrained/" + path)

    # Search for and load any existing pretrained models
    if (not OVERWRITE) and os.path.exists(os.path.join("pretrained/cpat", cnf.name)):
        cpat.load(os.path.join("pretrained/cpat", cnf.name, "cpat.pt"))
    else:
        train_cpat(cpat, cnf, verbose=True)

    bproj = init_bproj(cpat, cnf)

    if (not OVERWRITE) and os.path.exists(os.path.join("pretrained/bproj", cnf.name)):
        bproj.load(os.path.join("pretrained/bproj", cnf.name, "bproj.pt"))
    else:
        train_bproj(bproj, cnf, verbose=True)

    score = init_score(cnf)
    score_path = os.path.join("pretrained", "score", "Swiss-Roll_2.0")

    if (not OVERWRITE) and os.path.exists(score_path):
        score.load(os.path.join(score_path, "score.pt"))
    else:
        train_score(score, cnf, verbose=True)

    scones = SCONES(cpat, score, bproj, cnf)

    # Sample and test the model
    n_samples = 500
    Xs = cnf.source_dist.rvs(size=(n_samples,))
    Xs_th = torch.FloatTensor(Xs).to(cnf.device)
    Y = cnf.target_dist.rvs(size=(n_samples,))

    bproj_Xs_th = bproj.projector(Xs_th).detach()
    bproj_Xs = bproj_Xs_th.cpu().numpy()

    t0 = time.time()
    scones_samples = scones.sample(Xs_th, verbose=True, source_init=True)
    times.append(time.time() - t0)

    a, b = None, None
    scones_samples = (
        np.mean(scones_samples, axis=1)
        if scones_samples.shape[1] > 1
        else scones_samples[:, 0, :]
    )
    bproj_sinkhorn_dist = sliced_wasserstein_distance(Y, bproj_Xs, seed=cnf.seed)
    scones_sinkhorn_dist = sliced_wasserstein_distance(Y, scones_samples, seed=cnf.seed)
    pbar.set_description(
        f"Steps: {steps} | BProj SWD: {bproj_sinkhorn_dist:.2f} | SCONES SWD: {scones_sinkhorn_dist:.2f} | Time: {times[-1]:.2f}"
    )
    bproj_dists.append(bproj_sinkhorn_dist)
    scones_dists.append(scones_sinkhorn_dist)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1, aspect="equal")
    plt.scatter(*Xs.T, color="#330C2F", label="Source", s=6)
    plt.scatter(
        *cnf.target_dist.rvs(size=(n_samples,)).T, color="#7B287D", label="Target", s=6
    )
    plt.legend()
    plt.ylim(-15, 16)
    plt.xlim(-12, 16)
    plt.title("Source and Target")

    plt.subplot(1, 3, 2, aspect="equal")
    plt.scatter(*Xs.T, color="#330C2F", s=6)
    plt.scatter(*bproj_Xs.T, color="#7067CF", s=6)
    for i in range(n_samples):
        plt.plot(
            [Xs[i, 0], bproj_Xs[i, 0]],
            [Xs[i, 1], bproj_Xs[i, 1]],
            color="black",
            alpha=0.3,
            linewidth=0.3,
        )
    plt.ylim(-15, 16)
    plt.xlim(-12, 16)
    plt.title(f"BProj - SWD: {bproj_sinkhorn_dist:.2f}")

    plt.subplot(1, 3, 3, aspect="equal")
    plt.scatter(*Xs.T, color="#330C2F", s=6)
    plt.scatter(*scones_samples.reshape(-1, 2).T, color="#1d3557", s=6)
    for i in range(n_samples):
        plt.plot(
            [Xs[i, 0], scones_samples[i, 0]],
            [Xs[i, 1], scones_samples[i, 1]],
            color="black",
            alpha=0.3,
            linewidth=0.3,
        )
    plt.ylim(-15, 16)
    plt.xlim(-12, 16)
    plt.title(f"SCONES - SWD: {scones_sinkhorn_dist:.2f}")

    plt.tight_layout()
    plt.savefig(f"qualitative/Gaussian2SwissRoll_{steps}_{hidden_dims}.png")
    plt.close()
    pbar.update(1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(steps_to_run, times, label="SCONES")
plt.xlabel("Steps")
plt.ylabel("Time (s)")
plt.title("Time vs Steps")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(steps_to_run, bproj_dists, label="BProj")
plt.plot(steps_to_run, scones_dists, label="SCONES")
plt.xlabel("Steps")
plt.ylabel("SWD")
plt.title("SWD vs Steps")
plt.legend()
plt.savefig("qualitative/Time_vs_Steps.png")
plt.close()
