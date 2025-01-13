import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from ot import sliced_wasserstein_distance

from bproj import BaryProj, init_bproj, train_bproj
from config import Config
from cpat import Compatibility, init_cpat, train_cpat
from datasets import Gaussian, SwissRoll
from scones import SCONES
from score import Score, init_score, train_score
from sinkhorn import bw_uvp, sample_stats, sinkhorn, sq_bw_distance

"""
Given a configuration, train SCONES and BP and output
"""

hidden_dims = 512

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
    score_steps_per_class=30,
    score_sampling_lr=1e-4,
    scones_samples_per_source=1,
    seed=2039,
    bproj_hidden_layer_dim=512,
    cpat_hidden_layer_dim=512,
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

print("Bproj")
bproj_Xs_th = bproj.projector(Xs_th).detach()
bproj_Xs = bproj_Xs_th.cpu().numpy()

print("Scones")
scones_samples = scones.sample(Xs_th, verbose=True, source_init=True)

print("Sampling done")
a, b = None, None
"""
(
    np.ones((Xs.shape[0],)) / Xs.shape[0],
    np.ones((Xs.shape[0],)) / Xs.shape[0],
)
"""
print("bp shape", bproj_Xs_th.cpu().numpy().shape)
print("sc shape", scones_samples.shape)
print("Y shape", Y.shape)
scones_samples = (
    np.mean(scones_samples, axis=1)
    if scones_samples.shape[1] > 1
    else scones_samples[:, 0, :]
)
bproj_sinkhorn_dist = sliced_wasserstein_distance(Y, bproj_Xs, seed=cnf.seed)
scones_sinkhorn_dist = sliced_wasserstein_distance(Y, scones_samples, seed=cnf.seed)
print("bproj distance:", bproj_sinkhorn_dist)
print("scones distance:", scones_sinkhorn_dist)

plt.subplot(1, 2, 1)
plt.scatter(*Xs.T, color="#330C2F", label="Source")
plt.scatter(*cnf.target_dist.rvs(size=(n_samples,)).T, color="#7B287D", label="Target")
plt.legend()
# plt.ylim(-15, 16)
# plt.xlim(-12, 16)
plt.title("Source and Target")

plt.subplot(1, 2, 2)
plt.scatter(*bproj_Xs.T, label="BPROJ", color="#7067CF")
plt.scatter(*scones_samples.reshape(-1, 2).T, label="SCONES", color="#1d3557")
plt.legend()
# plt.ylim(-15, 16)
# plt.xlim(-12, 16)
plt.title("Source $\\to$ Target Transportation")

plt.gcf().set_size_inches(10, 5)
plt.savefig("Source_2_Target.png")
plt.show()

# np.save("Cutout_Bproj_Gaussian->SwissRoll.npy", bproj_Xs)
# np.save("Cutout_SCONES_Gaussian->SwissRoll.npy", scones_samples)
# np.save("Cutout.npy", scones_samples)
# np.save("Sources.npy", Xs)
# np.save("Target.npy", cnf.target_dist.rvs(size=(k,)))
