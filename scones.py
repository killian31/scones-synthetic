import os
import shutil

import numpy as np
import torch
import tqdm
from tqdm import trange

from config import Config, GaussianConfig
from nets import FCNN, FCNN2, FCCritic
from score import GaussianScore, Score


class GaussianSCONES:
    def __init__(self, cpat, prior, bproj, cnf):
        self.cpat = cpat
        self.bproj = bproj
        self.prior = prior
        self.cnf = cnf
        self.tgt_prec = torch.FloatTensor(np.linalg.inv(cnf.target_cov)).to(cnf.device)

    def score(self, source, target, s=1):
        # compute grad log p wrt targets
        cpat_grad = self.cpat.score(source, target)
        prior_grad = self.prior.score(target, s=s)
        return cpat_grad + prior_grad

    def sample(self, source, verbose=True):
        bs = self.cnf.scones_bs
        eps = self.cnf.scones_sampling_lr
        n_batches = int(np.ceil(self.cnf.cov_samples / bs))
        source_batches = [source[bs * i : bs * (i + 1)] for i in range(n_batches)]
        target_batches = [self.bproj.projector(s) for s in source_batches]
        samples = []

        if verbose:
            pbar = tqdm.tqdm(range(n_batches))
        for b in range(n_batches):
            source = source_batches[b]
            target = target_batches[b]
            for i in range(self.cnf.scones_iters):
                Z = torch.randn_like(target)
                score = self.score(source, target)
                with torch.no_grad():
                    target = target + (eps / 2) * score + np.sqrt(eps) * Z
                target.requires_grad = True
                # if verbose and i % 100 == 0:
                #    cov = self._est_covariance(source, target)
                #    print("")
                #    print(cov)
            if verbose:
                pbar.update(1)
            samples.append(target)
        if verbose:
            pbar.close()
        return torch.cat(samples, dim=0)

    def _est_covariance(self, source, target):
        source = source.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        joint = np.concatenate((source, target), axis=1).reshape((len(source), -1))
        return np.cov(joint, rowvar=False)

    def covariance(self, source, verbose=True):
        samples = self.sample(source, verbose=verbose)
        joint = np.concatenate(
            (source.detach().cpu().numpy(), samples.detach().cpu().numpy()), axis=1
        )
        return np.cov(joint, rowvar=False)


class SCONES:
    def __init__(self, cpat, score, bproj, cnf):
        self.cpat = cpat
        self.bproj = bproj
        self.score_est = score
        self.cnf = cnf

    def score(self, source, target, noise_std=1):
        # compute grad log p wrt targets
        cpat_grad = self.cpat.score(source, target)
        prior_grad = self.score_est.score(target, noise_std=noise_std)
        return cpat_grad + prior_grad

    def sample(self, source, source_init=False, verbose=True):
        n_samples = self.cnf.scones_samples_per_source * len(source)

        Xs = (
            torch.stack([source] * self.cnf.scones_samples_per_source, dim=1)
            .view(n_samples, -1)
            .to(self.cnf.device)
        )
        if source_init:
            Xt = torch.clone(Xs)
        else:
            Xt = torch.randn(size=[n_samples, 2]).to(self.cnf.device)

        for s in self.score_est.noise_scales:
            for _ in range(self.score_est.steps_per_class):
                a = (
                    self.score_est.sampling_lr
                    * (s / self.score_est.noise_scales[-1]) ** 2
                )
                noise = torch.randn(size=[n_samples, 2]).to(self.cnf.device)
                Xt.requires_grad = True
                scr = self.score(Xs, Xt, s)
                with torch.no_grad():
                    Xt = Xt + a * scr + np.sqrt(2 * a) * noise
        # denoise via tweedie's identity
        Xt.requires_grad = True
        Xt = Xt + self.score_est.noise_scales[-1] ** 2 * self.score(
            Xs, Xt, self.score_est.noise_scales[-1]
        )
        return (
            Xt.detach()
            .cpu()
            .numpy()
            .reshape(len(source), self.cnf.scones_samples_per_source, -1)
        )


if __name__ == "__main__":
    import imageio
    import matplotlib.pyplot as plt

    from bproj import init_bproj
    from cpat import init_cpat

    n_samples = 500
    gif_figs = []
    if not os.path.exists("figs"):
        os.makedirs("figs", exist_ok=True)
    try:
        for j in trange(10, 101):
            cnf = GaussianConfig(
                name="l=4_d=2_k=0",
                source_cov="data/d=2/0/source_cov.npy",
                target_cov="data/d=2/0/target_cov.npy",
                scale_huh=False,
                scones_iters=j,
                scones_bs=500,
                cov_samples=500,
                scones_sampling_lr=0.001,
                device="cpu",
                l=4,
                seed=2039,
            )

            torch.manual_seed(cnf.seed)
            np.random.seed(cnf.seed)

            cpat = init_cpat(cnf)
            cpat.load(os.path.join("pretrained/cpat", cnf.name, "cpat.pt"))
            bproj = init_bproj(cpat, cnf)
            bproj.load(os.path.join("pretrained/bproj", cnf.name, "bproj.pt"))
            prior = GaussianScore(cnf.target_dist, cnf)
            scones = GaussianSCONES(cpat, prior, bproj, cnf)
            ex_samples = cnf.target_dist.rvs(size=(n_samples,))
            source_dist = cnf.source_dist.rvs(size=(cnf.cov_samples,))

            learned_samples_bproj = (
                bproj.projector(torch.FloatTensor(source_dist).to(cnf.device))
                .detach()
                .cpu()
                .numpy()
            )
            learned_samples_scones = (
                scones.sample(
                    torch.FloatTensor(source_dist).to(cnf.device),
                    verbose=False,
                )
                .detach()
                .cpu()
                .numpy()
            )

            plt.figure(figsize=(18, 6))
            left_xlim = np.min(source_dist[:, 0])
            right_xlim = np.max(source_dist[:, 0])
            left_ylim = np.min(source_dist[:, 1])
            right_ylim = np.max(source_dist[:, 1])
            xlim = (left_xlim, right_xlim)
            ylim = (left_ylim, right_ylim)
            plt.subplot(1, 3, 1)
            plt.scatter(*ex_samples.T, s=6, color="blue")
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title("Target Samples")
            plt.subplot(1, 3, 2)
            plt.scatter(*source_dist.T, s=6, color="red")
            plt.scatter(*learned_samples_bproj.T, s=6, color="blue")
            for i in range(n_samples):
                plt.plot(
                    [source_dist[i, 0], learned_samples_bproj[i, 0]],
                    [source_dist[i, 1], learned_samples_bproj[i, 1]],
                    color="black",
                    alpha=0.3,
                    linewidth=0.3,
                )
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title("Barycentric Projection")
            plt.subplot(1, 3, 3)
            plt.scatter(*source_dist.T, s=6, color="red")
            plt.scatter(*learned_samples_scones.T, s=6, color="blue")
            for i in range(n_samples):
                plt.plot(
                    [
                        source_dist[i, 0],
                        learned_samples_scones[i, 0],
                    ],
                    [
                        source_dist[i, 1],
                        learned_samples_scones[i, 1],
                    ],
                    color="black",
                    alpha=0.3,
                    linewidth=0.3,
                )
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.title(f"SCONES Samples, Iteration {j}")
            plt.tight_layout()
            plt.savefig(f"figs/scones_{j}.png")
            gif_figs.append(f"figs/scones_{j}.png")
            plt.close()
    except KeyboardInterrupt:
        print(f"Interrupted at iteration {j}")

    with imageio.get_writer("scones.gif", mode="I") as writer:
        for filename in gif_figs:
            image = imageio.imread(filename)
            writer.append_data(image)
    print("Gif created!")
