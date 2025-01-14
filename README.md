# SCONES: Synthetic Experiments

This repository can be used to make extensive experiments on the SCONES method from
**Score-based Generative Neural Networks for Large-Scale Optimal Transport**.
([ArXiv](https://arxiv.org/abs/2110.03237)) by _Max Daniels, Tyler Maunu,
Paul Hand_.

## Setup

Create a virtual environment (Python >= 3.11) for example with pyenv, install poetry
and install the dependencies, with

```bash
pyenv virtualenv 3.11.9 scones
pyenv activate scones
pip install poetry
poetry install
```

## Running the code

### Gaussian to Gaussian

Experiments on Gaussian to Gaussian distribution can be run with the following command:

```bash
python g2g.py [-h] [--overwrite] [--device DEVICE] [--dims DIMS [DIMS ...]] [--lmbdas LMBDAS] [--hidden_dims HIDDEN_DIMS [HIDDEN_DIMS ...]] [--seed SEED] [--cpat_bs CPAT_BS]
              [--cpat_iters CPAT_ITERS] [--cpat_lr CPAT_LR] [--bproj_bs BPROJ_BS] [--bproj_iters BPROJ_ITERS] [--bproj_lr BPROJ_LR] [--scones_iters SCONES_ITERS]
              [--scones_sampling_lr SCONES_SAMPLING_LR [SCONES_SAMPLING_LR ...]] [--scones_bs SCONES_BS] [--cov_samples COV_SAMPLES] [--verbose]

options:
  -h, --help            show this help message and exit
  --overwrite           Overwrite existing models
  --device DEVICE       Device to use
  --dims DIMS [DIMS ...]
                        Dimensions to run
  --lmbdas LMBDAS       List of regularization parameters to test
  --hidden_dims HIDDEN_DIMS [HIDDEN_DIMS ...]
                        List of hidden layer dimensions for CPAT and BProj
  --seed SEED           Seed for reproducibility
  --cpat_bs CPAT_BS     Batch size for CPAT optimization
  --cpat_iters CPAT_ITERS
                        Number of CPAT iterations
  --cpat_lr CPAT_LR     Learning rate for CPAT (will be multiplied by dimension)
  --bproj_bs BPROJ_BS   Batch size for BProj optimization
  --bproj_iters BPROJ_ITERS
                        Number of BProj iterations
  --bproj_lr BPROJ_LR   Learning rate for BProj
  --scones_iters SCONES_ITERS
                        Number of SCONES iterations
  --scones_sampling_lr SCONES_SAMPLING_LR [SCONES_SAMPLING_LR ...]
                        Learning rate for SCONES sampling
  --scones_bs SCONES_BS
                        Batch size for SCONES sampling
  --cov_samples COV_SAMPLES
                        Number of samples for covariance
  --verbose             Print verbose output during training
```

### Gaussian to Swiss Roll

Experiments on Gaussian to Swiss Roll distribution can be run with the following command:

```bash
python g2sr.py [-h] [--overwrite] [--device DEVICE] [--score_lr SCORE_LR] [--score_iters SCORE_ITERS] [--score_bs SCORE_BS] [--score_noise_init SCORE_NOISE_INIT]
               [--score_noise_final SCORE_NOISE_FINAL] [--score_n_classes SCORE_N_CLASSES] [--score_steps_per_class SCORE_STEPS_PER_CLASS]
               [--scones_sampling_lrs SCONES_SAMPLING_LRS [SCONES_SAMPLING_LRS ...]] [--scones_iters SCONES_ITERS] [--scones_bs SCONES_BS]
               [--scones_samples_per_source SCONES_SAMPLES_PER_SOURCE] [--lmbdas LMBDAS [LMBDAS ...]] [--hidden_dims HIDDEN_DIMS [HIDDEN_DIMS ...]] [--score_hidden_dim SCORE_HIDDEN_DIM]
               [--seed SEED] [--runs RUNS] [--test_samples TEST_SAMPLES] [--verbose]


options:
  -h, --help            show this help message and exit
  --overwrite           Overwrite existing models if True.
  --device DEVICE       Device to use (cpu or cuda).
  --score_lr SCORE_LR   Learning rate for score model.
  --score_iters SCORE_ITERS
                        Number of score training iterations.
  --score_bs SCORE_BS   Batch size for score training.
  --score_noise_init SCORE_NOISE_INIT
                        Initial noise std for score training.
  --score_noise_final SCORE_NOISE_FINAL
                        Final noise std for score training.
  --score_n_classes SCORE_N_CLASSES
                        Number of different noise levels used during annealing.
  --score_steps_per_class SCORE_STEPS_PER_CLASS
                        Steps per noise level.
  --scones_sampling_lrs SCONES_SAMPLING_LRS [SCONES_SAMPLING_LRS ...]
                        List of step sizes for Langevin dynamics.
  --scones_iters SCONES_ITERS
                        Number of SCONES iterations.
  --scones_bs SCONES_BS
                        Number of independent samples to generate during scones sampling.
  --scones_samples_per_source SCONES_SAMPLES_PER_SOURCE
                        For each source sample, how many target samples to generate conditioned on that source
  --lmbdas LMBDAS [LMBDAS ...]
                        List of lambda regularization parameters to try.
  --hidden_dims HIDDEN_DIMS [HIDDEN_DIMS ...]
                        List of hidden layer dimensions for CPAT and BProj
  --score_hidden_dim SCORE_HIDDEN_DIM
                        Hidden layer dimension for score model.
  --seed SEED           Random seed for reproducibility.
  --runs RUNS           Number of independent runs per setting.
  --test_samples TEST_SAMPLES
                        Number of samples to draw for measuring performance.
  --verbose             Print more logs if True.
```
