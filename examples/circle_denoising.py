"""
Circle (S1) denoising example using Nonparametric Riemannian Empirical Bayes.

Run from the repository root:
    python examples/circle_denoising.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from reb import (
    get_manifold,
    multimodal_sampler,
    uniform_sampler,
    scoreMatchingKFoldCV,
    denoiser,
    oracle_denoiser,
    sq_loss,
    density_estimate,
    uniform_points,
)
from reb.plotting import S1scatter, S1_histogram

# ── Configuration ────────────────────────────────────────────────────────────
manifold_type = 'S1'
sigma2 = 0.15
n_samples = 5000
manifold = get_manifold(manifold_type)

priors = [
    ('uniform',  lambda n: uniform_sampler(manifold_type, n),
                  {}),
    ('2-modal',  lambda n: multimodal_sampler(manifold_type, n, tau2=0.10, num_modes=2),
                  {'tau2': 0.10, 'num_modes': 2}),
    ('3-modal',  lambda n: multimodal_sampler(manifold_type, n, tau2=0.05, num_modes=3),
                  {'tau2': 0.05, 'num_modes': 3}),
]

os.makedirs('fig/examples', exist_ok=True)

for name, G, _ in priors:
    print(f"\n── Prior: {name} ──")

    # Sample latent variables and add Riemannian Gaussian noise
    Theta = G(n_samples)
    X = manifold.random_riemannian_normal(Theta, 1. / sigma2, n_samples)

    # Hyperparameter selection via K-fold cross-validation
    M_grid = np.arange(2, 12)
    rho_perc = np.arange(2, 20, 1)
    params = scoreMatchingKFoldCV(manifold_type, X, M_grid, rho_perc, n_splits=5, random_state=42)
    M, rho = params['AIC']
    print(f"  Selected M={M}, rho={rho:.4f}")

    # Denoise
    delta = denoiser(manifold_type, X, M, rho, sigma2, X)
    oracle_delta = oracle_denoiser(
        manifold_type, 10_000, sigma2, X, n_bins=1000,
        G=G,
    )

    # Evaluate
    loss_N = sq_loss(manifold, X, Theta)
    loss_T = sq_loss(manifold, delta, Theta)
    loss_O = sq_loss(manifold, oracle_delta, Theta)
    print(f"  MSE  noisy:    {loss_N:.4f}")
    print(f"  MSE  denoised: {loss_T:.4f}")
    print(f"  MSE  oracle:   {loss_O:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    grid_size = 50
    grid_I = np.linspace(0, 2 * np.pi, grid_size)
    bottom, top, upperlim = 0.75, 0.9, 1.1

    fig, axs = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'projection': 'polar'})
    datasets = [Theta, X, delta, oracle_delta]
    titles = [r'$\Theta$', r'$X_i$', r'$\hat\delta_\mathcal{T}(X_i)$', r'$\delta_\mathcal{T}(X_i)$']
    losses = [None, loss_N, loss_T, loss_O]
    cmaps = ['Reds', 'Blues', 'Greens', 'Greens']

    for ax, data, title, loss, cmap in zip(axs, datasets, titles, losses, cmaps):
        bin_centers = uniform_points(manifold_type, grid_size)
        dists_all = np.array([manifold.metric.dist(g, bin_centers) for g in data])
        labels = np.argmin(dists_all, axis=1)
        hat_f = np.bincount(labels, minlength=grid_size) / len(data)
        hat_f /= hat_f.max()

        ax.bar(grid_I[:-1], top - bottom, width=np.diff(grid_I), bottom=bottom,
               color=plt.colormaps[cmap](hat_f[:-1]), alpha=0.85, edgecolor='none',
               align='edge', zorder=2)
        ax.set_ylim(0, upperlim)
        ax.set_yticks([])
        ax.bar(0, bottom, width=2 * np.pi, bottom=0, color="white", edgecolor="none", align="edge", zorder=3)
        ax.plot(grid_I, bottom * np.ones_like(grid_I), color='black', linewidth=1.2, zorder=4)
        ax.set_title(title, fontsize=14)
        if loss is not None:
            ax.set_xlabel(rf'MSE: {loss:.4f}', fontsize=12)

    fig.suptitle(f"S1 denoising — prior: {name}", fontsize=14)
    plt.tight_layout()
    outpath = f'fig/examples/S1_{name}.png'
    plt.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()
