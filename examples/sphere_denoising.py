"""
Sphere (S2) denoising example using Nonparametric Riemannian Empirical Bayes.

Run from the repository root:
    python examples/sphere_denoising.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from reb import (
    get_manifold,
    multimodal_sampler,
    equator_sampler,
    scoreMatchingKFoldCV,
    denoiser,
    oracle_denoiser,
    sq_loss,
    density_estimate,
)
from reb.plotting import S2scatter, S2plot_quiver, S2grid, S2grid_fib

# ── Configuration ────────────────────────────────────────────────────────────
manifold_type = 'S2'
sigma2 = 0.15
n_samples = 5000
manifold = get_manifold(manifold_type)

priors = [
    ('2-modal',  lambda n: multimodal_sampler(manifold_type, n, tau2=0.075, num_modes=2)),
    ('5-modal',  lambda n: multimodal_sampler(manifold_type, n, tau2=0.05,  num_modes=5)),
    ('equator',  lambda n: equator_sampler(manifold_type, n, tau2=0.0001)),
]

os.makedirs('fig/examples', exist_ok=True)

for name, G in priors:
    print(f"\n── Prior: {name} ──")

    # Sample latent variables and add Riemannian Gaussian noise
    Theta = G(n_samples)
    X = manifold.random_riemannian_normal(Theta, 1. / sigma2, n_samples)

    # Hyperparameter selection via K-fold cross-validation
    M_grid = np.arange(2, 9)
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
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    # Top row: density, gradient, score
    axs_top = [
        fig.add_subplot(gs[0, 0:4], projection='mollweide'),
        fig.add_subplot(gs[0, 4:8], projection='mollweide'),
        fig.add_subplot(gs[0, 8:12], projection='mollweide'),
    ]

    grid_res = 50
    grid, grid_theta, grid_phi = S2grid(grid_res)
    _, hat_f, hat_grad_f = density_estimate(manifold_type, X, M, grid)
    hat_f_grid = hat_f.reshape(grid_res, grid_res)
    axs_top[0].pcolormesh(grid_phi - np.pi, np.pi / 2 - grid_theta, hat_f_grid,
                          alpha=0.8, shading='auto', cmap='Blues')
    axs_top[0].grid(True, color='gray', lw=0.5)
    axs_top[0].set_title(r'$\hat f$', fontsize=16)

    grid_fib, _, _ = S2grid_fib(20)
    _, hat_f_fib, hat_grad_f_fib = density_estimate(manifold_type, X, M, grid_fib)
    score = hat_grad_f_fib / np.maximum(hat_f_fib, rho)[:, None]

    S2plot_quiver(grid_fib, hat_grad_f_fib, figax=(fig, axs_top[1]), scale=1, cmap='Blues')
    axs_top[1].set_title(r'$\nabla \hat f$', fontsize=16)

    S2plot_quiver(grid_fib, score, figax=(fig, axs_top[2]), scale=10, cmap='Greens', cvals=hat_f_fib)
    axs_top[2].set_title(r'$\nabla \log \hat f$', fontsize=16)

    # Bottom row: prior, noisy, denoised, oracle
    axs_bot = [fig.add_subplot(gs[1, i*3:(i+1)*3], projection='mollweide') for i in range(4)]
    datasets = [Theta, X, delta, oracle_delta]
    titles = [r'$\Theta$', r'$X_i$', r'$\hat\delta_\mathcal{T}(X_i)$', r'$\delta_\mathcal{T}(X_i)$']
    losses_list = [None, loss_N, loss_T, loss_O]
    colors = ['C3', 'C0', 'C2', 'C2']

    for ax, data, title, loss, color in zip(axs_bot, datasets, titles, losses_list, colors):
        S2scatter(data, ax, color=color, alpha=0.25)
        ax.set_title(title, fontsize=16)
        if loss is not None:
            ax.set_xlabel(rf'MSE: {loss:.3f}', fontsize=13)

    fig.suptitle(f"S2 denoising — prior: {name}", fontsize=14)
    outpath = f'fig/examples/S2_{name}.png'
    plt.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()
