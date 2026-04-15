"""
Astronomy application: denoising gamma-ray burst locations on S2.

Data: BATSE 4B catalog (right ascension & declination of 1637 gamma-ray bursts).

Run from the repository root:
    python examples/real_data/astro.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reb import (
    get_manifold,
    scoreMatchingKFoldCV,
    denoiser,
    density_estimate,
)
from reb.plotting import S2scatter, S2plot_quiver, S2grid_fib

manifold_type = 'S2'
manifold = get_manifold(manifold_type)

# ── Load data ─────────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), 'data', 'BATSE_4B.txt')
df = pd.read_csv(data_path, header=None, sep=r'\s+', encoding='utf-8')

ra  = df[5].values
dec = df[6].values
phi   = np.deg2rad(ra)
theta = np.pi / 2 - np.deg2rad(dec)
X_sph = np.column_stack([theta, phi])
X = manifold.spherical_to_extrinsic(X_sph)

sigma2 = 3.046e-2
print(f"n = {len(X)},  sigma2 = {sigma2:.4f}")

# ── Hyperparameter selection ──────────────────────────────────────────────────
M_grid = np.arange(1, 25)
rhoperc_grid = 1.5
params, scores = scoreMatchingKFoldCV(
    manifold_type, X, M_grid, rhoperc_grid,
    n_splits=5, return_scores=True, random_state=42,
)
M, rho = params['AIC']
print(f"Selected M={M}, rho={rho:.4f}")

# ── Denoise ───────────────────────────────────────────────────────────────────
delta = denoiser(manifold_type, X, M, rho, sigma2, X)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(20, 8), subplot_kw={'projection': 'mollweide'})

axs[0].grid(True, color='gray', lw=0.5)
axs[0].set_title(r'$X_i$ (observed)', y=1.1)
S2scatter(X, ax=axs[0], color='C0', alpha=0.25, s=10)

grid_fib, _, _ = S2grid_fib(50)
_, _hat_f, _hat_grad_f = density_estimate(manifold_type, X, M, grid_fib)
_score = _hat_grad_f / np.maximum(_hat_f, rho)[:, None]
S2plot_quiver(grid_fib, _score, figax=(fig, axs[1]), scale=50, cmap='Greens', cvals=_hat_f)
axs[1].set_title(r'$\nabla \log \hat f$', y=1.1)

axs[2].set_title(r'$\hat\delta_\mathcal{T}(X_i)$ (denoised)', y=1.1)
S2scatter(delta, ax=axs[2], color='C2', alpha=0.25, s=10)

tick_labels = ['120°', '60°', '0°', '60°', '120°']
for ax in axs:
    ax.set_xticks(np.linspace(-np.pi, np.pi, 7)[1:-1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(np.linspace(-np.pi / 2, np.pi / 2, 5))
    ax.set_yticklabels(['90°', '45°', '0°', '45°', '90°'])

plt.tight_layout()
os.makedirs('fig', exist_ok=True)
plt.savefig('fig/astro.png', bbox_inches='tight')
print("Saved fig/astro.png")
plt.show()
