"""
Structural biology application: denoising torsion angles (Ramachandran plot) on T2.

Uses phi/psi backbone dihedral angles extracted from PDB protein structures.
Requires: biopython  (pip install biopython)

Run from the repository root:
    python examples/real_data/chemi.py
"""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

import Bio.PDB
warnings.simplefilter('ignore', Bio.BiopythonWarning)

from reb import (
    get_manifold,
    denoiser,
    density_estimate,
)
from reb.plotting import T2_scatter, T2plot_quiver, T2grid

manifold_type = 'T2'
manifold = get_manifold(manifold_type)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
pdb_files = ['6x8j.pdb1', '1ADO.pdb1', '1ENO.pdb1']

fig, aaxs = plt.subplots(
    3, 4, figsize=(12.5, 10),
    gridspec_kw={"width_ratios": [0.5, 1, 1, 1]},
)

for idx, filename in enumerate(pdb_files):
    filepath = os.path.join(data_dir, filename)
    print(f"\nProcessing {filename}...")

    # ── Parse PDB: extract phi/psi angles ────────────────────────────────────
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure('protein', filepath)
    polypeptide = Bio.PDB.PPBuilder().build_peptides(structure[0])

    phi_list, psi_list = [], []
    for strand in polypeptide:
        for pt in strand.get_phi_psi_list():
            try:
                phi_list.append(float(pt[0]))
                psi_list.append(float(pt[1]))
            except TypeError:
                pass

    phi_arr, psi_arr = np.asarray(phi_list), np.asarray(psi_list)
    X = np.asarray([[np.cos(phi_arr), np.sin(phi_arr)],
                    [np.cos(psi_arr), np.sin(psi_arr)]]).T  # shape (n, 2, 2)

    # ── Estimate sigma2 from B-factors ────────────────────────────────────────
    total_b, count = 0.0, 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    total_b += atom.get_bfactor()
                    count += 1
    mean_b = total_b / count
    sigma2 = np.arctan(np.sqrt(mean_b / (8 * np.pi**2)) / 1.5) ** 2
    print(f"  n = {len(X)},  sigma2 = {sigma2:.4f}")

    # ── Fixed parameters (cross-validation optional) ──────────────────────────
    M, rho = 3, 1e-2

    # ── Density and denoising ─────────────────────────────────────────────────
    res = 50
    grid, grid_phi, grid_psi = T2grid(res)
    _, hat_f, hat_grad_f = density_estimate(manifold_type, X, M, grid, grad=True)

    delta = denoiser(manifold_type, X, M, rho, sigma2, X)

    # ── Plot ──────────────────────────────────────────────────────────────────
    protein_name = os.path.splitext(os.path.basename(filename))[0].split('.')[0]
    aaxs[idx, 0].cla()
    aaxs[idx, 0].axis('off')
    aaxs[idx, 0].text(0.5, 0.5, protein_name,
                      transform=aaxs[idx, 0].transAxes,
                      ha='center', va='center', fontsize=18)

    axs = aaxs[idx, 1:]

    axs[0].grid(True, color='gray', lw=0.5)
    T2_scatter(X, ax=axs[0])
    axs[0].set_title(r'$X_i$')

    _grid_small, _, _ = T2grid(30)
    _, _hat_f, _hat_grad_f = density_estimate(manifold_type, X, M, _grid_small, grad=True)
    _score = _hat_grad_f / np.maximum(_hat_f[:, None, None], rho)
    _lo, _hi = np.percentile(_hat_f, [5, 95])
    _hat_f_clip = np.clip(_hat_f, _lo, _hi)
    T2plot_quiver(_grid_small, _score, figax=(fig, axs[1]),
                  scale=2 * M, cmap='Greens', cvals=_hat_f_clip)
    axs[1].set_title(r'$\nabla \log \hat f$')

    axs[2].grid(True, color='gray', lw=0.5)
    T2_scatter(delta, ax=axs[2], color='C2')
    axs[2].set_title(r'$\hat\delta_\mathcal{T}(X_i)$')

    for ax in axs.flatten():
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_aspect('equal', adjustable='box')

os.makedirs('fig', exist_ok=True)
plt.savefig('fig/chemi.png', bbox_inches='tight')
print("\nSaved fig/chemi.png")
