import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from .density_estimation import density_estimate
from .helpers import get_manifold


def _spectral_dimension(manifold_type, M):
    if manifold_type == 'S1':
        return 2 * M + 1
    elif manifold_type == 'S2':
        return (M + 1) ** 2
    elif manifold_type == 'T2':
        return (2 * M + 1) ** 2
    elif manifold_type == 'SO3':
        return np.sum((2 * np.arange(M + 1) + 1) ** 2)
    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")


def _get_best_params(score_matrix, M_grid, rho_grid):
    """Return the best (M, rho) pair; break ties by smallest rho, smallest M."""
    masked = np.where(np.isfinite(score_matrix), score_matrix, np.inf)
    min_val = np.min(masked)
    if not np.isfinite(min_val):
        return (np.nan, np.nan)
    candidates = np.argwhere(masked == min_val)
    if np.any(rho_grid == None):
        best = max(candidates, key=lambda ix: -M_grid[ix[0]])
        return (M_grid[int(best[0])], int(best[0]))
    else:
        best = max(candidates, key=lambda ix: (-rho_grid[ix[1]], -M_grid[ix[0]]))
        ixM, ixRho = int(best[0]), int(best[1])
        return (M_grid[ixM], rho_grid[ixRho])


def _get_scores(manifold_type, n, cv_scores, M_grid, rho_grid):
    k_vals = np.array([_spectral_dimension(manifold_type, M) for M in M_grid])
    if np.any(rho_grid == None):
        k_penalty = k_vals
    else:
        k_penalty = k_vals[:, np.newaxis]
    AIC_scores = cv_scores + 2 * k_penalty / n
    BIC_scores = cv_scores + np.log(n) * k_penalty / n

    return (
        {"cv": cv_scores, "AIC": AIC_scores, "BIC": BIC_scores},
        {
            "cv": _get_best_params(cv_scores, M_grid, rho_grid),
            "AIC": _get_best_params(AIC_scores, M_grid, rho_grid),
            "BIC": _get_best_params(BIC_scores, M_grid, rho_grid),
        },
    )


def scoreMatchingKFoldCV(manifold_type, X, M_grid, rho_percentile=3,
                         n_splits=5,
                         return_scores=False,
                         random_state=None,
                         display_tqdm=True,
                         eps=1e-5):
    """
    Select truncation level M and density lower bound rho by K-fold cross-validation
    using the Hyvärinen score matching objective.

    Parameters
    ----------
    manifold_type : str
        'S1', 'S2', 'T2', or 'SO3'.
    X : np.ndarray
        Data points, shape (n, d).
    M_grid : array-like of int
        Candidate spectral truncation levels.
    rho_percentile : float or array-like of float
        Candidate percentile(s) of the estimated density used to set rho.
    n_splits : int
        Number of folds for K-fold CV.
    return_scores : bool
        If True, also return the score matrices for all (M, rho) pairs.
    random_state : int or None
    display_tqdm : bool

    Returns
    -------
    dict
        Best (M, rho) for each criterion: 'cv', 'AIC', 'BIC'.
    scores : dict (only if return_scores=True)
        Score matrices for each criterion.
    """
    M_grid = np.array(M_grid)
    rho_percentile = np.atleast_1d(rho_percentile)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_iter = (tqdm(kf.split(X), total=n_splits, desc="Folds") if display_tqdm else kf.split(X))

    cv_scores = np.zeros((len(M_grid), len(rho_percentile)), dtype=float)

    for train_idx, val_idx in fold_iter:
        X_train, X_val = X[train_idx], X[val_idx]
        for ixM, M in enumerate(M_grid):
            _, base_f, grad_f, lap_f = density_estimate(manifold_type, X_train, M, X_val, grad=True, laplacian=True)
            grad_sq = np.sum(grad_f.reshape(len(grad_f), -1) ** 2, axis=1)
            positive_f = base_f[base_f > 0]
            for ixRho, p in enumerate(rho_percentile):
                rho = np.percentile(positive_f, p)
                hat_f = np.maximum(base_f, rho) + eps
                score_vals = (2 * lap_f / hat_f) - (grad_sq / hat_f**2)
                cv_scores[ixM, ixRho] += np.mean(score_vals)

    cv_scores /= n_splits

    scores, params = _get_scores(manifold_type, len(X), cv_scores, M_grid, rho_percentile)

    resolved_params = {}
    for key, (M, best_p) in params.items():
        _, base_f_full = density_estimate(manifold_type, X, M, X, grad=False, laplacian=False)
        resolved_params[key] = (M, np.percentile(base_f_full[base_f_full > 0], best_p))

    if return_scores:
        return resolved_params, scores
    return resolved_params


def DensityKFoldCV(manifold_type, X, M_grid, rho_percentile=3,
                   n_splits=5,
                   return_scores=False,
                   random_state=None,
                   display_tqdm=True,
                   eps=1e-5):
    """
    Select truncation level M and density lower bound rho by K-fold cross-validation
    using a density estimation objective.

    Parameters
    ----------
    manifold_type : str
    X : np.ndarray
    M_grid : array-like of int
    rho_percentile : float or array-like of float
    n_splits : int
    return_scores : bool
    random_state : int or None
    display_tqdm : bool

    Returns
    -------
    dict
        Best (M, rho) for each criterion: 'cv', 'AIC', 'BIC'.
    """
    M_grid = np.array(M_grid)
    rho_percentile = np.atleast_1d(rho_percentile)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_iter = (tqdm(kf.split(X), total=n_splits, desc="Folds") if display_tqdm else kf.split(X))

    cv_scores = np.zeros((len(M_grid), len(rho_percentile)), dtype=float)

    for train_idx, val_idx in fold_iter:
        X_train, X_val = X[train_idx], X[val_idx]
        for ixM, M in enumerate(M_grid):
            _, base_f = density_estimate(manifold_type, X_train, M, X_val, grad=False, laplacian=False)
            positive_f = base_f[base_f > 0]
            for ixRho, p in enumerate(rho_percentile):
                rho = np.percentile(positive_f, p)
                hat_f = np.maximum(base_f, rho) + eps
                cv_scores[ixM, ixRho] += np.mean(hat_f**2) - 2 * np.mean(hat_f)

    cv_scores /= n_splits
    scores, params = _get_scores(manifold_type, len(X), cv_scores, M_grid, rho_percentile)

    resolved_params = {}
    for key, (M, best_p) in params.items():
        _, base_f_full = density_estimate(manifold_type, X, M, X, grad=False, laplacian=False)
        resolved_params[key] = (M, np.percentile(base_f_full[base_f_full > 0], best_p))

    if return_scores:
        return resolved_params, scores
    return resolved_params


def plot_cv_scores(cv_scores, M_grid, rho_grid, title="CV Scores", ax=None):
    """
    Plot the CV score matrix.

    Parameters
    ----------
    cv_scores : np.ndarray
        Score matrix of shape (len(M_grid), len(rho_grid)).
    M_grid : array-like
    rho_grid : array-like or None
    title : str
    ax : matplotlib.axes.Axes, optional
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    rho_grid = np.atleast_1d(rho_grid) if rho_grid is not None else None
    if (rho_grid is not None) and (len(rho_grid) > 1):
        im = ax.imshow(cv_scores, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(np.arange(len(rho_grid)))
        ax.set_xticklabels([f"{r:.2e}" if r < 0.01 else f"{r:.2f}" for r in rho_grid], rotation=45)
        ax.set_xlabel(r"$\rho$ (Lower Bound)")
        plt.colorbar(im, label='Score')
        ax.set_yticks(np.arange(len(M_grid)))
        ax.set_yticklabels(M_grid)
        ax.set_ylabel(r"$M$ (Expansion Degree)")
    else:
        ax.plot(M_grid, cv_scores.flatten(), marker='o')
        ax.set_xlabel(r"$M$ (Expansion Degree)")
    ax.set_title(title)
    if ax is None:
        plt.tight_layout()
        plt.show()
