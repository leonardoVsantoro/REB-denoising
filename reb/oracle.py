import numpy as np
from .helpers import get_manifold, uniform_points


def oracle_denoiser(manifold_type, oracle_samples, sigma2, X_to_denoise, G=None, n_bins=None):
    """
    Oracle denoiser using binned approximation of the score function.

    Requires access to samples from the true prior G (unavailable in practice,
    but useful as a benchmark for the empirical denoiser).

    Parameters
    ----------
    manifold_type : str
        'S1', 'S2', 'SO3', or 'T2'.
    oracle_samples : int or np.ndarray
        Either a pre-drawn array of samples from G, or an integer (requires G).
    sigma2 : float
        Noise variance.
    X_to_denoise : np.ndarray
        Noisy observations to denoise.
    G : callable, optional
        Prior sampler G(n_samples). Required if oracle_samples is an integer.
    n_bins : int, optional
        Number of bins for the approximate score. Defaults to len(oracle_samples)//10.

    Returns
    -------
    np.ndarray
        Denoised points in extrinsic coordinates.
    """
    oracle_samples = oracle_samples if not np.isscalar(oracle_samples) else G(oracle_samples)
    if n_bins is not None:
        n_bins = min(len(oracle_samples) // 10, n_bins)
    else:
        n_bins = len(oracle_samples) // 10

    manifold = get_manifold(manifold_type)
    bin_centers = uniform_points(manifold_type, n_bins)
    dists_all = np.array([manifold.metric.dist(g, bin_centers) for g in oracle_samples])
    labels = np.argmin(dists_all, axis=1)
    bin_weights = np.bincount(labels, minlength=n_bins) / len(oracle_samples)

    oracle_score = []
    for x in X_to_denoise:
        dists = manifold.metric.dist(x, bin_centers)
        logs = manifold.metric.log(x, bin_centers)
        weights = bin_weights * np.exp(-(dists**2) / (2 * sigma2))
        oracle_score.append(
            -(1 / sigma2) * (weights[:, None] * logs).sum(axis=0) / weights.sum()
        )

    oracle_score = np.array(oracle_score)
    oracle_tangent_vecs = X_to_denoise + sigma2 * oracle_score
    return manifold.metric.exp(oracle_tangent_vecs, X_to_denoise)


def oracle_denoiser__naive(manifold_type, oracle_samples, sigma2, X_to_denoise, G=None):
    """
    Oracle denoiser using exact pairwise distances to all prior samples.

    Slower than oracle_denoiser but does not require binning.

    Parameters
    ----------
    manifold_type : str
    oracle_samples : int or np.ndarray
    sigma2 : float
    X_to_denoise : np.ndarray
    G : callable, optional

    Returns
    -------
    np.ndarray
        Denoised points in extrinsic coordinates.
    """
    manifold = get_manifold(manifold_type)
    oracle_samples = oracle_samples if not np.isscalar(oracle_samples) else G(oracle_samples)

    oracle_score = []
    for x in X_to_denoise:
        dists = manifold.metric.dist(x, oracle_samples)
        logs = manifold.metric.log(x, oracle_samples)
        weights = np.exp(-(dists ** 2) / (2 * sigma2))
        oracle_score.append(
            -(1 / sigma2) * (weights[:, None] * logs).sum(axis=0) / weights.sum()
        )
    oracle_score = np.array(oracle_score)
    oracle_tangent_vecs = X_to_denoise + sigma2 * oracle_score
    return manifold.metric.exp(oracle_tangent_vecs, X_to_denoise)
