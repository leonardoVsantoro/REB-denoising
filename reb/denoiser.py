import numpy as np
from .density_estimation import density_estimate
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from .helpers import get_manifold

def denoiser(manifold_type, X, M, rho, sigma2, X_to_denoise, densityIn=None):
    """
    Perform a denoising step on a Riemannian manifold.

    Parameters
    ----------
    manifold_type : str
        Manifold type: 'S1' (circle), 'S2' (sphere), 'SO3' (rotation group), 'T2' (torus).
    X : np.ndarray
        Training data on the manifold in extrinsic coordinates, shape (n, d).
    M : int
        Spectral truncation level for density estimation.
    rho : float
        Density lower bound (regularization to avoid division by zero).
    sigma2 : float
        Noise variance.
    X_to_denoise : np.ndarray
        Noisy observations to denoise, shape (m, d).
    densityIn : tuple, optional
        Pre-computed (hat_f, hat_grad_f) to skip re-estimation.

    Returns
    -------
    delta : np.ndarray
        Denoised points on the manifold in extrinsic coordinates, shape (m, d).
    """
    manifold = get_manifold(manifold_type)
    if densityIn is None:
        _, hat_f, hat_grad_f = density_estimate(manifold_type, X, M, X_to_denoise)
    else:
        hat_f, hat_grad_f = densityIn
    hat_score = hat_grad_f / np.maximum(hat_f.reshape(hat_f.shape + (1,) * (hat_grad_f.ndim - 1)), rho)
    tangent_vecs = X_to_denoise + sigma2 * hat_score
    delta = manifold.metric.exp(tangent_vecs, X_to_denoise)
    return delta
