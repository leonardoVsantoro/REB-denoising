import numpy as np
from .helpers import get_manifold, uniform_points
from .priors import *
from tqdm import tqdm

def oracle_denoiser(manifold_type, oracle_samples, sigma2, X_to_denoise, G=None, n_bins = None):
    oracle_samples = oracle_samples if not np.isscalar(oracle_samples) else G(oracle_samples)
    if n_bins is not None:
        n_bins = min(len(oracle_samples)//10 , n_bins )
    else: n_bins  = len(oracle_samples)//10

    manifold = get_manifold(manifold_type)
    bin_centers = uniform_points(manifold_type, n_bins)          # (n_bins, D)
    dists_all = np.array([
        manifold.metric.dist(g, bin_centers) for g in oracle_samples
    ])                                                            # (N, n_bins)
    labels = np.argmin(dists_all, axis=1)                        # (N,)
    bin_weights = np.bincount(labels, minlength=n_bins) / len(oracle_samples)  # (n_bins,)

    # ------ Score estimation using bin centers (unchanged)
    oracle_score = []
    for x in X_to_denoise:
        dists   = manifold.metric.dist(x, bin_centers)
        logs    = manifold.metric.log(x, bin_centers)
        weights = bin_weights * np.exp(-(dists**2) / (2 * sigma2))
        oracle_score.append(
            -(1 / sigma2) * (weights[:, None] * logs).sum(axis=0) / weights.sum()
        )

    oracle_score = np.array(oracle_score)
    oracle_tangent_vecs = X_to_denoise + sigma2 * oracle_score
    return manifold.metric.exp(oracle_tangent_vecs, X_to_denoise)

def oracle_denoiser__naive(manifold_type, oracle_samples, sigma2, X_to_denoise, G = None ):
        '''
        Oracle denoiser using the score function estimated from samples of the generative model.
        Parameters:
        - manifold_type: 'S1', 'S2', or 'SO3'
        - num_oracle_samples: Number of samples to use for estimating the score function
        - G: Generative model with a .sample(num_samples) method that returns samples on the manifold
        - X_to_denoise: shape (M, D) - Noisy points to denoise
        - sigma2: Noise variance
        Returns:
        - oracle_delta: shape (M, D) - Denoised points on the manifold in extrinsic coordinates
        '''
        manifold = get_manifold(manifold_type)    
        oracle_samples = oracle_samples if not np.isscalar(oracle_samples) else G(oracle_samples)


        # ------ Oracle score estimation
        oracle_score = []
        for x in X_to_denoise:
            dists = manifold.metric.dist(x, oracle_samples)      # shape (N,)
            logs  = manifold.metric.log(x, oracle_samples)       # shape (N, dim)
            weights = np.exp(-(dists ** 2) / (2 * sigma2))
            oracle_score.append(
                - (1 / sigma2) * (weights[:, None] * logs).sum(axis=0) / weights.sum()
            )
        oracle_score = np.array(oracle_score)
        # ------ Oracle denoising
        oracle_tangent_vecs = X_to_denoise +  sigma2 * oracle_score
        oracle_delta =  manifold.metric.exp( oracle_tangent_vecs, X_to_denoise)
        return oracle_delta
