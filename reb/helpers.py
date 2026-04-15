from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.product_manifold import ProductManifold
import numpy as np


def get_manifold(manifold_type):
    """
    Return a geomstats manifold object for the given type.

    Parameters
    ----------
    manifold_type : str
        'S1' (circle), 'S2' (sphere), 'SO3' (rotation group), or 'T2' (torus).
    """
    if manifold_type == 'S1':
        return Hypersphere(1)
    elif manifold_type == 'S2':
        return Hypersphere(2)
    elif manifold_type == 'SO3':
        return SpecialOrthogonal(n=3)
    elif manifold_type == 'T2':
        return ProductManifold([Hypersphere(1), Hypersphere(1)])
    else:
        raise ValueError("Unsupported manifold type. Supported types are 'S1', 'S2', 'SO3' and 'T2'.")


def get_obs_from_G(manifold_type, G, sigma2, n_obs):
    """
    Sample noisy observations from a prior G on the given manifold.

    Parameters
    ----------
    manifold_type : str
    G : callable
        Prior sampler with signature G(n_samples) or object with G.sample(n_samples).
    sigma2 : float
        Noise variance for the Riemannian normal distribution.
    n_obs : int
        Number of observations to draw.

    Returns
    -------
    X : np.ndarray
        Noisy observations in extrinsic coordinates.
    """
    if hasattr(G, 'sample'):
        Theta = G.sample(n_obs)
    else:
        Theta = G(n_obs)
    manifold = get_manifold(manifold_type)
    return manifold.random_riemannian_normal(Theta, 1 / sigma2, n_obs)


def sq_loss(manifold, X, delta):
    """
    Mean squared geodesic distance between two sets of points.

    Parameters
    ----------
    manifold : geomstats manifold
    X : np.ndarray
        Ground-truth points.
    delta : np.ndarray
        Estimated (denoised) points.

    Returns
    -------
    float
        Mean squared geodesic distance.
    """
    return (manifold.metric.dist_broadcast(X, delta) ** 2).mean()


def uniform_points(manifold_type, N):
    """
    Return N approximately uniformly spread points on the manifold.

    Parameters
    ----------
    manifold_type : str
        'S1' or 'S2'.
    N : int
        Number of points.

    Returns
    -------
    np.ndarray
        Points in extrinsic coordinates.
    """
    if manifold_type == "S1":
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.stack((np.cos(theta), np.sin(theta)), axis=1)

    elif manifold_type == "S2":
        # Fibonacci sphere
        points = []
        phi = (1 + 5**0.5) / 2
        for i in range(N):
            z = 1 - 2 * (i + 0.5) / N
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / phi
            points.append((r * np.cos(theta), r * np.sin(theta), z))
        return np.array(points)
    else:
        raise ValueError("manifold must be 'S1' or 'S2'")


def parse_np_array(s):
    """Parse a numpy array from a bracketed string, e.g. '[1.0 2.0 3.0]'."""
    return np.fromstring(s.strip('[]'), sep=' ')
