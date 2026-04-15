import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere

sphere = Hypersphere(dim=2)

plt.rcParams.update({
    'font.size': 12,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': 'Palatino',
})


def _grid_theta_phi_from_X_grid(X_grid):
    Xg = np.asarray(X_grid)
    res = int(np.sqrt(Xg.shape[0]))
    Xg = Xg.reshape(res, res, 3)
    x, y, z = Xg[..., 0], Xg[..., 1], Xg[..., 2]
    grid_theta = np.arccos(np.clip(z, -1.0, 1.0))
    grid_phi = np.mod(np.arctan2(y, x), 2 * np.pi)
    return grid_theta, grid_phi


def S2grid_fib(grid_resolution=50):
    """
    Fibonacci spiral (sunflower) grid on S^2.

    Parameters
    ----------
    grid_resolution : int
        Square root of total number of points (total = grid_resolution**2).

    Returns
    -------
    X_grid : np.ndarray, shape (N, 3)
    grid_theta : np.ndarray
        Colatitude angles.
    grid_phi : np.ndarray
        Longitude angles.
    """
    N = grid_resolution ** 2
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = np.arange(N)
    grid_theta = np.arccos(1 - (2 * i + 1) / N)
    grid_phi = (2 * np.pi * i / golden_ratio) % (2 * np.pi)
    X_grid = np.stack([
        np.sin(grid_theta) * np.cos(grid_phi),
        np.sin(grid_theta) * np.sin(grid_phi),
        np.cos(grid_theta),
    ], axis=-1)
    return X_grid, grid_theta, grid_phi


def S2grid(grid_resolution=50):
    """
    Regular latitude-longitude grid on S^2.

    Parameters
    ----------
    grid_resolution : int
        Number of grid lines in each direction.

    Returns
    -------
    X_grid : np.ndarray, shape (grid_resolution**2, 3)
    grid_theta : np.ndarray
    grid_phi : np.ndarray
    """
    grid_theta, grid_phi = np.meshgrid(
        np.linspace(0, np.pi, grid_resolution),
        np.linspace(0, 2 * np.pi, grid_resolution),
    )
    X_grid = np.stack([
        np.sin(grid_theta) * np.cos(grid_phi),
        np.sin(grid_theta) * np.sin(grid_phi),
        np.cos(grid_theta),
    ], axis=-1).reshape(-1, 3)
    return X_grid, grid_theta, grid_phi


def S2scatter(X, ax, color, alpha=.5, s=5, lw=.5, title=None, marker=None):
    """
    Scatter plot on a Mollweide projection.

    Parameters
    ----------
    X : np.ndarray, shape (n, 3)
        Extrinsic coordinates of points on the sphere.
    ax : matplotlib.axes.Axes (mollweide projection)
    color : matplotlib color
    alpha : float
    s : float
    lw : float
    title : str, optional
    """
    X_sph = sphere.extrinsic_to_spherical(X)
    theta = X_sph[:, 0]
    phi = X_sph[:, 1]
    phi_mw = phi - np.pi
    theta_mw = np.pi / 2 - theta
    if marker is None:
        ax.scatter(phi_mw, theta_mw, s=s, alpha=alpha, color=color)
    ax.grid(True, color='gray', lw=lw)
    if title is not None:
        ax.set_title(title, fontsize=15)


def S2plot_quiver(grid, vals, figax=None, scale=1, skip=1, cmap='Greens', cvals=None):
    """
    Quiver plot of a tangent vector field on S^2 (Mollweide projection).

    Parameters
    ----------
    grid : np.ndarray, shape (N, 3)
        Grid points from S2grid or S2grid_fib.
    vals : np.ndarray, shape (N, 3)
        Tangent vectors at each grid point.
    figax : tuple (fig, ax), optional
    scale : float
    skip : int
        Stride for sub-sampling the grid.
    cmap : str
    cvals : np.ndarray, optional
        Values used to color the arrows (defaults to arrow magnitude).
    """
    if figax is None:
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'mollweide'})
    else:
        fig, ax = figax

    grid_resolution = int(grid.shape[0] ** 0.5)
    grid_theta, grid_phi = _grid_theta_phi_from_X_grid(grid)

    e_theta = np.stack([
        np.cos(grid_theta) * np.cos(grid_phi),
        np.cos(grid_theta) * np.sin(grid_phi),
        -np.sin(grid_theta),
    ], axis=-1)
    e_phi = np.stack([
        -np.sin(grid_phi),
        np.cos(grid_phi),
        np.zeros_like(grid_phi),
    ], axis=-1)

    vals_reshaped = vals.reshape(grid_resolution, grid_resolution, 3)
    U = np.sum(vals_reshaped * e_phi, axis=-1)
    V = -np.sum(vals_reshaped * e_theta, axis=-1)

    C = cvals.reshape(grid_resolution, grid_resolution) if cvals is not None else np.sqrt(U**2 + V**2)

    ax.quiver(
        (grid_phi - np.pi)[::skip, ::skip],
        (np.pi / 2 - grid_theta)[::skip, ::skip],
        U[::skip, ::skip],
        V[::skip, ::skip],
        C[::skip, ::skip],
        scale=scale,
        cmap=cmap,
        alpha=0.85,
        angles='xy',
        scale_units='xy',
    )
    ax.grid(True, linestyle='--', alpha=0.4)
