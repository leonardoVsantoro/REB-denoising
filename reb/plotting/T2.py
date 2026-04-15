import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.product_manifold import ProductManifold

circle = Hypersphere(dim=1)
torus = ProductManifold([circle, circle])

plt.rcParams.update({
    'font.size': 12,
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': 'Palatino',
})


def T2grid(grid_resolution=50):
    """
    Regular grid on the flat torus T^2 = S^1 x S^1.

    Parameters
    ----------
    grid_resolution : int
        Number of grid lines in each direction.

    Returns
    -------
    X_grid : np.ndarray, shape (grid_resolution**2, 2, 2)
        Extrinsic embedding of T^2 into R^2 x R^2.
    phi : np.ndarray
        First angle grid.
    psi : np.ndarray
        Second angle grid.
    """
    phi, psi = np.meshgrid(
        np.linspace(-np.pi, np.pi, grid_resolution),
        np.linspace(-np.pi, np.pi, grid_resolution),
        indexing="xy",
    )
    X_grid = (np.asarray([[np.cos(phi), np.sin(phi)],
                           [np.cos(psi), np.sin(psi)]]).T).reshape(-1, 2, 2)
    return X_grid, phi, psi


def T2_scatter(X, ax=None, color='C0', s=10, alpha=0.25):
    """
    Scatter plot on the flat torus (phi-psi plane).

    Parameters
    ----------
    X : np.ndarray, shape (n, 2, 2)
        Points on T^2 in extrinsic coordinates.
    ax : matplotlib.axes.Axes, optional
    color : matplotlib color
    s : float
    alpha : float
    """
    X_phi = np.arctan2(X[:, 1, 0], X[:, 0, 0])
    X_psi = np.arctan2(X[:, 1, 1], X[:, 0, 1])

    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)

    ax.scatter(X_phi, X_psi, color=color, alpha=alpha, s=s)
    ax.grid(True, color='gray', lw=0.5)
    ax.axis('square')

    radian_ticks = [i * np.pi / 4 for i in range(-4, 5)]
    radian_ticklabels = [r'$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', r'$0$',
                         r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
    ax.set_xticks(radian_ticks)
    ax.set_yticks(radian_ticks)
    ax.set_xticklabels(radian_ticklabels)
    ax.set_yticklabels(radian_ticklabels)
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])


def T2plot_quiver(grid, vals, figax=None, scale=1, skip=1, cmap='Greens', cvals=None):
    """
    Quiver plot of a tangent vector field on T^2.

    Parameters
    ----------
    grid : np.ndarray, shape (N, 2, 2)
        Grid points from T2grid.
    vals : np.ndarray, shape (N, 2, 2)
        Tangent vectors at each grid point.
    figax : tuple (fig, ax), optional
    scale : float
    skip : int
    cmap : str
    cvals : np.ndarray, optional
    """
    if figax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, ax = figax

    grid_resolution = int(grid.shape[0] ** 0.5)

    grid_phi = np.arctan2(grid[:, 1, 0], grid[:, 0, 0]).reshape(grid_resolution, grid_resolution)
    grid_psi = np.arctan2(grid[:, 1, 1], grid[:, 0, 1]).reshape(grid_resolution, grid_resolution)

    tangent_phi = np.stack([-grid[:, 1, 0], grid[:, 0, 0]], axis=1)
    tangent_psi = np.stack([-grid[:, 1, 1], grid[:, 0, 1]], axis=1)

    d_phi = np.einsum('ni,ni->n', vals[:, 0, :], tangent_phi).reshape(grid_resolution, grid_resolution)
    d_psi = np.einsum('ni,ni->n', vals[:, 1, :], tangent_psi).reshape(grid_resolution, grid_resolution)

    C = cvals.reshape(grid_resolution, grid_resolution) if cvals is not None else np.sqrt(d_phi**2 + d_psi**2)

    ax.quiver(
        grid_phi[::skip, ::skip],
        grid_psi[::skip, ::skip],
        d_phi[::skip, ::skip],
        d_psi[::skip, ::skip],
        C[::skip, ::skip],
        angles='xy',
        scale_units='xy',
        scale=scale,
        cmap=cmap,
        alpha=0.85,
    )
    ax.set_xlabel(r'$\varphi$', fontsize=13)
    ax.set_ylabel(r'$\psi$', fontsize=13)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                  [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                  [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()


def T2_imshow(grid, vals, figax=None):
    """
    Heatmap of a scalar field on T^2.

    Parameters
    ----------
    grid : np.ndarray, shape (N, 2, 2)
    vals : np.ndarray, shape (N,)
    figax : tuple (fig, ax), optional
    """
    if figax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
    else:
        fig, ax = figax

    grid_resolution = int(grid.shape[0] ** 0.5)
    grid_phi = np.arctan2(grid[:, 1, 0], grid[:, 0, 0]).reshape(grid_resolution, grid_resolution)
    grid_psi = np.arctan2(grid[:, 1, 1], grid[:, 0, 1]).reshape(grid_resolution, grid_resolution)

    im = ax.pcolormesh(grid_phi, grid_psi, np.asarray(vals).reshape(grid_phi.shape),
                       shading="auto", cmap="Blues", alpha=0.5)
    ax.grid(True, color='gray', lw=0.5)
    ax.axis('square')
    ax.set_title(r'$\hat{f}(\cdot)$')
    ax.set_xlabel(r'$\varphi$', fontsize=13)
    ax.set_ylabel(r'$\psi$', fontsize=13)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                  [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                  [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
