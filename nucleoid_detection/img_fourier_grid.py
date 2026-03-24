import numpy as np


def img_fourier_grid(dims):
    """Generate a Fourier-compatible grid with center of mass at (0, 0).

    Functions computed on this grid, when used in convolution via fft2,
    will not produce any shift. So (0, 0) will be at the origin (0,0)
    and everything else as expected.

    Parameters
    ----------
    dims : tuple
        Shape of the grid (1D, 2D, or 3D).

    Returns
    -------
    grids : tuple of ndarrays
        Coordinate grids (xi,) or (xi, yi) or (xi, yi, zi).
    """
    ndim = len(dims)
    if ndim not in (1, 2, 3):
        raise ValueError("Unsupported dimensionality!")

    axes = [np.arange(d, dtype=float) for d in dims]
    grids = np.meshgrid(*axes, indexing='ij')

    result = []
    for g in grids:
        g = np.fft.ifftshift(g)
        g = g - g.flat[0]
        result.append(g)

    return tuple(result)
