import numpy as np
from .fit_2d_gaussian_symmetric import fit_2d_gaussian_symmetric
from .fit_2d_gaussian_rotated import fit_2d_gaussian_rotated


def fit_nucleoids(xi, yi, data, fwhmp):
    """Perform symmetric and asymmetric Gaussian fits on detected spots.

    Parameters
    ----------
    xi : ndarray
        Row positions of spots.
    yi : ndarray
        Column positions of spots.
    data : ndarray
        2D image data.
    fwhmp : float
        Expected FWHM in pixels.

    Returns
    -------
    r : ndarray
        Array of shape (n, 12) with symmetric fit params (5) followed by
        rotated fit params (7) for each spot.
        Symmetric: [bg, amplitude, center_x, center_y, fwhm]
        Rotated: [bg, amplitude, center_x, center_y, fwhm_small, fwhm_large, orientation]
    """
    B = int(np.ceil(1.2 * fwhmp))
    g = np.arange(-B, B + 1)
    xj, yj = np.meshgrid(g, g, indexing='ij')
    w = np.ones_like(xj, dtype=float)

    dims = data.shape
    n = len(xi)
    r = np.zeros((n, 12))

    ignore = (xi - B < 0) | (xi + B >= dims[0]) | (yi - B < 0) | (yi + B >= dims[1])

    for i in range(n):
        if ignore[i]:
            continue

        xk = int(xi[i])
        yk = int(yi[i])
        cut = data[xk + g[0]:xk + g[-1] + 1, yk + g[0]:yk + g[-1] + 1].astype(float)

        fp, _, _ = fit_2d_gaussian_symmetric(cut, [0, 0], fwhmp, xj, yj)
        fp[2] += xk
        fp[3] += yk

        fp2, _ = fit_2d_gaussian_rotated(cut, [0, 0], [fwhmp * 0.9, fwhmp * 1.1], xj, yj, w)
        fp2[2] += xk
        fp2[3] += yk

        r[i, :5] = fp
        r[i, 5:] = fp2

    return r
