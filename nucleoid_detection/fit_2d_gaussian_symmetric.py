import numpy as np
from scipy.optimize import least_squares


def _gaussian_symmetric(p, x, y):
    """Symmetric 2D Gaussian: bg + amplitude * 2^(-(r/hw)^2)."""
    bg, br, cx, cy, width = p
    xc = x - cx
    yc = y - cy
    return bg + br * np.power(2.0, -(xc**2 + yc**2) / (width / 2)**2)


def _residuals_symmetric(p, x, y, img):
    return (_gaussian_symmetric(p, x, y) - img).ravel()


def fit_2d_gaussian_symmetric(img, center_guess=None, width_guess=None, x=None, y=None):
    """Fit an image with a symmetric 2D Gaussian peak.

    Parameters
    ----------
    img : ndarray
        2D image to fit.
    center_guess : array-like of length 2, optional
        Initial guess for center (x, y).
    width_guess : float, optional
        Initial guess for FWHM.
    x, y : ndarray, optional
        Coordinate grids. Generated from image shape if not provided.

    Returns
    -------
    params : ndarray
        [background, amplitude, center_x, center_y, fwhm]
    model : ndarray
        Fitted model image.
    chisq : float
        Chi-squared value.
    """
    img = img.astype(float)
    dims = img.shape

    if x is None or y is None:
        x, y = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing='ij')

    if width_guess is None:
        width_guess = np.mean(dims) / 3
    assert np.isscalar(width_guess) or len(np.atleast_1d(width_guess)) == 1
    width_guess = float(np.atleast_1d(width_guess)[0])

    if center_guess is None:
        center_guess = np.array([(dims[0] - 1) / 2, (dims[1] - 1) / 2])
    center_guess = np.asarray(center_guess, dtype=float)

    bg = max(img.min(), 0)
    br = img.max() - bg

    S = 4
    p0 = [bg, br, center_guess[0], center_guess[1], width_guess]
    lb = [0, 0, x.min(), y.min(), width_guess / S]
    ub = [np.inf, np.inf, x.max(), y.max(), width_guess * S]

    result = least_squares(
        _residuals_symmetric, p0, args=(x, y, img),
        bounds=(lb, ub), method='trf'
    )
    params = result.x

    model = _gaussian_symmetric(params, x, y)
    chisq = np.sum((model - img)**2 / np.maximum(model, 1e-6))

    return params, model, chisq
