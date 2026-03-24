import numpy as np
from scipy.optimize import least_squares


def _gaussian_rotated(p, x, y):
    """Rotated asymmetric 2D Gaussian."""
    bg, br, cx, cy, w1, w2, orient = p
    xc = x - cx
    yc = y - cy
    co = np.cos(orient)
    so = np.sin(orient)
    xr = co * xc + so * yc
    yr = -so * xc + co * yc
    return bg + br * np.power(2.0, -(xr**2 / (w1 / 2)**2 + yr**2 / (w2 / 2)**2))


def _residuals_rotated(p, x, y, img_w, sqrt_w):
    model = _gaussian_rotated(p, x, y) * sqrt_w
    return (model - img_w).ravel()


def fit_2d_gaussian_rotated(img, center_guess, width_guess, x=None, y=None, w=None):
    """Fit an image with a rotated, asymmetric 2D Gaussian peak.

    Parameters
    ----------
    img : ndarray
        2D image to fit.
    center_guess : array-like of length 2
        Initial guess for center (x, y).
    width_guess : array-like of length 2
        Initial guess for FWHM [small, large].
    x, y : ndarray, optional
        Coordinate grids.
    w : ndarray, optional
        Weight array (same shape as img). Default all ones.

    Returns
    -------
    params : ndarray
        [background, amplitude, center_x, center_y, fwhm_small, fwhm_large, orientation]
    model : ndarray
        Fitted model image (unweighted).
    """
    img = img.astype(float)
    dims = img.shape
    center_guess = np.asarray(center_guess, dtype=float)
    width_guess = np.asarray(width_guess, dtype=float)

    if x is None or y is None:
        x, y = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing='ij')

    if w is None:
        w = np.ones_like(img)

    sqrt_w = np.sqrt(w)
    img_w = img * sqrt_w

    bg = max(img.min(), 0)
    br = img.max() - bg

    S = 4
    mean_w = np.mean(width_guess)
    lb = [0, 0, x.min(), y.min(), mean_w / S, mean_w / S, -np.inf]
    ub = [np.inf, np.inf, x.max(), y.max(), mean_w * S, mean_w * S, np.inf]

    best_cost = np.inf
    best_p = None

    for angle in np.arange(0, np.pi + 0.01, np.pi / 4):
        p0 = [bg, br, center_guess[0], center_guess[1],
               width_guess[0], width_guess[1], angle]

        result = least_squares(
            _residuals_rotated, p0, args=(x, y, img_w, sqrt_w),
            bounds=(lb, ub), method='trf'
        )

        if result.cost < best_cost:
            best_cost = result.cost
            best_p = result.x.copy()

    params = best_p

    # Exchange small and large axis so params[4] <= params[5]
    if params[4] > params[5]:
        params[4], params[5] = params[5], params[4]
        params[6] = params[6] + np.pi / 2

    params[6] = params[6] % np.pi

    model = _gaussian_rotated(params, x, y)

    return params, model
