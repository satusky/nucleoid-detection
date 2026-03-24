import numpy as np


def omex_local_max(stack, mode='max', connection=8, threshold=-np.inf):
    """Find local maxima or minima in a 2D data stack.

    Parameters
    ----------
    stack : ndarray
        2D data array.
    mode : str
        'max' (default) or 'min'.
    connection : int
        4 or 8 (default) for neighbor connectivity.
    threshold : float
        Lower threshold for maxima (or upper for minima).

    Returns
    -------
    idx : ndarray
        Linear indices of found extrema.
    vi : ndarray
        Values at those positions.
    xi : ndarray
        Row positions.
    yi : ndarray
        Column positions.
    """
    assert stack.ndim == 2, "Parameter stack must be 2D matrix!"

    if mode == 'min':
        stack = -stack.copy()
    elif mode == 'max':
        stack = stack.copy()
    else:
        raise ValueError("Unknown mode!")

    # Find local maxima by comparing with shifted versions
    msk = (stack > threshold)
    msk &= (stack > np.roll(stack, 1, axis=0))
    msk &= (stack > np.roll(stack, -1, axis=0))
    msk &= (stack > np.roll(stack, 1, axis=1))
    msk &= (stack > np.roll(stack, -1, axis=1))

    if connection == 8:
        msk &= (stack > np.roll(np.roll(stack, 1, axis=0), 1, axis=1))
        msk &= (stack > np.roll(np.roll(stack, -1, axis=0), 1, axis=1))
        msk &= (stack > np.roll(np.roll(stack, 1, axis=0), -1, axis=1))
        msk &= (stack > np.roll(np.roll(stack, -1, axis=0), -1, axis=1))

    xi, yi = np.where(msk)
    idx = np.ravel_multi_index((xi, yi), stack.shape)
    vi = stack[xi, yi]

    # Sort descending
    order = np.argsort(-vi)
    idx = idx[order]
    vi = vi[order]
    xi = xi[order]
    yi = yi[order]

    if mode == 'min':
        vi = -vi

    return idx, vi, xi, yi
