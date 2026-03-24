import numpy as np
from scipy.spatial.distance import cdist


def omex_nearest_neighbour(pos, frames=None):
    """Compute nearest neighbour distances and indices for each position.

    Parameters
    ----------
    pos : ndarray
        Position array of shape (N, ndim).
    frames : ndarray, optional
        Frame labels of shape (N,). Distances are only computed between
        positions with the same frame number.

    Returns
    -------
    nn : ndarray
        Array of shape (N, 2) with [nearest_distance, nearest_index] per row.
    """
    n = pos.shape[0]
    nn = np.zeros((n, 2))

    if n == 0:
        return nn

    if frames is None:
        frames = np.ones(n, dtype=int)

    for kf in np.unique(frames):
        idx = np.where(frames == kf)[0]
        pos2 = pos[idx, :]
        n2 = pos2.shape[0]

        if n2 == 1:
            nn[idx[0], :] = [np.inf, idx[0]]
            continue

        d = cdist(pos2, pos2)
        np.fill_diagonal(d, np.inf)

        dmin_idx = np.argmin(d, axis=1)
        dmin_val = d[np.arange(n2), dmin_idx]

        # Map local indices back to global
        nn[idx, 0] = dmin_val
        nn[idx, 1] = idx[dmin_idx]

    return nn
