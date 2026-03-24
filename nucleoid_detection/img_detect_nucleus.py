import numpy as np
from skimage.morphology import dilation, convex_hull_object
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from .img_smooth import img_smooth


def img_detect_nucleus(d, p):
    """Detect the nucleus in an image using heuristics.

    Parameters
    ----------
    d : ndarray
        2D image.
    p : dict
        Parameters with keys: 'pad', 'smooth', 'threshold', 'area'.

    Returns
    -------
    m : ndarray
        Binary mask of detected nucleus.
    a : float
        Area of the detected nucleus in pixels.
    """
    P = int(p['pad'])
    dp = np.pad(d.astype(float), P, mode='symmetric')
    ds, _, _ = img_smooth(dp, p['smooth'])
    ds = ds[P:-P, P:-P]

    T = p['threshold']
    m = ds > (ds.max() * T + ds.min() * (1 - T))

    m = binary_fill_holes(m)

    # Keep only regions with area >= p['area']
    labeled = label(m)
    S = p['area']
    m_filtered = np.zeros_like(m)
    for region in regionprops(labeled):
        if region.area >= S:
            m_filtered[labeled == region.label] = True
    m = m_filtered

    # Convex hull of each object
    if m.any():
        m = convex_hull_object(m)

    # Dilate slightly
    m = dilation(m, np.ones((5, 5)))

    a = float(m.sum())

    return m, a
