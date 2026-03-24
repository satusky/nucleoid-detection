import numpy as np
from skimage.morphology import dilation, erosion
from .img_smooth_mask import img_smooth_mask
from .omex_nearest_neighbour import omex_nearest_neighbour


def prepare_image(im, mask, px):
    """Prepare image for spot detection.

    Parameters
    ----------
    im : ndarray
        Raw image.
    mask : ndarray
        Nucleus mask (True inside nucleus).
    px : float
        Pixel size in meters.

    Returns
    -------
    im : ndarray
        Adjusted image for display.
    data : ndarray
        Background-subtracted smoothed image for detection.
    obj : ndarray
        Object mask (True outside nucleus, excluding borders).
    """
    im = im.astype(float)

    dilation_size = int(np.ceil(0.5e-6 / px))
    mask = dilation(mask, np.ones((dilation_size, dilation_size)))

    B = int(np.ceil(0.5e-6 / px))
    x, y = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing='ij')
    mask = mask | (x < B) | (x >= mask.shape[0] - B) | (y < B) | (y >= mask.shape[1] - B)

    obj = ~mask

    im_sm = img_smooth_mask(im, obj, 0.12e-6 / px)
    im_bg = img_smooth_mask(im, obj, 0.6e-6 / px)

    data = im_sm - 0.75 * im_bg
    data[~obj] = 0
    data[data < 0] = 0

    h_outside = im[~obj].max() if (~obj).any() else 1
    h_inside = im[obj].max() if obj.any() else 1
    im[~obj] = im[~obj] / h_outside * h_inside / 6

    obj = erosion(obj, np.ones((5, 5)))

    return im, data, obj


def in_nucleus(pos, nucleus):
    """Check if positions fall within the nucleus mask."""
    pos = np.round(pos).astype(int)
    pos[:, 0] = np.clip(pos[:, 0], 0, nucleus.shape[0] - 1)
    pos[:, 1] = np.clip(pos[:, 1], 0, nucleus.shape[1] - 1)
    return nucleus[pos[:, 0], pos[:, 1]]


def reduce_events(spots, fwhm, t):
    """Remove spots that are too close to each other, keeping the narrower one."""
    spots = spots.copy()
    fwhm = fwhm.copy()

    while True:
        nn = omex_nearest_neighbour(spots)

        if not np.any(nn[:, 0] < t):
            break

        i1 = np.argmin(nn[:, 0])
        i2 = int(nn[i1, 1])

        i = i2 if fwhm[i1] < fwhm[i2] else i1
        spots[i, :] = [np.inf, np.inf]

    idx = nn[:, 0] != np.inf
    return idx
