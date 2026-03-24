import numpy as np
from .img_fourier_grid import img_fourier_grid


def img_smooth_mask(im, msk, fwhmp, mode='fft'):
    """Smooth a 2D image on pixels marked by a mask with a 2D Gaussian.

    Parameters
    ----------
    im : ndarray
        2D image.
    msk : ndarray
        Binary mask (True where valid).
    fwhmp : float
        FWHM of Gaussian kernel in pixels.
    mode : str
        'fft' (default) or 'conv2'.

    Returns
    -------
    sm : ndarray
        Smoothed image.
    """
    im = im.astype(float)
    msk = msk.astype(float)

    if mode == 'conv2':
        from scipy.signal import convolve2d
        L = int(np.ceil(3 * fwhmp))
        x, y = np.meshgrid(np.arange(-L, L + 1), np.arange(-L, L + 1), indexing='ij')
        k = np.exp(-4 * np.log(2) * (x**2 + y**2) / fwhmp**2)
        k = k / k.sum()

        im_sm = convolve2d(im * msk, k, mode='same')
        msk_sm = convolve2d(msk, k, mode='same')

    elif mode == 'fft':
        x, y = img_fourier_grid(im.shape)
        k = np.exp(-4 * np.log(2) * (x**2 + y**2) / fwhmp**2)
        k = k / k.sum()

        otf = np.fft.fft2(k)
        im_sm = np.real(np.fft.ifft2(np.fft.fft2(im * msk) * otf))
        msk_sm = np.real(np.fft.ifft2(np.fft.fft2(msk) * otf))

    else:
        raise ValueError("Unknown mode")

    T = 0.1
    m = msk_sm > T
    sm = np.zeros_like(im)
    sm[m] = im_sm[m] / msk_sm[m]

    return sm
