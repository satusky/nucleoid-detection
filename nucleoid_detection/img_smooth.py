import numpy as np
from .img_fourier_grid import img_fourier_grid


def img_smooth(img, s, kernel_type=0):
    """Blur a 2D image with a kernel of size s in pixels.

    Properties: No shift, total sum stays constant.

    Parameters
    ----------
    img : ndarray
        2D image.
    s : float or array-like
        Kernel size parameter(s) in pixels.
        - type 0: Gaussian FWHM (scalar or [sx, sy])
        - type 1: disk radius
        - type 2: square half-length
        - type 3: ring size
        - type 4: [fwhm_x, fwhm_y, rotation_angle_rad]
    kernel_type : int
        0 = Gaussian with FWHM s (default)
        1 = disk of radius s
        2 = square of length 2*s+1
        3 = ring
        4 = rotated Gaussian

    Returns
    -------
    smooth : ndarray
        Smoothed image.
    kernel_max : float
        Maximum value of the kernel.
    kernel : ndarray
        The smoothing kernel used.
    """
    img = img.astype(float)
    s = np.atleast_1d(np.asarray(s, dtype=float))
    if len(s) == 1:
        s = np.array([s[0], s[0]])

    x, y = img_fourier_grid(img.shape)

    if kernel_type == 0:
        kernel = np.power(2.0, -(x**2 / (s[0] / 2)**2 + y**2 / (s[1] / 2)**2))
        kernel = kernel / kernel.sum()
    elif kernel_type == 1:
        kernel = ((x**2 + y**2) <= s[0]**2).astype(float)
    elif kernel_type == 2:
        kernel = ((np.abs(x) <= s[0]) & (np.abs(y) <= s[1])).astype(float)
    elif kernel_type == 3:
        r2 = x**2 / (s[0] / 2)**2 + y**2 / (s[1] / 2)**2
        kernel = r2 * np.power(2.0, -r2)
        kernel = kernel / kernel.sum()
    elif kernel_type == 4:
        cp = np.cos(s[2])
        sp = np.sin(s[2])
        xr = cp * x + sp * y
        yr = -sp * x + cp * y
        kernel = np.power(2.0, -(xr**2 / (s[0] / 2)**2 + yr**2 / (s[1] / 2)**2))
        kernel = kernel / kernel.sum()
    else:
        raise ValueError("Unknown type!")

    smooth = np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(kernel)))
    kernel_max = kernel.max()

    return smooth, kernel_max, kernel
