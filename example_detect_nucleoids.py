"""Detect DNA and EdU nucleoid spots in STED images.

This script shows how DNA and EdU spots are detected in STED images.
An initial threshold is applied, then Gaussian peak functions are fit to
each identified spot position to refine the positions.

Requires that the nucleus is determined before.

Part of "The TFAM to mtDNA ratio defines inner-cellular nucleoid
populations with distinct activity levels"
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage.measure import find_contours
from nucleoid_detection import omex_local_max, fit_nucleoids, prepare_image


def interactive_threshold(im, data, obj, px, t, fig_name):
    """Interactive threshold selection for spot detection.

    Keyboard instructions:
        escape/enter  - accept current threshold
        q             - quit without saving
        up/down       - increase/decrease threshold
        shift+up/down - larger threshold steps
        v/b           - adjust display contrast
        c             - toggle spot markers

    Parameters
    ----------
    im : ndarray
        Display image.
    data : ndarray
        Detection image.
    obj : ndarray
        Object mask.
    px : float
        Pixel size in meters.
    t : float
        Initial threshold (0-1).
    fig_name : str
        Figure title.

    Returns
    -------
    output : dict
        {'threshold': t, 'R': [min, max], 'positions': ndarray of (xi, yi)}
    quit : bool
        True if user pressed 'q'.
    """
    # Boundaries of object
    contours = find_contours(obj.astype(float), 0.5)

    R = [0, max(data.max(), 50)]
    dt = 0.01
    C = [0, 0.7 * im.max()]
    dc = C[1] / 10

    state = {'show': True, 'quit': False, 'xi': np.array([]), 'yi': np.array([])}

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    fig.canvas.manager.set_window_title(fig_name)

    def update():
        ax.clear()
        ax.imshow(im.T, origin='lower', cmap='hot', vmin=C[0], vmax=C[1], aspect='equal')
        ax.set_title(f't={t:.2f}')

        # Draw object boundaries
        for contour in contours:
            ax.plot(contour[:, 0], contour[:, 1], 'm', linewidth=0.8)

        # Apply threshold
        T = t * R[1] + (1 - t) * R[0]
        _, vi, xi, yi = omex_local_max(data, 'max', 8, T)

        # Keep only those inside object
        if len(xi) > 0:
            m = obj[xi, yi]
            xi = xi[m]
            yi = yi[m]
            vi = vi[m]

        state['xi'] = xi
        state['yi'] = yi

        if state['show'] and len(xi) > 0:
            marker_size = round(150e-9 / px)
            ax.plot(xi, yi, 'o', markersize=marker_size, color=[0, 0.4, 0],
                    markerfacecolor='none', markeredgewidth=1)

        ax.set_title(f't={t:.2f}, n={len(xi)}')
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal t, C
        if event.key == 'escape' or event.key == 'enter':
            plt.close(fig)
        elif event.key == 'q':
            print('Will quit. Nothing will be saved.')
            state['quit'] = True
            plt.close(fig)
        elif event.key == 'up':
            step = 10 * dt if (hasattr(event, 'modifiers') and 'shift' in str(event.modifiers)) else dt
            t = min(1, t + step)
            update()
        elif event.key == 'down':
            step = 10 * dt if (hasattr(event, 'modifiers') and 'shift' in str(event.modifiers)) else dt
            t = max(dt, t - step)
            update()
        elif event.key == 'v':
            C[1] += dc
            update()
        elif event.key == 'b':
            C[1] = max(C[0] + dc, C[1] - dc)
            update()
        elif event.key == 'c':
            state['show'] = not state['show']
            update()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    plt.show()

    positions = np.column_stack([state['xi'], state['yi']]) if len(state['xi']) > 0 else np.empty((0, 2))
    output = {'threshold': t, 'R': R, 'positions': positions}

    return output, state['quit']


def fit_spots(xi, yi, im, px, fwhm):
    """Fit Gaussian peaks to detected spot positions."""
    print(' fit spots')
    fwhmp = fwhm / px
    spots = fit_nucleoids(xi, yi, im, fwhmp)
    # Convert pixel columns to physical units
    spots[:, [4, 9, 10]] *= px
    # Prepend initial xi, yi
    spots = np.column_stack([xi, yi, spots])
    return spots


def main():
    # Load example data
    dna = tifffile.imread('data/HDFa_EdU-incubation-18h_ROI9_DNA_AlexaFluor594_20nm-px.tiff')
    edu = tifffile.imread('data/HDFa_EdU-incubation-18h_ROI9_EdU_StarRed_20nm-px.tiff')
    nucleus = tifffile.imread('data/HDFa_EdU-incubation-18h_ROI9_nucleus-mask_20nm-pixelsize.tiff')
    nucleus = nucleus > 0
    px = 20e-9  # 20nm pixel size

    # --- Threshold DNA spot detection and fit ---
    print('threshold DNA spots')
    threshold = 0.1

    im, d, obj = prepare_image(dna, nucleus, px)
    output, quit_flag = interactive_threshold(im, d, obj, px, threshold, 'DNA example image')

    if not quit_flag and len(output['positions']) > 0:
        output['spots'] = fit_spots(
            output['positions'][:, 0].astype(int),
            output['positions'][:, 1].astype(int),
            dna, px, 100e-9
        )
        np.savez('data/dna_spots.npz', **output)
        print(f'  Saved {len(output["positions"])} DNA spots')

    # --- Threshold EdU spot detection and fit ---
    print('threshold EdU spots')
    threshold = 0.2

    im, d, obj = prepare_image(edu, nucleus, px)
    output, quit_flag = interactive_threshold(im, d, obj, px, threshold, 'EdU example image')

    if not quit_flag and len(output['positions']) > 0:
        output['spots'] = fit_spots(
            output['positions'][:, 0].astype(int),
            output['positions'][:, 1].astype(int),
            edu, px, 100e-9
        )
        np.savez('data/edu_spots.npz', **output)
        print(f'  Saved {len(output["positions"])} EdU spots')


if __name__ == '__main__':
    main()
