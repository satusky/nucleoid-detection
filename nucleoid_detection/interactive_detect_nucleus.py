import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from .img_detect_nucleus import img_detect_nucleus


def interactive_detect_nucleus(imgs, p, name):
    """Interactively detect the nucleus in a cell image.

    Keyboard instructions:
        escape/enter  - close the figure, continue with processing
        q             - close the figure, indicates abort
        up/down       - increase/decrease threshold
        left/right    - less/more smoothing
        a/s           - larger/smaller minimal area size

    Parameters
    ----------
    imgs : list of ndarray
        List of images to cycle through.
    p : dict
        Parameters with keys: 'pad', 'smooth', 'threshold', 'area'.
    name : str
        Figure window name.

    Returns
    -------
    nucleus : ndarray
        Binary mask of detected nucleus.
    p : dict
        Updated parameters.
    quit : bool
        True if user pressed 'q' to abort.
    """
    F = 1.1
    A = 1.5
    state = {'counter': 0, 'quit': False, 'nucleus': None}

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.canvas.manager.set_window_title(name)

    def update():
        img = imgs[state['counter']]
        nucleus, _ = img_detect_nucleus(img, p)
        state['nucleus'] = nucleus

        ax.clear()
        ax.imshow(img.T, origin='lower', cmap='hot', aspect='equal')

        # Draw contours of detected nucleus
        contours = find_contours(nucleus.astype(float), 0.5)
        for contour in contours:
            ax.plot(contour[:, 0], contour[:, 1], 'w', linewidth=1.5)

        ax.set_title(f"smooth {p['smooth']:.0f}, threshold {p['threshold']:.2f}, area {p['area']:.0f}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'escape' or event.key == 'enter':
            plt.close(fig)
        elif event.key == 'q':
            state['quit'] = True
            plt.close(fig)
        elif event.key == 'down':
            p['threshold'] /= F
            update()
        elif event.key == 'up':
            p['threshold'] *= F
            update()
        elif event.key == 'left':
            p['smooth'] /= F
            update()
        elif event.key == 'right':
            p['smooth'] *= F
            update()
        elif event.key == 'a':
            p['area'] *= A
            update()
        elif event.key == 's':
            p['area'] /= A
            update()
        elif event.key == 'enter':
            state['counter'] = (state['counter'] + 1) % len(imgs)
            update()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    plt.show()

    return state['nucleus'], p, state['quit']
