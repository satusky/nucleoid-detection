"""Detect the cell nucleus in STED images.

This script shows how the cell nucleus was detected in 2-color confocal and
STED nucleoid measurements. Both STED channels are given to an interactive
nucleus detection where thresholds can be adjusted.

Part of "The TFAM to mtDNA ratio defines inner-cellular nucleoid
populations with distinct activity levels"
"""

import tifffile
import numpy as np
from nucleoid_detection import interactive_detect_nucleus


def main():
    # Load example data
    dna = tifffile.imread('data/HDFa_EdU-incubation-18h_ROI9_DNA_AlexaFluor594_20nm-px.tiff').astype(float)
    edu = tifffile.imread('data/HDFa_EdU-incubation-18h_ROI9_EdU_StarRed_20nm-px.tiff').astype(float)

    # Nucleus detection parameters
    params = {'pad': 20, 'smooth': 30, 'threshold': 0.3, 'area': 30000}

    # Interactive nucleus detection
    nucleus, params, quit_flag = interactive_detect_nucleus(
        [dna + edu, dna, edu], params, 'Example data'
    )

    if not quit_flag and nucleus is not None:
        # Write nucleus mask
        tifffile.imwrite(
            'data/HDFa_EdU-incubation-18h_ROI9_nucleus-mask_20nm-pixelsize.tiff',
            (nucleus * 255).astype(np.uint8)
        )
        print("Nucleus mask saved.")


if __name__ == '__main__':
    main()
