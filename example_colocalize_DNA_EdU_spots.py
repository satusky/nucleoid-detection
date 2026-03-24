"""Colocalize DNA and EdU spots to determine EdU-positive nucleoids.

This script combines fitted DNA and EdU spot detections to determine how many
nucleoids are EdU-positive, i.e., how many detected DNA spots also had an EdU
spot close by. The decision is done by matching DNA and EdU spot pairs and
choosing all pairs with mutual distance below a certain threshold.

Requires that the nucleoids are determined before.

Part of "The TFAM to mtDNA ratio defines inner-cellular nucleoid
populations with distinct activity levels"
"""

import numpy as np
import tifffile
from nucleoid_detection import in_nucleus, reduce_events


def main():
    nucleus = tifffile.imread('data/HDFa_EdU-incubation-18h_ROI9_nucleus-mask_20nm-pixelsize.tiff')
    nucleus = nucleus > 0

    AB = 15e-9   # antibody size
    px = 20e-9   # 20nm pixel size
    T = 0.1e-6   # minimal distance between next
    TR = 1.0

    # Load spot data
    dna_data = np.load('data/dna_spots.npz', allow_pickle=True)
    dna = dna_data['spots']

    edu_data = np.load('data/edu_spots.npz', allow_pickle=True)
    edu = edu_data['spots']

    # Remove spots on nucleus (cols 4,5 are fitted center positions, 0-indexed)
    ix = in_nucleus(edu[:, 4:6], nucleus)
    edu = edu[~ix]
    ix = in_nucleus(dna[:, 4:6], nucleus)
    dna = dna[~ix]

    # Reduce by nearest neighbors too close (col 6 is FWHM)
    ix = reduce_events(edu[:, 4:6], edu[:, 6], T / px)
    edu = edu[ix]

    ix = reduce_events(dna[:, 4:6], dna[:, 6], T / px)
    dna = dna[ix]

    n_edu = edu.shape[0]
    n_dna = dna.shape[0]

    # Compute pairwise distances between EdU and DNA spots
    d = np.sqrt(
        (edu[:, 4:5] - dna[:, 4:5].T)**2 +
        (edu[:, 5:6] - dna[:, 5:6].T)**2
    ) * px

    # Spot sizes
    R1 = np.tile(edu[:, 6:7], (1, n_dna))    # size of EdU
    R2 = np.tile(dna[:, 6:7].T, (n_edu, 1))  # size of DNA

    # Subtract antibody-shell size
    R1 = np.maximum(0, R1 - AB)
    R2 = np.maximum(0, R2 - AB)

    # Relative distance
    dr = d / (R1 + R2)

    # EdU-positive DNA spots
    edu_positive = np.any(dr < TR, axis=0)
    n_edu_positive = edu_positive.sum()

    print(f'{n_dna} DNA spots, {n_edu} EdU spots, '
          f'{n_edu_positive} DNA spots with a single EdU spot within the valid distance')


if __name__ == '__main__':
    main()
