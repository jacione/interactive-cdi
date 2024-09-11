import numpy as np
import scipy.ndimage as ndi

import src.utils as ut


class Support2D:
    def __init__(self, size, initial_oversampling=1.75):
        self.array = np.zeros((size, size), dtype="?")
        corner = int(round(size * (1 - 1 / initial_oversampling) / 2))
        self.array[corner:-corner, corner:-corner] = True

    def shrinkwrap(self, image, sigma=1.0, threshold=0.1):
        # Update the support region to more closely match the reconstructing image.
        # Remove the following line once you're ready to test!
        raise NotImplementedError("Implement shrinkwrap in src/support.py")

    def where(self, where_true, where_false):
        return np.where(self.array, where_true, where_false)


if __name__ == "__main__":
    pass