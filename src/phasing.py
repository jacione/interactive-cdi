"""
The main phase retrieval solver class

Nick Porter
"""
import numpy as np
from scipy import ndimage as ndi

import src.utils as ut
import src.support as support


class Solver:
    def __init__(self, diffraction):
        self.diffraction = np.array(diffraction)
        self.imsize = self.diffraction.shape[0]
        self.ctr = self.imsize // 2
        self.support = support.Support2D(self.imsize)
        self.pixel_size = None

        self.fs_image = self.diffraction * np.exp(2j * np.pi * np.random.random((self.imsize, self.imsize))) + 0.0
        self.ds_image = ut.ifft(self.fs_image)
        self.ds_prev = np.copy(self.ds_image)

    def fft(self):
        # Hint: the HIO constraint will be much easier if you update self.ds_prev here!
        # Another hint: It might be useful to take a look at some of the functions in src/utils.py
        raise NotImplementedError("Some solver methods have not been implemented in src/phasing.py!")

    def modulus_constraint(self):
        raise NotImplementedError("Some solver methods have not been implemented in src/phasing.py!")

    def ifft(self):
        raise NotImplementedError("Some solver methods have not been implemented in src/phasing.py!")

    def er_constraint(self):
        raise NotImplementedError("Some solver methods have not been implemented in src/phasing.py!")

    def hio_constraint(self, beta=0.9):
        raise NotImplementedError("Some solver methods have not been implemented in src/phasing.py!")

    def er_iteration(self):
        self.fft()
        self.modulus_constraint()
        self.ifft()
        self.er_constraint()

    def hio_iteration(self, beta=0.9):
        self.fft()
        self.modulus_constraint()
        self.ifft()
        self.hio_constraint(beta)

    def shrinkwrap(self, sigma=1.0, threshold=0.1):
        self.support.shrinkwrap(self.ds_image, sigma, threshold)

    def set_scale(self, det_pitch, det_dist, wavelength):
        # The units get lumped into the 10**-6 term at the end: (10^-3 * 10^-9 / 10^-6) = 10^-6
        try:
            self.pixel_size = (det_dist * wavelength) / (det_pitch * self.imsize) * 10**-6
            assert self.pixel_size > 0
        except (ZeroDivisionError, AssertionError):
            self.pixel_size = None

    def gaussian_blur(self, sigma=2.0):
        self.ds_image = ut.normalize(ndi.gaussian_filter(np.abs(self.ds_image), sigma)) * \
                        np.exp(1j * ndi.gaussian_filter(np.angle(self.ds_image), sigma))

    def center(self):
        row, col = ndi.center_of_mass(self.support.array)
        rshift = int(self.ctr-row)
        cshift = int(self.ctr-col)
        self.support.array = np.roll(self.support.array, (rshift, cshift), axis=(0, 1))
        self.ds_image = np.roll(self.ds_image, (rshift, cshift), axis=(0, 1))

    def remove_twin(self):
        self.ds_image[self.ctr:] *= 0
        self.ds_image[:, self.ctr:] *= 0

    def reset(self):
        self.support = support.Support2D(self.imsize)
        self.fs_image = self.diffraction * np.exp(2j * np.pi * np.random.random((self.imsize, self.imsize))) + 0.0
        self.ds_image = ut.ifft(self.fs_image)
        self.ds_prev = np.copy(self.ds_image)


if __name__ == "__main__":
    pass
