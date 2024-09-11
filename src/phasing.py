"""
The main phase retrieval solver class

Nick Porter
"""
from tkinter import TclError
import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass

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

    def set_scale(self, det_pitch, det_dist, wavelength):
        # The units get lumped into the 10**-6 term at the end: (10^-3 * 10^-9 / 10^-6) = 10^-6
        try:
            self.pixel_size = (det_dist * wavelength) / (det_pitch * self.imsize) * 10**-6
            assert self.pixel_size > 0
        except (ZeroDivisionError, AssertionError):
            self.pixel_size = None

    def run_recipe(self, recipe):

        pass

    def fft(self):
        self.ds_prev = np.copy(self.ds_image)
        self.fs_image = ut.fft(self.ds_image)

    def modulus_constraint(self):
        self.fs_image = self.diffraction * np.exp(1j * np.angle(self.fs_image))

    def ifft(self):
        self.ds_image = ut.ifft(self.fs_image)

    def er_constraint(self):
        self.ds_image = self.ds_image * self.support.array + 0.0

    def er_iteration(self):
        self.fft()
        self.modulus_constraint()
        self.ifft()
        self.er_constraint()

    def hio_constraint(self, beta=0.9):
        self.ds_image = self.support.where(self.ds_image, self.ds_prev - beta*self.ds_image)

    def hio_iteration(self, beta=0.9):
        self.fft()
        self.modulus_constraint()
        self.ifft()
        self.hio_constraint(beta)

    def shrinkwrap(self, sigma=1.0, threshold=0.1):
        self.support.shrinkwrap(self.ds_image, sigma, threshold)

    def gaussian_blur(self, sigma=2.0):
        self.ds_image = ut.normalize(gaussian_filter(np.abs(self.ds_image), sigma)) * \
                        np.exp(1j * gaussian_filter(np.angle(self.ds_image), sigma))

    def center(self):
        row, col = center_of_mass(self.support.array)
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
