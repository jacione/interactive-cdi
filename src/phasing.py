"""
The main phase retrieval solver class

Nick Porter
"""
import numpy as np
from scipy.ndimage import gaussian_filter

import src.utils as ut
import src.support as support


class Solver:
    def __init__(self, diffraction):
        self.diffraction = np.array(diffraction)
        self.imsize = self.diffraction.shape[0]
        self.support = support.Support2D(self.imsize)

        self.fs_image = self.diffraction * np.exp(2j * np.pi * np.random.random((self.imsize, self.imsize)))
        self.ds_image = ut.ifft(self.fs_image)
        self.ds_prev = np.copy(self.ds_image)

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

    def remove_twin(self):
        row, col = np.meshgrid(np.arange(self.imsize), np.arange(self.imsize))
        self.ds_image[row > self.imsize/2] *= 0
        self.ds_image[col > self.imsize/2] *= 0

    def reset(self):
        self.support = support.Support2D(self.imsize)
        self.fs_image = self.diffraction * np.exp(2j * np.pi * np.random.random((self.imsize, self.imsize)))
        self.ds_image = ut.ifft(self.fs_image)
        self.ds_prev = np.copy(self.ds_image)


if __name__ == "__main__":
    pass
