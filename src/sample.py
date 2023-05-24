"""
Classes of simulated objects to perform phase retrieval
"""
from abc import ABC, abstractmethod

import numpy as np
import skimage.draw as draw
import skimage.transform as transform
from matplotlib import pyplot as plt

import src.utils as ut


RNG = np.random.default_rng(1234)


class Sample(ABC):
    def __init__(self):
        self.image = None

    def update(self, arr):
        self.image = arr

    def diffract(self):
        return np.abs(ut.fft(self.image))**2

    def detect(self, accums=1, saturation=1.0, max_val=None, angle=0, order=3):
        output = np.zeros_like(self.diffract())
        for _ in range(int(accums)):
            img = self.diffract()
            if max_val is not None:
                img = max_val * ut.normalize(img)
                img = RNG.poisson(img)
            img = np.clip(saturation*img, 0, np.max(img))
            if max_val is not None:
                img = np.fix(max_val * ut.normalize(img))
            if angle != 0:
                img = transform.rotate(img, angle, order=order)
            output += img
        return np.sqrt(output)

    @abstractmethod
    def show(self):
        pass


class MeasuredData(Sample):
    def __init__(self):
        super().__init__()

    def show(self):
        plt.figure()
        plt.title("Ummm...")
        plt.show(block=False)
        plt.draw()


class RandomShapes(Sample):
    def __init__(self, size, seed=None, nshapes=2):
        super().__init__()
        half = size // 2
        qtr = size // 4
        mask = np.zeros((size, size))
        start = (qtr, qtr)
        end = (half + qtr - 1, half + qtr - 1)
        mask_x, mask_y = draw.rectangle(start, end)
        mask[mask_x, mask_y] = 1
        # Generate shapes based on defined parameters
        shapes, _ = draw.random_shapes(image_shape=(half, half), min_shapes=nshapes, max_shapes=nshapes,
                                       min_size=qtr / 2, num_channels=2, intensity_range=((0, 150), (0, 255)),
                                       allow_overlap=True, random_seed=seed)
        # Convert object amplitude range --> 0 < x < 1
        shapes = 1 - (shapes / 255)
        # Convert object phase range --> -pi < x < pi
        shapes[:, :, 1] = 2 * np.pi * shapes[:, :, 1] - np.pi
        # Combine amplitude and phase to get complex values
        shapes = shapes[:, :, 0] * np.exp(1j * shapes[:, :, 1])
        # Get rid of any negative zeros (they mess up the phase at small amplitudes)
        shapes = shapes + 0. + 0.j
        # Initialize a complex array for the object space
        original_object = np.zeros((size, size)) + 0j
        # Place the shapes in the object-space array
        original_object[mask_x, mask_y] = shapes

        self.image = original_object

    def show(self):
        plt.subplot(121, xticks=[], yticks=[], title="Amplitude")
        plt.imshow(np.abs(self.image), cmap="gray")
        plt.subplot(122, xticks=[], yticks=[], title="Phase")
        plt.imshow(np.angle(self.image), cmap="hsv", interpolation_stage="rgba", vmin=-np.pi, vmax=np.pi)
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
