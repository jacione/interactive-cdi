"""
Classes of simulated objects to perform phase retrieval
"""
from pathlib import Path
from tkinter.filedialog import askopenfilenames

import numpy as np
# import skimage.draw as draw
import scipy.ndimage as ndi
from PIL import Image


RNG = np.random.default_rng(1234)
MAX_SIZE = 1024
INIT_DATA = f"{Path(__file__).parents[1].as_posix()}/example_data/ideal_1.tif"


def im_convert(image, ctr=None):
    # Convert to grayscale if necessary
    if image.ndim == 3:
        image = np.sum(image, axis=-1)

    # Find the brightest point in the image
    shape = np.array(image.shape)
    arr_ctr = shape // 2
    if ctr is None:
        ctr = np.unravel_index(np.argmax(image, axis=None), image.shape)
    image = np.roll(image, (arr_ctr[0] - ctr[0], arr_ctr[1] - ctr[1]), axis=(0, 1))

    # Crop to a square
    x, y = (shape - np.min(shape)) // 2
    if x != 0:
        image = image[x:-x]
    if y != 0:
        image = image[:, y:-y]

    if image.shape[0] > MAX_SIZE:
        scale_coeff = MAX_SIZE / np.array(image.shape)
        image = ndi.zoom(image, scale_coeff, order=5, prefilter=True)

    image = np.sqrt(image)

    return image, ctr


class LoadData:
    def __init__(self, filepath=INIT_DATA):
        self.image, self.ctr = im_convert(np.asarray(Image.open(filepath)))
        self.n_images = 1
        self.bkgd = None
        self.n_bkgds = 0

    def load_data(self):
        fs = askopenfilenames()
        if len(fs) == 0:
            return
        self.n_images = len(fs)
        if self.n_images == 1:
            self.image, self.ctr = im_convert(np.asarray(Image.open(fs[0])))
        else:
            imstack = [np.asarray(Image.open(f)) for f in fs]
            self.image, self.ctr = im_convert(np.sum(imstack, axis=0))
        self.bkgd = None
        self.n_bkgds = 0

    def load_bkgd(self):
        fs = askopenfilenames()
        if len(fs) == 0:
            return
        self.n_bkgds = len(fs)
        if self.n_bkgds == 1:
            self.bkgd, _ = im_convert(np.asarray(Image.open(fs[0])), self.ctr)
        else:
            imstack = [np.asarray(Image.open(f)) for f in fs]
            self.bkgd, _ = im_convert(np.sum(imstack, axis=0), self.ctr)
        # Background subtraction only works when the scale of the background matches the scale of the data.
        self.bkgd *= np.sqrt(self.n_images / self.n_bkgds)

    def preprocess(self, sub_bkgd=False, do_binning=False, binning=1, do_gaussian=False, sigma=1, do_thresh=False,
                   thresh=1, do_vign=False, vsigma=1):
        image = np.copy(self.image)
        if sub_bkgd and self.bkgd is not None:
            image = image - self.bkgd
        if do_binning:
            image = ndi.zoom(image, 1/binning, order=1)
        if do_gaussian:
            image = ndi.gaussian_filter(image, sigma=sigma)
        if do_vign:
            step = image.shape[0] * 1j
            x, y = np.mgrid[-1:1:step, -1:1:step]
            mask = np.exp(-(x**2+y**2)/(2*vsigma**2))
            image = image * mask
        if do_thresh:
            image[image < np.quantile(image, thresh)] = 0
        return image


# class RandomShapes:
#     def __init__(self, size, seed=None, nshapes=2):
#         half = size // 2
#         qtr = size // 4
#         mask = np.zeros((size, size))
#         start = (qtr, qtr)
#         end = (half + qtr - 1, half + qtr - 1)
#         mask_x, mask_y = draw.rectangle(start, end)
#         mask[mask_x, mask_y] = 1
#         # Generate shapes based on defined parameters
#         shapes, _ = draw.random_shapes(image_shape=(half, half), min_shapes=nshapes, max_shapes=nshapes,
#                                        min_size=qtr / 2, num_channels=2, intensity_range=((0, 150), (0, 255)),
#                                        allow_overlap=True, random_seed=seed)
#         # Convert object amplitude range --> 0 < x < 1
#         shapes = 1 - (shapes / 255)
#         # Convert object phase range --> -pi < x < pi
#         shapes[:, :, 1] = 2 * np.pi * shapes[:, :, 1] - np.pi
#         # Combine amplitude and phase to get complex values
#         shapes = shapes[:, :, 0] * np.exp(1j * shapes[:, :, 1])
#         # Get rid of any negative zeros (they mess up the phase at small amplitudes)
#         shapes = shapes + 0. + 0.j
#         # Initialize a complex array for the object space
#         original_object = np.zeros((size, size)) + 0j
#         # Place the shapes in the object-space array
#         original_object[mask_x, mask_y] = shapes
#
#         self.exit_wave = original_object
#
#     def get_amplitude(self, accums=1, saturation=1.0, max_val=None, angle=0, order=3):
#         output = np.zeros(self.exit_wave.shape)
#         for _ in range(int(accums)):
#             img = np.abs(ut.fft(self.exit_wave)) ** 2
#             if max_val is not None:
#                 img = max_val * ut.normalize(img)
#                 img = RNG.poisson(img)
#             img = np.clip(saturation*img, 0, np.max(img))
#             if max_val is not None:
#                 img = np.fix(max_val * ut.normalize(img))
#             if angle != 0:
#                 img = transform.rotate(img, angle, order=order)
#             output += img
#         return output
#
#     def show(self):
#         plt.subplot(121, xticks=[], yticks=[], title="Amplitude")
#         plt.imshow(np.abs(self.exit_wave), cmap="gray")
#         plt.subplot(122, xticks=[], yticks=[], title="Phase")
#         plt.imshow(np.angle(self.exit_wave), cmap="hsv", interpolation_stage="rgba", vmin=-np.pi, vmax=np.pi)
#         plt.tight_layout()
#         plt.show(block=False)
#         plt.draw()
