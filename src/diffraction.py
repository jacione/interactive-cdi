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

    def preprocess(self, sub_bkgd=False, do_binning=False, binning=1, do_cropping=False, cropping=1, do_gaussian=False,
                   sigma=1, do_thresh=False, thresh=1, do_vign=False, vsigma=1):
        image = np.copy(self.image)
        if sub_bkgd and self.bkgd is not None:
            image = image - self.bkgd
            image[image < 0] = 0
        if do_binning:
            image = ndi.zoom(image, 1/binning, order=1)
        if do_cropping and cropping < 1:
            n = int(image.shape[0] * (1-cropping) / 2)
            image = image[n:-n, n:-n]
        if do_gaussian:
            image = ndi.gaussian_filter(image, sigma=sigma)
        if sub_bkgd and self.bkgd is None:
            image = image - np.median(image)
            image[image < 0] = 0
        if do_thresh:
            image[image < np.quantile(image, thresh)] = 0
        if do_vign:
            step = image.shape[0] * 1j
            x, y = np.mgrid[-1:1:step, -1:1:step]
            mask = np.exp(-(x**2+y**2)/(2*vsigma**2))
            image = image * mask
        return image
