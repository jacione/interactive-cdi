"""
Classes of simulated objects to perform phase retrieval
"""
from pathlib import Path
from tkinter.filedialog import askopenfilenames

import numpy as np
import scipy.ndimage as ndi
from PIL import Image


RNG = np.random.default_rng(1234)
MAX_SIZE = 1024
INIT_DATA = f"{Path(__file__).parents[1].as_posix()}/example_data/ideal_1.tif"


def im_convert(image, ctr=None):
    # Convert to grayscale if necessary
    if image.ndim == 3:
        image = np.sum(image, axis=-1)

    # Find the center point in the diffraction pattern
    shape = np.array(image.shape)
    arr_ctr = shape // 2
    if ctr is None:
        blurred = ndi.gaussian_filter(image, 10)
        ctr = np.unravel_index(np.argmax(blurred, axis=None), image.shape)
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
        # Background subtraction only works when the scale of the background matches the scale of the data. Since the
        # images have been square-rooted (see the last line of im_convert) that means that the
        self.bkgd *= np.sqrt(self.n_images / self.n_bkgds)

    def preprocess(self, sub_bkgd=False, do_binning=False, binning=1, do_cropping=False, cropping=1, do_gaussian=False,
                   sigma=1, do_thresh=False, thresh=1):
        image = np.copy(self.image)
        if sub_bkgd:
            # Background subtraction
            if self.bkgd is not None:
                # Subtracting a known background is easy...
                pass
            else:
                # But what if you haven't actually measured a background image?
                pass
        if do_binning:
            # Bin the image.
            # Hint: the `binning` variable is the number of pixels binned in _each_ dimension. For example, if you want
            # to combine every 2x2 square of pixels, then set binning=2.
            pass
        if do_cropping and cropping < 1:
            # Crop the image.
            # Hint: the `cropping` variable is the ratio between the original and cropped image sizes in _each_
            # dimension. For example, cropping=0.5 will turn a 10x10 image into a 5x5.
            pass
        if do_gaussian:
            # Apply a gaussian filter to the image.
            # Hint: https://docs.scipy.org/doc/scipy/reference/ndimage.html
            pass
        if do_thresh:
            # Set an amplitude threshold for the image.
            pass
        # Remove the following line when you're ready to test!
        raise NotImplementedError("Preprocessing needs to be implemented in src/diffraction.py")
        return image
