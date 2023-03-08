import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog as fd
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
from pathlib import Path
import shutil


def fft(arr, modulus=False):
    """Perform a correctly shifted fast Fourier transform"""
    if modulus:
        return np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr))))**2
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(arr)))


def ifft(arr, modulus=False):
    if modulus:
        return np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(arr))))**2
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(arr)))


def log(comp_arr):
    """Take the logarithm of the amplitude of a complex array while keeping the phase intact."""
    amp = np.abs(comp_arr)
    phi = np.angle(comp_arr)
    return np.log(amp+1) * np.exp(1j*phi)


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def pad_to_size(arr, N_new):
    # Assuming square array
    N = arr.shape[0]
    pad = (N_new - N) // 2
    if 2*pad + N == N_new:
        return np.pad(arr, ((pad, pad), (pad, pad)))
    elif 2*pad + N == N_new - 1:
        return np.pad(arr, ((pad, pad+1), (pad, pad+1)))
    else:
        raise IndexError(f'Error padding to desired size: N_new={N_new}, N_old={N}, N_pad={pad}, N_out={2*pad + N}')


def comp_to_rgb(comp_img):
    amp = normalize(np.abs(comp_img))
    phi = normalize(np.angle(comp_img))
    one = np.ones_like(amp)
    hsv = np.dstack((phi, one, amp))
    return colors.hsv_to_rgb(hsv)


def add_phase_wheel(ax, corner, size):
    # TODO
    pass


def get_save_dir():
    root = tk.Tk()
    root.withdraw()
    return fd.askdirectory()


def get_file():
    """Opens a GUI dialog to select a single file"""
    root = tk.Tk()
    root.withdraw()
    try:
        return fd.askopenfilename()
    except KeyError:
        return None


def save_file(ext):
    """Opens a save-as dialog"""
    root = tk.Tk()
    root.withdraw()
    try:
        return fd.asksaveasfilename(defaultextension=ext, filetypes=[(ext.lower(), ext.upper())])
    except KeyError:
        return


def open_image(as_square=False):
    root = tk.Tk()
    root.withdraw()
    try:
        fname = fd.askopenfilename()
    except KeyError:
        return
    img = imread(fname, as_gray=True)
    if as_square:
        size = np.min(img.shape)
        img = resize(img, (size, size))
    return img


def save_array_as_image(img, cmap='plasma'):
    """Saves a 2D array as an image using matplotlib. """
    root = tk.Tk()
    root.withdraw()
    try:
        fname = fd.asksaveasfilename(defaultextension='png', filetypes=[('PNG', 'png')])
    except KeyError:
        return
    plt.imsave(fname, img, cmap=cmap)


def save_gif(images):
    """Saves a 3D array (image stack) as an animated gif."""
    root = tk.Tk()
    root.withdraw()
    try:
        fname = fd.asksaveasfilename(defaultextension='gif')
    except KeyError:
        return
    dname = Path(__file__).parent
    temp = dname / 'temp'
    temp.mkdir()
    [plt.imsave(f'{temp}/im_{n:03}.png', im, cmap='gray', vmin=0, vmax=1) for n, im in enumerate(images)]
    images = [Image.open(f'{temp}/im_{n:03}.png') for n in range(len(images))]
    images[0].save(fname, save_all=True, append_images=images[1:], duration=300, loop=0)
    shutil.rmtree(temp)


if __name__ == "__main__":
    yy, xx = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    plt.imshow(screw_uz(xx, yy), origin='lower', cmap='hsv')
    plt.show()
