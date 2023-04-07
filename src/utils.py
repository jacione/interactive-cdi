import tkinter as tk

import numpy as np
from matplotlib import colors


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


def complex_composite_image(comp_img, dark_background=False):
    amp = normalize(np.abs(comp_img))
    phi = normalize(np.angle(comp_img))
    one = np.ones_like(amp)
    if dark_background:
        hsv = np.dstack((phi, one, amp))
    else:
        hsv = np.dstack((phi, amp, one))
    return colors.hsv_to_rgb(hsv)


def direct_to_photo_image(ds_image):
    amp = normalize(np.abs(ds_image))
    phi = normalize(np.angle(ds_image))
    one = np.ones_like(amp)
    image = 255 * colors.hsv_to_rgb(np.dstack((phi, one, amp)))
    height, width = image.shape[:2]
    data = f'P6 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')


def fourier_to_photo_image(fs_image):
    amp = normalize(np.log(np.abs(fs_image)+1))
    phi = normalize(np.angle(fs_image))
    one = np.ones_like(amp)
    image = 255 * colors.hsv_to_rgb(np.dstack((phi, one, amp)))
    height, width = image.shape[:2]
    data = f'P6 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')


def phase_to_photo_image(image):
    image = (image + np.pi) / (2*np.pi)
    one = np.ones_like(image)
    image = 255 * colors.hsv_to_rgb(np.dstack((image, one, one)))
    height, width = image.shape[:2]
    data = f'P6 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')


def amp_to_photo_image(image, mask=None):
    if mask is not None:
        image = np.clip(image, np.min(image), np.max(mask * image))
    image = 255 * normalize(image)
    height, width = image.shape
    data = f'P5 {width} {height} 255 '.encode() + image.astype(np.uint8).tobytes()
    return tk.PhotoImage(width=width, height=height, data=data, format='PPM')


if __name__ == "__main__":
    pass