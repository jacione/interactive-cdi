import tkinter as tk

import numpy as np
from matplotlib import colors
import scipy.ndimage as ndi


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


def pad_to_size(arr, n_new):
    # Assuming square array
    n_old = arr.shape[0]
    pad = (n_new - n_old) // 2
    if 2*pad + n_old == n_new:
        return np.pad(arr, ((pad, pad), (pad, pad)))
    elif 2*pad + n_old == n_new - 1:
        return np.pad(arr, ((pad, pad+1), (pad, pad+1)))
    else:
        raise IndexError(f'Error padding to desired size: N_new={n_new}, N_old={n_old}, N_pad={pad}, '
                         f'N_out={2 * pad + n_old}')


def complex_composite_image(comp_img, dark_background=False):
    amp = normalize(np.abs(comp_img))
    phi = normalize(np.angle(comp_img))
    one = np.ones_like(amp)
    if dark_background:
        hsv = np.dstack((phi, one, amp))
    else:
        hsv = np.dstack((phi, amp, one))
    return colors.hsv_to_rgb(hsv)


def sig_round(x):
    rounded = np.format_float_positional(x, precision=1, unique=False, fractional=False, trim='-')
    if '.' in rounded:
        digit = int(rounded[-1])
    else:
        digit = int(rounded[0])
    power = int(np.floor(np.log10(float(rounded))))
    return digit, power


if __name__ == "__main__":
    pass
