"""
Classes of simulated objects to perform phase retrieval
"""
from abc import ABC, abstractmethod

import numpy as np

import utils as ut


class Sample(ABC):
    def __init__(self):
        self.direct = None
        self.fourier = None

    def update(self, arr):
        self.direct = arr
        self.fourier = ut.fft(arr)

    def diffract(self):
        return np.abs(self.fourier)**2
