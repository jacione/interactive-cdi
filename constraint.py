"""
Classes of constraints to perform phase retrieval
"""
from abc import ABC, abstractmethod

import numpy as np

import utils as ut


RNG = np.random.default_rng()


class ConstraintError(ValueError):
    pass


class Constraint(ABC):
    def __init__(self):
        self._name = ""
        self.total_reps = 1
        pass

    def __add__(self, other):
        if not isinstance(other, Constraint):
            raise ConstraintError
        return CompositeConstraint(self, other)

    def __rmul__(self, reps):
        if not isinstance(reps, int):
            raise ConstraintError
        return RepeatConstraint(self, reps)

    def __str__(self):
        return self._name

    @abstractmethod
    def apply(self, arr, data):
        pass


class CompositeConstraint(Constraint):
    def __init__(self, alg_1, alg_2):
        super().__init__()
        self.total_reps = alg_1.total_reps + alg_2.total_reps
        self._name = f"({alg_1}+{alg_2})"
        self.alg_1 = alg_1
        self.alg_2 = alg_2

    def apply(self, arr, data):
        return self.alg_2.apply(self.alg_1.apply(arr, data), data)


class RepeatConstraint(Constraint):
    def __init__(self, alg, reps):
        super().__init__()
        self.total_reps = reps * alg.total_reps
        self._name = f"{reps}*{alg}"
        self.alg = alg
        self.reps = reps

    def apply(self, arr, data):
        for _ in range(self.reps):
            arr = self.alg.apply(arr, data)
        return arr


class Fourier(Constraint):
    def __init__(self):
        super().__init__()
        self._name = "Fourier"

    def apply(self, arr, data):
        # forward-propagate the object into Fourier space
        diff = ut.fft(arr)
        # Apply the known diffraction value to those pixels
        diff = data * np.exp(1j * np.angle(diff))
        # back-propagate the object back to direct space
        return ut.ifft(diff)


class PartialFourier(Constraint):
    def __init__(self, fraction):
        super().__init__()
        self._name = "PartialFourier"
        self.fraction = fraction

    def apply(self, arr, data):
        # forward-propagate the object into Fourier space
        diff = ut.fft(arr)
        # Randomly select a fraction of the pixels to "fix" in the diffraction pattern
        update_pixels = RNG.random(arr.shape) < self.fraction
        # Apply the known diffraction value to those pixels
        diff[update_pixels] = data[update_pixels] * np.exp(1j * np.angle(diff[update_pixels]))
        # back-propagate the object back to direct space
        return ut.ifft(diff)


class HIO(Constraint):
    def __init__(self, beta=0.9):
        super().__init__()
        self._name = "HIO"
        self.beta = beta

    def apply(self, arr, data):
        pass


class ER(Constraint):
    def __init__(self):
        super().__init__()
        self._name = "ER"

    def apply(self, arr, data):
        pass


if __name__ == "__main__":
    recipe = 10*(HIO()+ER())
    print(recipe)
    pass

