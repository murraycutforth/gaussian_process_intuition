import numpy as np
from scipy.special import kv, gamma


def squared_exponential(x1: np.ndarray, x2: np.ndarray, A: float, l: float):
    difference = x1[:, np.newaxis] - x2[np.newaxis, :]
    return A**2 * np.exp(- np.square(difference) / (2.0 * l**2))


def exp_sine_squared(x1: np.ndarray, x2: np.ndarray, A: float, l: float, period: float):
    difference = x1[:, np.newaxis] - x2[np.newaxis, :]
    return A * np.exp(- 2.0 / l**2 * np.square(np.sin(np.pi * np.abs(difference) / period)))


def matern(x1: np.ndarray, x2: np.ndarray, A: float, l: float, v: float):
    difference = x1[:, np.newaxis] - x2[np.newaxis, :] + np.finfo(float).eps
    distance = np.abs(difference)  # Assumes x1 and x2 are 1D coordinates so this is Euclidean distance
    t1 = A / (gamma(v) * 2**(v - 1.0))
    t2 = np.sqrt(2 * v) * distance / l
    t3 = kv(v, t2)
    return t1 * t2**v * t3


def rational_quadratic(x1: np.ndarray, x2: np.ndarray, A: float, l: float, alpha: float):
    difference = x1[:, np.newaxis] - x2[np.newaxis, :] + np.finfo(float).eps
    return A * np.power(1 + np.square(difference) / (2 * alpha * l**2), -1 * alpha)


def constant(x1: np.ndarray, x2: np.ndarray, val: float):
    return np.full(shape=(len(x1), len(x2)), fill_value=val)


def white(x1: np.ndarray, x2: np.ndarray, sigma: float):
    return sigma**2 * np.eye(len(x1), len(x2))