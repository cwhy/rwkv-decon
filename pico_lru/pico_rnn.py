from jax import numpy as np


def rnn_step(x, u, a, b, c, d):
    xp = a * x + b * u
    y = c * xp + d * u
    return xp, y


def init_params_(state_size: int, input_size: int, output_size: int):
    return {
        'a': np.zeros((state_size, state_size)),
        'b': np.zeros((state_size, input_size)),
        'c': np.zeros((output_size, state_size)),
        'd': np.zeros((output_size, input_size)),
    }
