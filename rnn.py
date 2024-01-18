import numpy as np
from dataclasses import dataclass
import utils


'''

Typically just need to follow these steps to solve a ML problem

1. Obtain training data
2. Define the model
3. Define a loss fuction
4. Run through the training data, calculating loss from the ideal value
5. Calculate gradients for that loss and use an optimzer to adjust the variables to fit the data.
6. Evaluate your results.

'''


@dataclass
class Params:
    W_aa: np.ndarray
    W_ax: np.ndarray
    W_ya: np.ndarray
    b_a: np.ndarray
    b_y: np.ndarray


def init_params(n_a, n_x, n_y):
    return Params(
        np.random.random((n_a, n_a)),
        np.random.random((n_a, n_x)),
        np.random.random((n_y, n_a)),
        np.random.random((n_a, 1)),
        np.random.random((n_y, 1)),
    )

def rnn_cell(p: Params, a_prev: np.ndarray, x_t: np.ndarray):
    print(p.W_aa)
    print(p.W_ax)
    print(p.W_ya)
    print(p.b_a)
    print(p.b_y)

    a = np.dot(p.W_aa, a_prev) + np.dot(p.W_ax, x_t) + p.b_a
    a = np.tanh(a)

    yt_hat = np.dot(p.W_ya, a) + p.b_y
    yt_hat = utils.softmax(yt_hat)

    return a, yt_hat # TODO: may need to return additional values for backwards pass