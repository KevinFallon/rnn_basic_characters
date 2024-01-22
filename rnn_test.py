import unittest
import rnn
import numpy as np
from numpy.testing import assert_allclose

class TestRNN(unittest.TestCase):

    def test_init_params(self):
        np.random.seed(42)
        n_a, n_x, n_y = 3, 4, 5

        res = rnn.init_params(n_a, n_x, n_y)
        

        W_aa_exp = [
            [0.37454012, 0.95071431, 0.73199394],
            [0.59865848, 0.15601864, 0.15599452],
            [0.05808361, 0.86617615, 0.60111501],
        ]

        W_ax_exp = [
            [0.70807258, 0.02058449, 0.96990985, 0.83244264],
            [0.21233911, 0.18182497, 0.18340451, 0.30424224],
            [0.52475643, 0.43194502, 0.29122914, 0.61185289],
        ]

        W_ya_exp = [
            [0.13949386, 0.29214465, 0.36636184],
            [0.45606998, 0.78517596, 0.19967378],
            [0.51423444, 0.59241457, 0.04645041],
            [0.60754485, 0.17052412, 0.06505159],
            [0.94888554, 0.96563203, 0.80839735],
        ]

        b_a_exp = [
            [0.30461377],
            [0.09767211],
            [0.68423303],
        ]

        b_y_exp = [
            [0.44015249],
            [0.12203823],
            [0.49517691],
            [0.03438852],
            [0.9093204 ],
        ]

        assert_allclose(res.W_aa, W_aa_exp)
        assert_allclose(res.W_ax, W_ax_exp, atol=1e-8)
        assert_allclose(res.W_ya, W_ya_exp)
        assert_allclose(res.b_a, b_a_exp)
        assert_allclose(res.b_y, b_y_exp)


    def test_rnn_cell(self):
        np.random.seed(42)

        # n_a = 2, n_x = 4, n_y = 3
        p = rnn.Params(
            W_aa = [
                [0.2, 0.25],
                [.5, 0.1],
            ],
            W_ax = [
                [0.3, 0.1, 0.8, 0.4],
                [0.9, 0.3, 0.6, 0.2],
            ],
            W_ya = [
                [0.75, 0.4],
                [0.6, 0.4],
                [0.6, 0.6],
            ],
            b_a = [
                [0.6],
                [0.2],
            ],
            b_y = [
                [0.5],
                [0.9],
                [0.1],
            ],
        )

       # Only have 1 Training example for x_t and a_prev to make manual computation easier.
        x_t = [
            [0.9],
            [0.6],
            [0.4],
            [0.8],
        ]
        a_prev = [
            [0.4],
            [0.7],
        ]

        a, yt_hat = rnn.rnn_cell(p, a_prev, x_t)

        a_exp = [
            [0.9493345935462566],
            [0.9526788436890776],
        ]
        yt_hat_exp = [
            [0.33364447],
            [0.43167715],
            [0.23467838],
        ]
        assert_allclose(a, a_exp, atol=1e-8)
        # Python rounds the yt_hat_exp to 6 decimal places hence 1e-6
        assert_allclose(yt_hat, yt_hat_exp, atol=1e-6)


if __name__ == '__main__':
    unittest.main()