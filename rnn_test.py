import unittest
import rnn

class TestRNN(unittest.TestCase):

    def test_init_params(self):
        n_a, n_x, n_y = 3, 4, 5
        res = rnn.init_params(n_a, n_x, n_y)
        
        self.assertEqual(res.W_aa.shape, (n_a, n_a))
        self.assertEqual(res.W_ax.shape, (n_a, n_x))
        self.assertEqual(res.W_ya.shape, (n_y, n_a))
        self.assertEqual(res.b_a.shape, (n_a, 1))
        self.assertEqual(res.b_y.shape, (n_y, 1))


if __name__ == '__main__':
    unittest.main()