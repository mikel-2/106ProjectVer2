import unittest
from ez_diffusion import forward_eq, inverse_eq, compute_error
import numpy as np

class TestEZDiffusion(unittest.TestCase):

    def test_forward_eq(self):
        a, v, t = 1.0, 1.0, 0.3
        Rpred, Mpred, Vpred = forward_eq(a, v, t)
        self.assertTrue(0 < Rpred < 1)
        self.assertTrue(Mpred > 0)
        self.assertTrue(Vpred > 0)

    def test_inverse_eq(self):
        Robs, Mobs, Vobs = 0.75, 0.5, 0.1
        v_est, a_est, t_est = inverse_eq(Robs, Mobs, Vobs)
        self.assertAlmostEqual(v_est, 1.0, delta=0.5)

    def test_zero_division_handling(self):
        self.assertTrue(np.isnan(inverse_eq(1, 0.3, 0.1)[0]))

if __name__ == '__main__':
    unittest.main()
