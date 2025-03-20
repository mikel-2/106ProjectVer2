import unittest
from src.ez_diffusion import forward_eq, inverse_eq, compute_error

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
        self.assertAlmostEqual(a_est, 1.0, delta=0.5)
        self.assertAlmostEqual(t_est, 0.3, delta=0.1)

    def test_compute_error(self):
        true_params = (1.0, 1.0, 0.3)
        est_params = (1.0, 1.0, 0.3)
        bias, squared_error = compute_error(true_params, est_params)
        self.assertEqual(np.sum(bias), 0)
        self.assertEqual(np.sum(squared_error), 0)

if __name__ == '__main__':
    unittest.main()
