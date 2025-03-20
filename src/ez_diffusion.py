import numpy as np

#forward equations for model
def forward_eq(a, v, t):
    y = np.exp(-a * v)
    Rpred = 1 / (y + 1)
    Mpred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    Vpred = (a / (2 * v ** 3)) * (1 - 2 * a * v * y - y ** 2) / ((y + 1) ** 2)
    return Rpred, Mpred, Vpred

#inverse equations for model
def inverse_eq(Robs, Mobs, Vobs):
    L = np.log(Robs / (1 - Robs))
    v_est = np.sign(Robs - 0.5) * 4 * np.sqrt(L * (Robs ** 2 * L - Robs * L + Robs - 0.5) / Vobs)
    a_est = L / v_est
    t_est = Mobs - (a_est / (2 * v_est)) * (1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est))
    return v_est, a_est, t_est

#error calculation
def compute_error(true_params, est_params):
    bias = np.array(true_params) - np.array(est_params)
    squared_error = np.square(bias)
    return bias, squared_error
