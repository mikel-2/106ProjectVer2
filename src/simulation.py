import numpy as np
import pandas as pd
from ez_diffusion import forward_eq, inverse_eq, compute_error

def simulate_data(a, v, t, N):
    Rpred, Mpred, Vpred = forward_eq(a, v, t)

    # Safe Robs values
    Robs = np.random.binomial(N, Rpred) / N
    Robs = np.clip(Robs, 0.001, 0.999)  # Guaranteed valid range

    Mobs = np.random.normal(Mpred, np.sqrt(Vpred / N))
    Vobs = np.random.gamma((N - 1) / 2, 2 * Vpred / (N - 1))

    return Robs, Mobs, Vobs

def run_simulation():
    np.random.seed(42)  # For reproducibility
    N_values = [10, 40, 4000]
    results = []

    for N in N_values:
        for _ in range(1000):  # 1000 repetitions per N
            true_params = (np.random.uniform(0.5, 2),  # a
                           np.random.uniform(0.5, 2),  # v
                           np.random.uniform(0.1, 0.5))  # t

            Robs, Mobs, Vobs = simulate_data(*true_params, N)

            # Final Robs safeguard
            if Robs <= 0 or Robs >= 1:
                continue  # Skip invalid values

            est_params = inverse_eq(Robs, Mobs, Vobs)
            if np.isnan(est_params[0]):
                continue  # Skip invalid estimates

            bias, squared_error = compute_error(true_params, est_params)

            results.append({
                "N": N,
                "True_a": true_params[0], "True_v": true_params[1], "True_t": true_params[2],
                "Est_a": est_params[0], "Est_v": est_params[1], "Est_t": est_params[2],
                "Bias_a": bias[0], "Bias_v": bias[1], "Bias_t": bias[2],
                "Squared_Error": np.sum(squared_error)
            })

    # Save results to CSV
    pd.DataFrame(results).to_csv("data/results.csv", index=False)
    print(f"✅ Successfully generated {len(results)} rows of data")

if __name__ == "__main__":
    run_simulation()
