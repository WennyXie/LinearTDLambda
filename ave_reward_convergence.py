import numpy as np
import time
import warnings
import os
import pickle

"""
Average-Reward TD(λ) with diminishing step sizes (NO PLOTTING).
- Step-size schedule (match discounted-setting convention):
    beta_t  = beta_0 / (t + T0)   with beta_0 = beta_const * T0
    alpha_t = c_delta * beta_t
  where T0 = 10^7 (large offset) so early-time behavior ≈ constant stepsize beta_const.
- This script runs experiments and saves results only (dist_mean, dist_std, dist_runs).
- No matplotlib imports and no figure generation here.
"""

# ------------------------ Global / Experiment Parameters ------------------------
n_states = 15           # |S|
n_actions = 5           # |A|
feature_dim = 5         # Dimension of features in X

# ----------------- Stepsize parameters -----------------
beta_const = 0.01          # “Effective” constant stepsize you want to mimic
T0 = 10_000_000            # offset for diminishing stepsize (t0 = 1e7)
beta_0 = beta_const * T0   # scale so that beta_0 / T0 ≈ beta_const
c_deltas = [0.1, 0.5, 1.0] # α_t = c_δ * β_t
# --------------------------------------------------------

n_steps = 2_000_000     # Trajectory length per run
n_runs = 10             # Number of independent runs to average

# Number of simulations to estimate A, b (for projection; follows your original surrogate)
compute_A_b_simulations = 50_000

# Data I/O
lambdas = [0.1, 0.5, 0.9]
data_dir = "avg_reward_td_lambda_data"
save_data = True        # Save results to pickle
load_data = True        # If True, try to load saved results first

# ------------------------ MDP Definition ------------------------
P = np.zeros((n_states, n_actions, n_states))
R = np.zeros((n_states, n_actions))

for s in range(n_states):
    for a in range(n_actions):
        if s >= 2:
            if a == 0:
                if s - 1 >= 0:
                    P[s, a, s - 1] = 1.0
            else:
                if s - 2 >= 0:
                    P[s, a, s - 2] = 1.0
                else:
                    P[s, a, 0] = 1.0
        elif s == 1:
            P[s, a, 0] = 1.0
        elif s == 0:
            P[s, a, :] = 1.0 / n_states
        R[s, a] = 1.0 if s == 0 else 0.0

# Uniform behavior policy
pi = np.ones((n_states, n_actions)) / n_actions

# Feature matrix
X = np.array([
    [0.07, 0.11, 0.18, 0.14, 0.61],
    [0.13, 0.19, 0.32, 0.26, 0.45],
    [0.11, 0.17, 0.28, 0.22, 0.39],
    [0.24, 0.36, 0.60, 0.48, 0.84],
    [0.18, 0.28, 0.46, 0.36, 1.00],
    [0.20, 0.30, 0.50, 0.40, 1.06],
    [0.31, 0.47, 0.78, 0.62, 1.45],
    [0.29, 0.45, 0.74, 0.58, 1.39],
    [0.42, 0.64, 1.06, 0.84, 1.84],
    [0.40, 0.62, 1.02, 0.80, 1.78],
    [0.47, 0.73, 1.20, 0.94, 2.39],
    [0.53, 0.81, 1.34, 1.06, 2.23],
    [0.58, 0.90, 1.48, 1.16, 2.78],
    [0.60, 0.92, 1.52, 1.20, 2.84],
    [0.67, 1.03, 1.70, 1.34, 3.45]
])
assert X.shape[1] == feature_dim, "feature_dim does not match X.shape[1]"
assert X.shape[0] == n_states, "n_states does not match X.shape[0]"

# ------------------------ Helper Functions ------------------------
def compute_A_b_avg_reward(c_delta, beta_0_base, lambda_, current_P, current_R, current_X, current_pi):
    """
    Monte Carlo surrogate for A and b in the augmented parameter space [J; w].
    This matches your original logic (uses base constants beta_0_base and
    alpha_eff_base = c_delta * beta_0_base), solely to build a stable projection
    center w0 and kernel projector P_ker. The diminishing schedule is applied
    only in the main TD(λ) runs (not here).
    """
    alpha_eff_base = c_delta * beta_0_base
    print(f"Computing A, b for projection (c_delta={c_delta}, beta0={beta_0_base}, lambda={lambda_})...")
    start_time = time.time()
    n_sim = compute_A_b_simulations

    A_sum = np.zeros((1 + feature_dim, 1 + feature_dim))
    b_sum = np.zeros(1 + feature_dim)

    state = np.random.randint(n_states)
    z = np.zeros(feature_dim)

    for _ in range(n_sim):
        phi = current_X[state, :]
        action = np.random.randint(n_actions)
        probs = current_P[state, action, :]
        if not np.isclose(np.sum(probs), 1.0):
            probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones(n_states) / n_states
        next_state = np.random.choice(n_states, p=probs)
        reward = current_R[state, action]
        next_phi = current_X[next_state, :]

        z = lambda_ * z + phi

        A_y = np.zeros((1 + feature_dim, 1 + feature_dim))
        A_y[0, 0] = -beta_0_base
        A_y[1:, 0] = -alpha_eff_base * z
        A_y[1:, 1:] = alpha_eff_base * np.outer(z, next_phi - phi)

        b_y = np.zeros(1 + feature_dim)
        b_y[0] = beta_0_base * reward
        b_y[1:] = alpha_eff_base * reward * z

        A_sum += A_y
        b_sum += b_y
        state = next_state

    A = A_sum / n_sim
    b = b_sum / n_sim

    try:
        A_pinv = np.linalg.pinv(A, rcond=1e-10)
    except np.linalg.LinAlgError:
        print("Warning: Pseudo-inverse calculation failed for A. Using zeros.")
        A_pinv = np.zeros_like(A)

    w0 = -A_pinv @ b
    P_ker = np.eye(1 + feature_dim) - A_pinv @ A

    end_time = time.time()
    print(f"A, b computation finished in {end_time - start_time:.2f} seconds.")
    return A, b, w0, P_ker


def run_avg_reward_td_lambda(lambda_, c_delta, beta_0_base, w0, P_ker):
    """
    One run of average-reward TD(λ) with diminishing step sizes:
        beta_t  = beta_0_base / (t + T0)
        alpha_t = c_delta * beta_t
    Tracks d(ŵ_t, W*) where ŵ_t = [J_hat; w].
    """
    w = np.zeros(feature_dim)
    J_hat = 0.0
    z = np.zeros(feature_dim)
    dist_history_list = []
    state = np.random.randint(n_states)

    for step in range(n_steps):
        # Projection onto affine solution set using (w0, P_ker)
        w_hat = np.concatenate(([J_hat], w))
        proj_w_hat = w0 + P_ker @ (w_hat - w0)
        dist = np.linalg.norm(w_hat - proj_w_hat)
        dist_history_list.append(dist)

        # Transition sample
        phi = X[state, :]
        action = np.random.randint(n_actions)
        reward = R[state, action]
        probs = P[state, action, :]
        if not np.isclose(np.sum(probs), 1.0):
            probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones(n_states) / n_states
        next_state = np.random.choice(n_states, p=probs)
        next_phi = X[next_state, :]

        # TD error and trace
        delta = reward - J_hat + np.dot(next_phi, w) - np.dot(phi, w)
        z = lambda_ * z + phi

        # Diminishing step sizes
        beta_t = beta_0_base / (step + T0)
        alpha_t = c_delta * beta_t

        # Updates
        J_hat += beta_t * (reward - J_hat)
        w += alpha_t * delta * z

        # Numerical guard
        current_w_hat = np.concatenate(([J_hat], w))
        if np.isnan(current_w_hat).any() or np.isinf(current_w_hat).any():
            print(f"!!! Instability detected at step {step} for c_delta={c_delta}, beta0={beta_0_base}, lambda={lambda_}")
            remaining = n_steps - (step + 1)
            if remaining > 0:
                dist_history_list.extend([np.nan] * remaining)
            warnings.warn(f"Run failed due to instability (c_delta={c_delta}, beta0={beta_0_base}, lambda={lambda_})", RuntimeWarning)
            return np.array(dist_history_list)

        state = next_state

    return np.array(dist_history_list)


def get_experiment_data_filename(c_delta, beta_0_base, lambda_):
    """Generate a unique filename for each experiment configuration."""
    return f"avg_reward_td_lambda_c{str(c_delta).replace('.','p')}_b{str(beta_0_base).replace('.','p')}_l{str(lambda_).replace('.','p')}_n{n_runs}_steps{n_steps//1000}k.pkl"


def save_experiment_data(c_delta, beta_0_base, lambda_, dist_mean, dist_std, dist_runs):
    """Save experiment data to a pickle file."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = os.path.join(data_dir, get_experiment_data_filename(c_delta, beta_0_base, lambda_))
    data = {
        'c_delta': c_delta,
        'beta_0': beta_0_base,
        'beta_const': beta_const,
        'lambda_': lambda_,
        'n_steps': n_steps,
        'n_runs': n_runs,
        't0': T0,
        'dist_mean': dist_mean,
        'dist_std': dist_std,
        'dist_runs': dist_runs,
        'timestamp': time.time()
    }

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {filename}")
    return filename


def load_experiment_data(c_delta, beta_0_base, lambda_):
    """Load experiment data from a pickle file (if present)."""
    filename = os.path.join(data_dir, get_experiment_data_filename(c_delta, beta_0_base, lambda_))
    if not os.path.exists(filename):
        print(f"No data file found for c_delta={c_delta}, beta_0={beta_0_base}, lambda_={lambda_}. File not found: {filename}")
        return None, None, None

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {filename}")
        print(f"  Original data: c_delta={data['c_delta']}, beta_0={data['beta_0']}, beta_const={data.get('beta_const')}, "
              f"lambda_={data['lambda_']}, steps={data['n_steps']}, runs={data['n_runs']}, t0={data.get('t0')}")
        return data['dist_mean'], data['dist_std'], data['dist_runs']
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def run_experiments(lambda_, c_delta, beta_0_base):
    """
    Try to load, otherwise run fresh experiments under diminishing step sizes.
    Returns (dist_mean, dist_std, dist_runs).
    """
    dist_mean = dist_std = dist_runs = None
    if load_data:
        dist_mean, dist_std, dist_runs = load_experiment_data(c_delta, beta_0_base, lambda_)

    if dist_mean is None or dist_std is None or dist_runs is None:
        # Compute projection objects (w0, P_ker) using your original surrogate A,b
        _, _, w0, P_ker = compute_A_b_avg_reward(c_delta, beta_0_base, lambda_, P, R, X, pi)

        # Run n_runs trajectories
        dist_runs = np.zeros((n_runs, n_steps))
        print(f"Running AvgReward TD(λ) [diminishing]: c_delta={c_delta}, "
              f"beta0={beta_0_base} (≈const {beta_const}), lambda={lambda_} "
              f"({n_runs} runs, {n_steps} steps, t0={T0})...")
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for run in range(n_runs):
                run_start = time.time()
                dist_history = run_avg_reward_td_lambda(lambda_, c_delta, beta_0_base, w0, P_ker)
                L = min(n_steps, len(dist_history))
                dist_runs[run, :L] = dist_history[:L]
                if L < n_steps:
                    dist_runs[run, L:] = np.nan
                run_end = time.time()
                if np.isnan(dist_runs[run, -1]) and L < n_steps:
                    print(f"  Run {run+1}/{n_runs} FAILED (instability) in {run_end - run_start:.2f} sec.")
                else:
                    print(f"  Run {run+1}/{n_runs} completed in {run_end - run_start:.2f} sec.")
        end = time.time()
        print(f"Experiment (c_delta={c_delta}, beta0={beta_0_base}, lambda={lambda_}) finished in {(end - start)/60:.2f} minutes.")

        # Aggregate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dist_mean = np.nanmean(dist_runs, axis=0)
            dist_std = np.nanstd(dist_runs, axis=0)

        failed_runs = np.sum(np.isnan(dist_runs[:, -1]))
        if failed_runs > 0:
            print(f"*** Warning: {failed_runs}/{n_runs} runs failed due to instability for c_delta={c_delta}, beta0={beta_0_base}, lambda={lambda_} ***")

        max_std = np.nanmax(dist_std) if not np.all(np.isnan(dist_std)) else 0
        print(f"---- Max Std Dev for c_delta={c_delta}, beta0={beta_0_base}, lambda={lambda_}: {max_std:.4e} ----")

        if save_data:
            save_experiment_data(c_delta, beta_0_base, lambda_, dist_mean, dist_std, dist_runs)

    return dist_mean, dist_std, dist_runs


# ------------------------ Main: run & save only (no plotting) ------------------------
def main():
    if save_data and not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")

    total_start_time = time.time()
    print("Running Average-Reward TD(λ) experiments (NO PLOTS).")
    print(f"Diminishing schedule: beta_t = beta_0 / (t + {T0}), alpha_t = c_delta * beta_t")
    print(f"With scaling: beta_const = {beta_const}, so initial effective step ≈ beta_const")

    all_data = {}
    for c_delta in c_deltas:
        for lambda_ in lambdas:
            dist_mean, dist_std, dist_runs = run_experiments(lambda_, c_delta, beta_0)
            all_data[(c_delta, beta_0, lambda_)] = {
                'dist_mean': dist_mean,
                'dist_std': dist_std,
                'dist_runs': dist_runs
            }

    total_end_time = time.time()
    print(f"Total execution time: {(total_end_time - total_start_time)/60:.2f} minutes.")

    # Optionally save a comprehensive bundle for convenience
    if save_data:
        comprehensive_data = {
            'parameters': {
                'n_states': n_states,
                'n_actions': n_actions,
                'feature_dim': feature_dim,
                'beta_const': beta_const,
                'beta_0': beta_0,
                'c_deltas': c_deltas,
                'n_steps': n_steps,
                'n_runs': n_runs,
                'lambdas': lambdas,
                't0': T0
            },
            'experiment_data': all_data,
            'timestamp': time.time()
        }
        comprehensive_filename = os.path.join(
            data_dir, f'all_avg_reward_td_lambda_diminishing_{n_steps//1000}k_bconst{str(beta_const).replace(".","p")}_n{n_runs}.pkl'
        )
        with open(comprehensive_filename, 'wb') as f:
            pickle.dump(comprehensive_data, f)
        print(f"All data saved to {comprehensive_filename}")

    return all_data


if __name__ == "__main__":
    main()
