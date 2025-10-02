
import numpy as np
import time
import warnings
import os
import pickle

"""
TD(λ) experiments with diminishing step sizes (NO PLOTTING VERSION).
- Learning rate schedule: alpha_t = alpha_0 / (t + t0), with t0 = 1e7.
- This script runs experiments and saves results to pickle files only.
- No matplotlib imports; no figures are generated here.
"""

# ------------------------ Global / Experiment Parameters ------------------------
n_states = 15
n_actions = 5
feature_dim = 5

ALPHAS_CONST = [0.005, 0.01]
T0 = 10_000_000
alphas_to_test = [a_const * T0 for a_const in ALPHAS_CONST]

# Simulation scale
n_steps = 2_000_000
n_runs = 10

# Sweep lists
gammas_to_plot = [0.9]
lambdas = [0.1, 0.5, 0.9]

# Data I/O
data_dir = "td_lambda_data"

# ------------------------ MDP Definition (same as before) ------------------------
P = np.zeros((n_states, n_actions, n_states))
R = np.zeros((n_states, n_actions))

for s in range(n_states):
    for a in range(n_actions):
        if s >= 2:
            if a == 0:
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

# Uniform policy
pi = np.ones((n_states, n_actions)) / n_actions

# Induced transition under policy
P_pi = np.zeros((n_states, n_states))
for s in range(n_states):
    for a in range(n_actions):
        P_pi[s, :] += pi[s, a] * P[s, a, :]

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

# Stationary distribution (here uniform)
D_pi = np.diag(np.ones(n_states) / n_states)

# Reward vector under policy
r_pi = np.zeros(n_states)
for s in range(n_states):
    for a in range(n_actions):
        r_pi[s] += pi[s, a] * R[s, a]

# ------------------------ Helper Functions ------------------------
def compute_A_b(gamma, lambda_):
    """
    Build A, b, and the projection objects for computing distance to the solution set W*.

    - A = X^T D_pi (gamma * P_lambda - I) X
    - b = X^T D_pi r_lambda
    - w0 = -A^+ b         (center used for orthogonal projection)
    - P_ker = I - A^+ A   (projects onto ker(A))
    """
    I = np.eye(n_states)
    try:
        inv_term = np.linalg.inv(I - lambda_ * gamma * P_pi)
    except np.linalg.LinAlgError:
        inv_term = np.linalg.pinv(I - lambda_ * gamma * P_pi, rcond=1e-10)

    P_lambda = inv_term @ P_pi
    r_lambda = inv_term @ r_pi

    A = X.T @ D_pi @ (gamma * P_lambda - I) @ X
    b = X.T @ D_pi @ r_lambda

    try:
        A_pinv = np.linalg.pinv(A, rcond=1e-10)
    except np.linalg.LinAlgError:
        print(f"Warning: Pseudo-inverse failed (gamma={gamma}, lambda={lambda_}). Using zeros.")
        A_pinv = np.zeros_like(A)

    w0 = -A_pinv @ b
    P_ker = np.eye(feature_dim) - A_pinv @ A
    return A, b, w0, P_ker


def run_td_lambda(gamma, lambda_, alpha_0, w0, P_ker):
    """
    One TD(λ) run with diminishing step-size:
        alpha_t = alpha_0 / (t + T0)

    We track d(w_t, W*) via orthogonal projection using (w0, P_ker).
    """
    if not np.all(np.isfinite(w0)) or not np.all(np.isfinite(P_ker)):
        print(f"!!! Error: Non-finite w0 or P_ker (gamma={gamma}, lambda={lambda_}, alpha0={alpha_0}).")
        return np.full(n_steps, np.nan)

    w = np.zeros(feature_dim)
    dist_history_list = []
    state = np.random.randint(n_states)
    z = np.zeros(feature_dim)

    for step in range(n_steps):
        # Projected point (onto W* affine set)
        proj_w = w0 + P_ker @ (w - w0)
        if not np.all(np.isfinite(proj_w)):
            print(f"!!! Instability in proj_w at step {step} (gamma={gamma}, lambda={lambda_}, alpha0={alpha_0})")
            remaining_steps = n_steps - step
            if remaining_steps > 0:
                dist_history_list.extend([np.nan] * remaining_steps)
            warnings.warn("Run failed due to instability", RuntimeWarning)
            dist_history_list = dist_history_list[:n_steps]
            while len(dist_history_list) < n_steps:
                dist_history_list.append(np.nan)
            return np.array(dist_history_list)

        # Track distance to solution set
        dist = np.linalg.norm(w - proj_w)
        if not np.isfinite(dist):
            print(f"!!! Non-finite distance at step {step} (gamma={gamma}, lambda={lambda_}, alpha0={alpha_0})")
            remaining_steps = n_steps - step
            if remaining_steps > 0:
                dist_history_list.extend([np.nan] * remaining_steps)
            warnings.warn("Run failed due to instability", RuntimeWarning)
            dist_history_list = dist_history_list[:n_steps]
            while len(dist_history_list) < n_steps:
                dist_history_list.append(np.nan)
            return np.array(dist_history_list)

        dist_history_list.append(dist)

        # Sample action and features
        action = np.random.randint(n_actions)
        phi = X[state, :]

        # Next state sampling
        probs = P[state, action, :]
        if not np.isclose(np.sum(probs), 1.0):
            sum_probs = np.sum(probs)
            probs = probs / sum_probs if sum_probs > 0 else np.ones(n_states) / n_states
        try:
            next_state = np.random.choice(n_states, p=probs)
        except ValueError as e:
            print(f"ValueError sampling next state: {e}. Probs: {probs}, Sum: {np.sum(probs)}")
            next_state = np.random.randint(n_states)

        reward = R[state, action]

        # TD(λ) core
        next_phi = X[next_state, :]
        next_q = np.dot(next_phi, w)
        current_q = np.dot(phi, w)

        delta = reward + gamma * next_q - current_q
        z = gamma * lambda_ * z + phi

        # Diminishing step size
        alpha_t = alpha_0 / (step + T0)

        # Parameter update
        w_update = alpha_t * delta * z
        w += w_update

        # Numerical guard
        if np.isnan(w).any() or np.isinf(w).any() or np.max(np.abs(w)) > 1e12:
            print(f"!!! Instability post-update at step {step+1} (gamma={gamma}, lambda={lambda_}, alpha0={alpha_0})")
            remaining_steps = n_steps - (step + 1)
            if remaining_steps > 0:
                dist_history_list.extend([np.nan] * remaining_steps)
            warnings.warn("Run failed due to instability", RuntimeWarning)
            dist_history_list = dist_history_list[:n_steps]
            while len(dist_history_list) < n_steps:
                dist_history_list.append(np.nan)
            return np.array(dist_history_list)

        state = next_state

    # Final distance refresh for the last point
    if len(dist_history_list) == n_steps:
        proj_w = w0 + P_ker @ (w - w0)
        dist = np.linalg.norm(w - proj_w)
        if not np.isfinite(dist):
            dist = np.nan
        dist_history_list[-1] = dist

    return np.array(dist_history_list[:n_steps])


def run_experiments(gamma, lambda_, alpha_0):
    """
    Runs n_runs independent TD(λ) trajectories for given (gamma, lambda, alpha_0),
    returning mean/std across runs and the raw run matrix.
    """
    try:
        A, b, w0, P_ker = compute_A_b(gamma, lambda_)
        if not np.all(np.isfinite(w0)) or not np.all(np.isfinite(P_ker)):
            print(f"!!! compute_A_b produced NaN/Inf (gamma={gamma}, lambda={lambda_}).")
            return np.full(n_steps, np.nan), np.full(n_steps, np.nan), np.full((n_runs, n_steps), np.nan)
    except Exception as e:
        print(f"!!! Error during compute_A_b (gamma={gamma}, lambda={lambda_}): {e}")
        return np.full(n_steps, np.nan), np.full(n_steps, np.nan), np.full((n_runs, n_steps), np.nan)

    dist_runs = np.zeros((n_runs, n_steps))
    print(f"Running: gamma={gamma}, lambda={lambda_}, alpha0={alpha_0} ({n_runs} runs, {n_steps} steps, t0={T0})...")
    start_time_exp = time.time()
    valid_runs_count = 0

    with warnings.catch_warnings():
        warnings.simplefilter("always", category=RuntimeWarning)
        for run in range(n_runs):
            run_start_time = time.time()
            run_failed = False
            try:
                with warnings.catch_warnings(record=True) as w_list:
                    dist_history = run_td_lambda(gamma, lambda_, alpha_0, w0, P_ker)
                    if any(issubclass(item.category, RuntimeWarning) for item in w_list):
                        run_failed = True
                        print(f"  Run {run+1}/{n_runs} FAILED (RuntimeWarning)")
                run_end_time = time.time()

                if run_failed or np.any(np.isnan(dist_history)):
                    if not run_failed:
                        print(f"  Run {run+1}/{n_runs} FAILED (NaN detected) in {run_end_time - run_start_time:.2f} sec.")
                    dist_runs[run, :] = np.nan
                elif len(dist_history) != n_steps:
                    print(f"  Run {run+1}/{n_runs} incorrect length ({len(dist_history)}). Marked as failed.")
                    dist_runs[run, :] = np.nan
                else:
                    dist_runs[run, :] = dist_history
                    valid_runs_count += 1
                    print(f"  Run {run+1}/{n_runs} completed in {run_end_time - run_start_time:.2f} sec.")
            except Exception as e:
                run_end_time = time.time()
                print(f"  Run {run+1}/{n_runs} FAILED (Exception: {e}) in {run_end_time - run_start_time:.2f} sec.")
                dist_runs[run, :] = np.nan

    end_time_exp = time.time()
    print(f"Experiment (gamma={gamma}, lambda={lambda_}, alpha0={alpha_0}) finished in {(end_time_exp - start_time_exp)/60:.2f} minutes.")

    if valid_runs_count == 0:
        print(f"*** ALL RUNS FAILED for gamma={gamma}, lambda={lambda_}, alpha0={alpha_0} ***")
        dist_mean = np.full(n_steps, np.nan)
        dist_std = np.full(n_steps, np.nan)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dist_mean = np.nanmean(dist_runs, axis=0)
            dist_std = np.nanstd(dist_runs, axis=0)
        failed = n_runs - valid_runs_count
        if failed > 0:
            print(f"*** Warning: {failed}/{n_runs} runs failed due to instability for (gamma={gamma}, lambda={lambda_}, alpha0={alpha_0}) ***")

    max_std = np.nanmax(dist_std) if not np.all(np.isnan(dist_std)) else 0
    print(f"---- Max Std Dev (gamma={gamma}, lambda={lambda_}, alpha0={alpha_0}): {max_std:.4e} ----")
    return dist_mean, dist_std, dist_runs


def get_experiment_data_filename(gamma, lambda_, alpha_0):
    """Generate a unique filename for each configuration."""
    alpha_str = str(alpha_0).replace('.', 'p')
    gamma_str = str(gamma).replace('.', 'p')
    lambda_str = str(lambda_).replace('.', 'p')
    return f"td_lambda_g{gamma_str}_l{lambda_str}_a0{alpha_str}_n{n_runs}_steps{n_steps//1000}k.pkl"


def save_experiment_data(gamma, lambda_, alpha_0, dist_mean, dist_std, dist_runs):
    """Save results to a pickle file (only if mean is not all-NaN)."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = os.path.join(data_dir, get_experiment_data_filename(gamma, lambda_, alpha_0))
    data = {
        'gamma': gamma,
        'lambda': lambda_,
        'alpha_0': alpha_0,
        'n_steps': n_steps,
        'n_runs': n_runs,
        't0': T0,
        'dist_mean': dist_mean,
        'dist_std': dist_std,
        'dist_runs': dist_runs,
        'timestamp': time.time()
    }
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")
    return filename


# ------------------------ Main: run & save only (no plots) ------------------------
def main():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")

    overall_start_time = time.time()
    print("Running TD(λ) experiments (NO PLOTS).")
    print(f"Diminishing schedule: alpha_t = alpha_0 / (t + {T0})")

    for current_gamma in gammas_to_plot:
        print(f"\n=== Processing Gamma = {current_gamma} ===")
        for lambda_ in lambdas:
            print(f"  --- Lambda = {lambda_} ---")
            for alpha_0 in alphas_to_test:
                print(f"    alpha_0 = {alpha_0}")
                dist_mean, dist_std, dist_runs = run_experiments(current_gamma, lambda_, alpha_0)
                if not np.all(np.isnan(dist_mean)):
                    save_experiment_data(current_gamma, lambda_, alpha_0, dist_mean, dist_std, dist_runs)
                else:
                    print(f"    Skipped saving: all runs failed for (gamma={current_gamma}, lambda={lambda_}, alpha0={alpha_0}).")

    overall_end_time = time.time()
    print(f"\nTotal script execution time: {(overall_end_time - overall_start_time)/60:.2f} minutes.")


if __name__ == "__main__":
    main()
