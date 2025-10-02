import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
import warnings
import os
import pickle

# --- Simulation Parameters ---
n_states = 15  # Define for context, not used in computation here
n_actions = 5
feature_dim = 5
ALPHAS_CONST = [0.005, 0.01]
T0 = 10_000_000
alphas_to_test = [a_const * T0 for a_const in ALPHAS_CONST]
n_steps = 1500000
n_runs = 10    # Number of runs (MUST match the saved data)
gammas_to_plot = [0.9]  # List of gammas for rows
lambdas = [0.1, 0.5, 0.9]  # Lambdas for columns

data_dir = "td_lambda_data"

# --- Helper Functions ---

def get_experiment_data_filename(gamma, lambda_, alpha_0):
    """Generate the unique filename for loading (use 2000k for existing files)"""
    alpha_str = str(alpha_0).replace('.', 'p')
    gamma_str = str(gamma).replace('.', 'p')
    lambda_str = str(lambda_).replace('.', 'p')
    return f"td_lambda_g{gamma_str}_l{lambda_str}_a0{alpha_str}_n{n_runs}_steps2000k.pkl"

def load_experiment_data(gamma, lambda_, alpha_0):
    """Load experiment data from a pickle file and truncate to n_steps"""
    filename = os.path.join(data_dir, get_experiment_data_filename(gamma, lambda_, alpha_0))

    if not os.path.exists(filename):
        print(f"Data file not found: {filename}")
        return None, None, None

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if 'dist_mean' not in data or 'dist_std' not in data or 'dist_runs' not in data:
            print(f"Error: Missing required data keys in {filename}")
            return None, None, None

        # Truncate data to n_steps (1500000)
        if len(data['dist_mean']) < n_steps or data['dist_runs'].shape[1] < n_steps:
            print(f"Error: Data in {filename} has fewer than {n_steps} steps.")
            return None, None, None

        dist_mean = data['dist_mean'][:n_steps]
        dist_std = data['dist_std'][:n_steps]
        dist_runs = data['dist_runs'][:, :n_steps]

        if data.get('n_runs') != n_runs:
            print(f"Warning: n_runs mismatch in {filename}. Expected {n_runs}, found {data.get('n_runs')}.")

        print(f"Data successfully loaded from {filename} (truncated to {n_steps} steps)")
        return dist_mean, dist_std, dist_runs

    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None, None, None

# --- Main Plotting Function ---
def main():
    # --- Plotting Setup ---
    plt.rcParams.update({
        'font.size': 26, 'axes.labelsize': 26, 'axes.titlesize': 26,
        'legend.fontsize': 14, 'xtick.labelsize': 22, 'ytick.labelsize': 22,
        'figure.titlesize': 28
    })
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'd']
    
    reporting_interval = 150000

    # --- Check if data directory exists ---
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found. Cannot load data.")
        return

    # --- Create a Single Figure with 1x3 Subplots for Lambdas ---
    print("\n--- Starting Plot Generation from Saved Data ---")
    overall_start_time = time.time()

    fig, axes = plt.subplots(1, len(lambdas), figsize=(6 * len(lambdas), 5))
    if len(lambdas) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    steps = np.arange(n_steps)
    figure_has_data = False

    # --- Use a single gamma value ---
    current_gamma = gammas_to_plot[0]
    print(f"\n=== Processing γ = {current_gamma} ===")

    # --- Loop through lambdas (columns) ---
    for col_idx, lambda_ in enumerate(lambdas):
        ax = axes[col_idx]
        print(f"  --- Plotting subplot for λ = {lambda_} ---")
        current_subplot_min = np.inf
        current_subplot_max = -np.inf
        subplot_has_data = False

        # --- Loop through alphas to plot ---
        for i, alpha_0 in enumerate(alphas_to_test):
            print(f"    Loading data for α = {alpha_0}...")
            dist_mean, dist_std, _ = load_experiment_data(current_gamma, lambda_, alpha_0)

            if dist_mean is None or dist_std is None:
                print(f"    Skipping plot for alpha={alpha_0} as data could not be loaded.")
                ax.text(0.5, 0.5 - i*0.1, f'α={alpha_0}: No Data',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color=colors[i % len(colors)], fontsize=12, alpha=0.5)
                continue

            dist_mean = dist_mean[:n_steps]
            dist_std = dist_std[:n_steps]

            if np.all(np.isnan(dist_mean)):
                print(f"    Skipping plot for alpha={alpha_0} as loaded data indicates failed runs.")
                ax.text(0.5, 0.5 - i*0.1, f'α={alpha_0}: Failed Runs',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color=colors[i % len(colors)], fontsize=12)
                continue

            if alpha_0 == 0.01:
                print(f"\n    --- Values for γ={current_gamma}, λ={lambda_}, α={alpha_0} ---")
                
                # Print the initial value at Step 0
                initial_value = dist_mean[0]
                print(f"      Step {0:<9,}: Mean Distance = {initial_value:.6f}")

                # Loop for subsequent milestones
                for step_milestone in range(reporting_interval, n_steps + 1, reporting_interval):
                    index_to_report = step_milestone - 1
                    if index_to_report < len(dist_mean):
                        value_at_step = dist_mean[index_to_report]
                        print(f"      Step {step_milestone:<9,}: Mean Distance = {value_at_step:.6f}")
                print("    --------------------------------------------\n")

            subplot_has_data = True
            figure_has_data = True

            # --- Update subplot min/max ---
            valid_mean_indices = ~np.isnan(dist_mean)
            if np.any(valid_mean_indices):
                current_curve_min = np.min(dist_mean[valid_mean_indices])
                current_curve_max = np.max(dist_mean[valid_mean_indices])
                current_subplot_min = min(current_subplot_min, current_curve_min)
                current_subplot_max = max(current_subplot_max, current_curve_max)

                valid_std_indices = ~np.isnan(dist_std)
                combined_valid = valid_mean_indices & valid_std_indices
                if np.any(combined_valid):
                    upper_bound = (dist_mean + dist_std)[combined_valid]
                    current_curve_max_with_std = np.nanmax(upper_bound)
                    if np.isfinite(current_curve_max_with_std):
                        current_subplot_max = max(current_subplot_max, current_curve_max_with_std)

            # Plot mean line (this runs for ALL alphas)
            color = colors[i % len(colors)]
            ax.plot(steps, dist_mean,
                    linewidth=3.5, color=color, label=f'α_0={alpha_0/T0}', zorder=i+2)

        # --- Subplot Formatting ---
        ax.set_title(rf'$\lambda$={lambda_}', fontsize=26)

        if col_idx == 0:
            ax.set_ylabel(rf'$\gamma$={current_gamma}' + '\n' + r'$d({w}_t, W^*)$', 
                          fontsize=26, labelpad=10)

        ax.legend(loc='best', fontsize=14)

        if subplot_has_data and np.isfinite(current_subplot_min) and np.isfinite(current_subplot_max):
            padding_ratio = 0.05
            yrange = current_subplot_max - current_subplot_min
            if yrange <= 1e-9:
                yrange = max(abs(current_subplot_max), abs(current_subplot_min), 0.1) * 0.2
                if yrange <= 1e-9: yrange = 0.1
            padding = yrange * padding_ratio
            ymin_final = current_subplot_min - padding
            ymax_final = current_subplot_max + padding
            ymin_final = max(ymin_final, 0)
            if ymax_final <= ymin_final: ymax_final = ymin_final + yrange * 0.1
            ax.set_ylim(bottom=ymin_final, top=ymax_final)
        else:
            ax.set_ylim(bottom=0, top=1)

        ax.set_xlim(0, n_steps)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(18)
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
        ax.set_xlabel('Steps', fontsize=26)

    # --- Figure Final Adjustments and Saving ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if figure_has_data:
        alpha_strs = "_".join([str(a).replace('.','p') for a in alphas_to_test])
        output_filename = f'PLOT_td_lambda_conv_combined_gammas_alphas_{alpha_strs}_n{n_runs}_1x3.pdf'
        try:
            plt.savefig(output_filename, dpi=150)
            print(f"\nFigure saved as {output_filename}")
        except Exception as e:
            print(f"Error saving figure {output_filename}: {e}")
    else:
        print("Skipping save as no data was successfully plotted.")

    plt.close(fig)

    overall_end_time = time.time()
    print(f"\nTotal plotting script execution time: {(overall_end_time - overall_start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()