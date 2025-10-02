import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import pickle
import time

# --- Parameters ---
n_steps = 1500000
n_runs = 10
T0 = 10_000_000
beta_0 = 0.01 * T0
alphas = [0.01, 0.02, 0.1]  # Changed from c_deltas
lambdas = [0.1, 0.5, 0.9]
data_dir = "avg_reward_td_lambda_data"

# --- Define colors and marker styles ---
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'd']

def get_experiment_data_filename(alpha, beta_0, lambda_):
    """Generate unique filename for experiment configuration (use c_delta and steps2000k for existing files)"""
    # Map alpha to corresponding c_delta for filename compatibility
    alpha_to_c_delta = {0.01: 0.1, 0.02: 0.5, 0.1: 1.0}
    c_delta = alpha_to_c_delta.get(alpha, alpha)  # Fallback to alpha if not mapped
    return f"avg_reward_td_lambda_c{str(c_delta).replace('.','p')}_b{str(beta_0).replace('.','p')}_l{str(lambda_).replace('.','p')}_n{n_runs}_steps2000k.pkl"

def load_experiment_data(alpha, beta_0, lambda_):
    """Load experiment data from pickle file and truncate to n_steps"""
    filename = os.path.join(data_dir, get_experiment_data_filename(alpha, beta_0, lambda_))
    
    if not os.path.exists(filename):
        print(f"Data file for alpha={alpha}, beta_0={beta_0}, lambda_={lambda_} not found. File does not exist: {filename}")
        return None, None, None
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded data from {filename}")
        # print(f"  Raw data: c_delta={data['c_delta']}, beta_0={data['beta_0']}, lambda_={data['lambda_']}, steps={data['n_steps']}, runs={data['n_runs']}")
        
        # Truncate data to n_steps (1500000)
        if len(data['dist_mean']) < n_steps or data['dist_runs'].shape[1] < n_steps:
            print(f"Error: Data in {filename} has fewer than {n_steps} steps.")
            return None, None, None
        
        dist_mean = data['dist_mean'][:n_steps]
        dist_std = data['dist_std'][:n_steps]
        dist_runs = data['dist_runs'][:, :n_steps]
        
        return dist_mean, dist_std, dist_runs
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def load_comprehensive_data():
    """Attempt to load comprehensive data file (use steps2000k for existing files)"""
    comprehensive_filename = os.path.join(data_dir, f'all_avg_reward_td_lambda_data_2000k_beta{str(beta_0).replace(".","p")}_n{n_runs}.pkl')
    
    if not os.path.exists(comprehensive_filename):
        print(f"Comprehensive data file not found: {comprehensive_filename}")
        return None
    
    try:
        with open(comprehensive_filename, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded comprehensive data from {comprehensive_filename}")
        # Truncate experiment data to n_steps
        for key in data['experiment_data']:
            data['experiment_data'][key]['dist_mean'] = data['experiment_data'][key]['dist_mean'][:n_steps]
            data['experiment_data'][key]['dist_std'] = data['experiment_data'][key]['dist_std'][:n_steps]
            data['experiment_data'][key]['dist_runs'] = data['experiment_data'][key]['dist_runs'][:, :n_steps]
        
        # Remap keys to use alpha instead of c_delta
        new_experiment_data = {}
        c_delta_to_alpha = {0.1: 0.01, 0.5: 0.02, 1.0: 0.1}
        for (c_delta, beta, lam), value in data['experiment_data'].items():
            alpha = c_delta_to_alpha.get(c_delta, c_delta)
            new_experiment_data[(alpha, beta, lam)] = value
        data['experiment_data'] = new_experiment_data
        
        return data
    except Exception as e:
        print(f"Error loading comprehensive data: {e}")
        return None

def create_new_visualization():
    """Create new visualization: one plot per lambda, different colors for different alpha values"""
    print("Creating new visualization, one plot per lambda with different alpha values in different colors...")
    
    # Attempt to load comprehensive data
    all_data = load_comprehensive_data()
    
    # If comprehensive data not found, try loading individual experiment data
    if all_data is None:
        print("Comprehensive data not found, attempting to load individual experiment data...")
        all_data = {'experiment_data': {}}
        
        for lambda_ in lambdas:
            for alpha in alphas:
                dist_mean, dist_std, dist_runs = load_experiment_data(alpha, beta_0, lambda_)
                if dist_mean is not None:
                    all_data['experiment_data'][(alpha, beta_0, lambda_)] = {
                        'dist_mean': dist_mean,
                        'dist_std': dist_std,
                        'dist_runs': dist_runs
                    }
    
    if not all_data['experiment_data']:
        print("No experiment data found. Please ensure data files exist or check the path.")
        return
    
    fig, axes = plt.subplots(1, len(lambdas), figsize=(6 * len(lambdas), 5), sharex=True)
    if len(lambdas) == 1:
        axes = [axes]
    
    steps = np.arange(n_steps)
    
    # <<< START: MODIFICATION >>>
    reporting_interval = 150000
    # <<< END: MODIFICATION >>>

    # Plot subplots for each lambda
    for j, lambda_ in enumerate(lambdas):
        ax = axes[j]
        
        # Plot a line for each alpha
        for i, alpha in enumerate(alphas):
            key = (alpha, beta_0, lambda_)
            if key in all_data['experiment_data']:
                data = all_data['experiment_data'][key]
                dist_mean = data['dist_mean']
                dist_std = data['dist_std']
                
                if np.all(np.isnan(dist_mean)):
                    print(f"Skipping alpha={alpha}, lambda={lambda_}, as the run completely failed.")
                    continue

                # <<< START: MODIFICATION FOR PRINTING VALUES >>>
                if alpha == 0.1:
                    print(f"\n    --- Values for λ={lambda_}, β={beta_0}, α={alpha} (11 points) ---")
                    
                    # Print the initial value at Step 0
                    initial_value = dist_mean[0]
                    print(f"      Step {0:<9,}: Mean Distance = {initial_value:.6f}")

                    # Loop for subsequent milestones, including the last one
                    for step_milestone in range(reporting_interval, n_steps + 1, reporting_interval):
                        index_to_report = step_milestone - 1
                        if index_to_report < len(dist_mean):
                            value_at_step = dist_mean[index_to_report]
                            print(f"      Step {step_milestone:<9,}: Mean Distance = {value_at_step:.6f}")
                    print("    -----------------------------------------------------------\n")
                # <<< END: MODIFICATION FOR PRINTING VALUES >>>

                # Plot mean line
                color = colors[i % len(colors)]
                line, = ax.plot(steps, dist_mean, 
                               linewidth=2.5, color=color, 
                               label=rf'$\alpha$={alpha}')
            else:
                print(f"Data for alpha={alpha}, lambda={lambda_} not found")
        
        # Set chart title and labels
        ax.set_title(rf'$\lambda$={lambda_}, $\beta$={beta_0/T0}', fontsize=24)
        if j == 0:
            ax.set_ylabel(r'$d(w_t, W^*)$', fontsize=24)
        ax.set_xlabel('Steps', fontsize=24)
        
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(16)
        ax.set_xlim(0, n_steps)
        
        # Auto-adjust Y-axis
        padding_ratio = 0.05
        valid_data = [all_data['experiment_data'].get((a, beta_0, lambda_), {}).get('dist_mean', np.array([np.nan])) 
                      for a in alphas]
        valid_data = [d for d in valid_data if not np.all(np.isnan(d))]
        
        if valid_data:
            min_values = [np.nanmin(d) for d in valid_data]
            min_mean_y = min(min_values) if min_values else 0.0
            
            initial_steps_limit = max(1, n_steps // 10)
            max_values = []
            for d in valid_data:
                if len(d) >= initial_steps_limit:
                    valid_initial = d[:initial_steps_limit]
                    valid_initial = valid_initial[~np.isnan(valid_initial)]
                    if len(valid_initial) > 0:
                        max_values.append(np.nanmax(valid_initial))
            
            ymax_data = max(max_values) if max_values else min_mean_y + 1.0
            
            if np.isnan(ymax_data): ymax_data = min_mean_y + 1.0
            if np.isnan(min_mean_y): min_mean_y = 0.0
            
            yrange = ymax_data - min_mean_y
            if yrange <= 1e-9: yrange = max(abs(ymax_data), abs(min_mean_y), 1.0) * 0.2
            
            padding = yrange * padding_ratio
            ymin_final = min_mean_y - padding
            ymax_final = ymax_data + padding
            
            ymin_final = max(ymin_final, 0)
            if ymax_final <= ymin_final: ymax_final = ymin_final + max(0.1, yrange * 0.2)
            
            ax.set_ylim(bottom=ymin_final, top=ymax_final)
        
        ax.legend(fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    output_filename = f'new_avg_reward_td_lambda_conv_by_lambda_1p5Msteps_beta{str(beta_0).replace(".","p")}_n{n_runs}_alpha.pdf'
    
    plt.savefig(output_filename, dpi=150)
    plt.show()
    print(f"Chart saved as {output_filename}")

if __name__ == "__main__":
    create_new_visualization()