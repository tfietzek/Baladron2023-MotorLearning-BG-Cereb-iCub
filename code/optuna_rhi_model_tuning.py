import ANNarchy as ann
import optuna
from optuna.samplers import TPESampler
import numpy as np
import os
from typing import Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from rhi_network.reaching_model import *
from run_rubber_hand_reach import training, testing
from mlp_utils import merge_training_data
from monitoring import PopMonitor, ConMonitor


def update_model_params(
        learn_tau: float,
        w_lat_strD1: float,
        w_lat_snr: float = 0.2,
        w_lat_m1: float = 0.3,
        w_lat_vl: float = 0.2,
        w_fb_m1: float = 0.5,
        pre_threshold: float = 0.1,
        post_threshold: float = 0.2,
        rpe_threshold: float = 0.1,
):
    # update model parameters
    S1_StrD1.tau = learn_tau

    # lateral weights
    StrD1_StrD1.w = w_lat_strD1
    SNr_SNr.w = w_lat_snr
    VL_VL.w = w_lat_vl
    M1_M1.w = w_lat_m1

    # feedback weights
    M1_StrD1.w = w_fb_m1

    # thresholds
    S1_StrD1.threshold_pre = pre_threshold
    S1_StrD1.threshold_post = post_threshold
    StrD1_SNc.threshold = rpe_threshold


def define_parameter_bounds() -> Dict[str, Tuple[float, float]]:
    """Define the bounds for each hyperparameter."""
    return {
        'learn_tau': (100.0, 5000.0),
        'w_lat_strD1': (0.05, 1.0),
        'w_lat_snr': (0.05, 1.0),
        'w_lat_m1': (0.05, 1.0),
        'w_lat_vl': (0.05, 1.0),
        'w_fb_m1': (0.1, 1.5),
        'pre_threshold': (0.0, 0.7),
        'post_threshold': (0.0, 0.7),
        'rpe_threshold': (0.0, 0.5),
        'temperature_softmax': (0.05, 1.5)
    }


def calculate_m1_activity_error(df_test: pd.DataFrame, min_activity_threshold: float = 0.3) -> float:
    """Calculate error based on maximum M1 activity levels per theta.
    Penalizes if the maximum average activity for a given theta is below threshold."""
    error = 0.0

    # Group by theta
    theta_groups = df_test.groupby('theta')['bg_m1_output'].agg(list)

    for theta, m1_outputs in theta_groups.items():
        # Convert list of lists to numpy array
        m1_activities = np.array(m1_outputs)

        # Calculate mean activity pattern across trials
        mean_activity_pattern = np.mean(m1_activities, axis=0)

        # Get maximum activity across neurons
        max_activity = np.max(mean_activity_pattern)

        # Penalize if maximum activity is below threshold
        if max_activity < min_activity_threshold:
            error += (min_activity_threshold - max_activity) ** 2

    return error


def calculate_sparseness_error(df_test: pd.DataFrame) -> float:
    """Calculate similarity between M1 activity and sparse goal using Jensen-Shannon divergence."""
    error = 0.0

    # Group by theta
    theta_groups = df_test.groupby('theta')

    for theta, group in theta_groups:
        # Get average M1 activity pattern for this theta
        m1_activities = np.array(group['bg_m1_output'].tolist())
        mean_m1_pattern = np.mean(m1_activities, axis=0)

        # Normalize to create probability distribution
        m1_dist = mean_m1_pattern / np.sum(mean_m1_pattern)

        # Get sparse goal for this theta
        sparse_goal = group['sparse_goal'].iloc[0]
        sparse_dist = sparse_goal / np.sum(sparse_goal)

        # Calculate Jensen-Shannon divergence
        # Handle zero distributions
        if np.sum(m1_dist) < 1e-3 or np.sum(sparse_dist) < 1e-3:
            js_divergence = 1.0  # Maximum dissimilarity
        else:
            m = 0.5 * (m1_dist + sparse_dist)
            js_divergence = 0.5 * (
                    np.sum(m1_dist * np.log(m1_dist / (m + 1e-10) + 1e-10)) +  # avoid log(0) and divide by 0
                    np.sum(sparse_dist * np.log(sparse_dist / (m + 1e-10) + 1e-10))
            )

        error += js_divergence

    # JS divergence is already bounded 0-1 per theta, so max return is 1.0
    return error / len(theta_groups)


def objective(trial: optuna.Trial, df: pd.DataFrame,
              data_set: str = "RHI_j11_sigma2",
              weight_theta_error: float = 0.01,  # Dividing by ~100 to bring theta errors to 0.01-0.25 range
              weight_activity_error: float = 1.0,  # Brings activity errors (up to 33*0.3² = 2.97) to ~1.-3. range
              weight_sparseness_error: float = 1.0,  # Typical range: 0.2-0.6 for common cases, up to 1.0 max
              min_activity_threshold: float = 0.3) -> float:
    """Modified objective function incorporating new error terms."""
    save_path = f'results/{data_set}/optuna_trials/trial_{trial.number}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    bounds = define_parameter_bounds()

    # Sample parameters including temperature
    params = {
        'learn_tau': trial.suggest_float('learn_tau', *bounds['learn_tau']),
        'w_lat_strD1': trial.suggest_float('w_lat_strD1', *bounds['w_lat_strD1']),
        'w_lat_snr': trial.suggest_float('w_lat_snr', *bounds['w_lat_snr']),
        'w_lat_m1': trial.suggest_float('w_lat_m1', *bounds['w_lat_m1']),
        'w_lat_vl': trial.suggest_float('w_lat_vl', *bounds['w_lat_vl']),
        'w_fb_m1': trial.suggest_float('w_fb_m1', *bounds['w_fb_m1']),
        'pre_threshold': trial.suggest_float('pre_threshold', *bounds['pre_threshold']),
        'post_threshold': trial.suggest_float('post_threshold', *bounds['post_threshold']),
        'rpe_threshold': trial.suggest_float('rpe_threshold', *bounds['rpe_threshold']),
        'temperature_softmax': trial.suggest_float('temperature_softmax', *bounds['temperature_softmax'])
    }

    try:
        # Reset weights and update parameters
        S1_StrD1.w = 0.0
        update_model_params(**{k: v for k, v in params.items() if k != 'temperature_softmax'})

        # Training
        training(
            df_train=df_train,
            save_path=save_path,
            save_model=True,
            pop_monitors=None,
            con_monitors=None,
            shuffle=False,
            m1_scaling=1.0,
        )

        # Testing with new temperature parameter
        df_test = testing(
            df_test=df_test,
            save_path=save_path,
            reach_time=300.0,
            pop_monitors=None,
            shuffle=False,
            temperature_softmax=params['temperature_softmax'],
            append_sparse_goal=True,
        )

        # Calculate different error components
        theta_error = weight_theta_error * np.mean((df_test['theta'] - df_test['bg_theta_output']) ** 2)
        activity_error = weight_activity_error * calculate_m1_activity_error(df_test, min_activity_threshold)
        sparseness_error = weight_sparseness_error * calculate_sparseness_error(df_test)

        total_error = theta_error + activity_error + sparseness_error

        # Save trial results with raw and weighted errors
        trial_results = {
            'trial_number': trial.number,
            'total_error': total_error,
            'theta_error_raw': theta_error / weight_theta_error,
            'theta_error_weighted': theta_error,
            'activity_error_raw': activity_error / weight_activity_error,
            'activity_error_weighted': activity_error,
            'sparseness_error_raw': sparseness_error / weight_sparseness_error,
            'sparseness_error_weighted': sparseness_error,
            **params
        }

        pd.DataFrame([trial_results]).to_csv(
            os.path.join(save_path, 'trial_results.csv'),
            index=False
        )

        return total_error

    except Exception as e:
        print(f"Error in trial {trial.number}: {str(e)}")
        return float('inf')


def run_optimization(df: pd.DataFrame,
                     n_trials: int = 100,
                     data_set: str = "RHI_j11_sigma2",
                     storage: str = "sqlite:///optuna_results.db",
                     study_name: str = "hyperparameter_optimization") -> optuna.Study:
    """Run the hyperparameter optimization sequentially."""

    # Create study using Optuna's built-in storage
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,  # Allow resuming existing study
        direction="minimize",
        sampler=TPESampler(seed=42)
    )

    # Create objective function with only required arguments
    from functools import partial
    objective_partial = partial(objective, df=df, data_set=data_set)

    # Run optimization sequentially
    study.optimize(
        objective_partial,
        n_trials=n_trials,
        n_jobs=1,  # Run sequentially
        gc_after_trial=True,
        show_progress_bar=True
    )

    print("\nOptimization completed!")
    print(f"Best trial MSE: {study.best_value}")
    print("Best parameters:", study.best_params)

    # Save best parameters
    best_params_df = pd.DataFrame([study.best_params])
    os.makedirs(f'results/{data_set}/optuna_trials/', exist_ok=True)
    best_params_df.to_csv(
        os.path.join(f'results/{data_set}/optuna_trials/', 'best_params.csv'),
        index=False
    )

    return study


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default="RHI_j11_sigma2",
                        choices=("RHI_j11_sigma2", "RHI_j12_sigma4"))
    parser.add_argument('--rhi_data_path', type=str,
                        default="data_out/data_RHI_jitter_1_1_sigma_prop_2.npz")
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--storage', type=str, default="sqlite:///optuna_results.db")
    args = parser.parse_args()

    # Set up paths
    cpg_path = f'results/{args.data_set}/network_inverse_kinematic/best_inverse_results.npz'

    # Compile ANNarchy once at the start
    compile_folder = f'annarchy/optuna_rhi_model_tuning/{args.data_set}/'
    if not os.path.exists(compile_folder):
        os.makedirs(compile_folder)
    ann.compile(directory=compile_folder, clean=True)

    # Load training data
    df = merge_training_data(
        rhi_path=args.rhi_data_path,
        cpg_path=cpg_path,
        save_name=f'{args.data_set}_training_optuna.parquet'
    )

    # Shuffle the dataframe with a fixed seed, we don't want to do it in the hyperparameter optimization
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Run optimization
    study = run_optimization(
        df=df,
        n_trials=args.n_trials,
        storage=args.storage,
        data_set=args.data_set,
        study_name=f"hyperparameter_optimization_{args.data_set}"
    )

    # If hyperopt is finished run whole experiment
    if study.best_trial.state == optuna.trial.TrialState.COMPLETE:
        import matplotlib.pyplot as plt
        from run_rubber_hand_reach import plot_training_error, plot_average_m1_firing_rates, plot_theta_errors
        from mlp_utils import merge_test_data

        # Get best parameters
        best_params = study.best_params

        # Reset weights and update parameters
        S1_StrD1.w = 0.0
        update_model_params(**{k: v for k, v in best_params.items() if k != 'temperature_softmax'})

        training_path = f'results/{args.data_set}/optuna_best_model_training/'
        testing_path = f'results/{args.data_set}/optuna_best_model_testing/'

        training(
            df_train=df,
            save_path=f'results/{args.data_set}/optuna_trials/trial_{study.best_trial.number}/',
            save_model=True,
            pop_monitors=None,
            con_monitors=None,
            shuffle=True,
            m1_scaling=1.0,
        )

        # Testing with new temperature parameter
        df_train = testing(
            df_test=df.copy(),
            save_path=training_path,
            reach_time=300.0,
            pop_monitors=None,
            shuffle=True,
            temperature_softmax=best_params['temperature_softmax'],
        )

        plot_training_error(df_train=df_train, save_path=training_path)
        plot_average_m1_firing_rates(df_train=df_train, save_path=training_path)

        # test on incongruent s1 representations
        df_test = merge_test_data(rhi_path=args.rhi_data_path, cpg_path=cpg_path, save_name=f'{args.data_set}_test_optuna.parquet')

        print('Beginning testing...')
        df_test = testing(df_test=df_test,
                          reach_time=300.,
                          pop_monitors=None,
                          save_path=testing_path,
                          sub_samples=None,
                          temperature_softmax=best_params['temperature_softmax'],)

        plot_theta_errors(df_test=df_test, save_path=testing_path, scatter_subset=100_000)

