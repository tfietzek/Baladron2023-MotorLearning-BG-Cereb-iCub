import ANNarchy as ann
import optuna
from optuna.samplers import TPESampler
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Tuple
import pandas as pd
import multiprocessing

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
        'w_lat_strD1': (0.01, 1.0),
        'w_lat_snr': (0.01, 2.0),
        'w_lat_m1': (0.01, 2.0),
        'w_lat_vl': (0.01, 2.0),
        'w_fb_m1': (0.2, 1.5),
        'pre_threshold': (0.0, 0.5),
        'post_threshold': (0.0, 0.5),
        'rpe_threshold': (0.0, 0.5)
    }


def objective(trial: optuna.Trial, df_train: pd.DataFrame) -> float:
    """Objective function for Optuna optimization."""
    # Create unique compile and save paths based on worker process
    process_id = multiprocessing.current_process().name
    compile_path = f'annarchy/optuna_trials/{process_id}/'
    save_path = f'results/optuna_trials/trial_{trial.number}/'

    if not os.path.exists(compile_path):
        os.makedirs(compile_path, exist_ok=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Compile Model
    ann.compile(directory=compile_path, clean=True)

    # Get parameter bounds
    bounds = define_parameter_bounds()

    # Sample parameters within bounds
    params = {
        'learn_tau': trial.suggest_float('learn_tau', *bounds['learn_tau']),
        'w_lat_strD1': trial.suggest_float('w_lat_strD1', *bounds['w_lat_strD1']),
        'w_lat_snr': trial.suggest_float('w_lat_snr', *bounds['w_lat_snr']),
        'w_lat_m1': trial.suggest_float('w_lat_m1', *bounds['w_lat_m1']),
        'w_lat_vl': trial.suggest_float('w_lat_vl', *bounds['w_lat_vl']),
        'w_fb_m1': trial.suggest_float('w_fb_m1', *bounds['w_fb_m1']),
        'pre_threshold': trial.suggest_float('pre_threshold', *bounds['pre_threshold']),
        'post_threshold': trial.suggest_float('post_threshold', *bounds['post_threshold']),
        'rpe_threshold': trial.suggest_float('rpe_threshold', *bounds['rpe_threshold'])
    }

    try:
        # Update model parameters
        update_model_params(**params)

        # Train the model
        training(
            df_train=df_train.copy(),
            save_path=save_path,
            save_model=True,
            pop_monitors=None,
            con_monitors=None
        )

        # Test the model
        df_train = testing(
            df_test=df_train,
            save_path=save_path,
            pop_monitors=None,
            temperature_softmax=0.1
        )

        # Calculate error metric (mean squared error between predicted and true theta)
        mse = np.mean((df_train['theta'] - df_train['bg_theta_output']) ** 2)

        return mse

    except Exception as e:
        print(f"Error in trial {trial.number}: {str(e)}")
        return float('inf')  # Return infinity for failed trials


def run_optimization(df_train: pd.DataFrame,
                     n_trials: int = 100,
                     n_jobs: int = 4,
                     storage: str = "sqlite:///optuna_results.db",
                     study_name: str = "hyperparameter_optimization") -> optuna.Study:
    """Run the hyperparameter optimization with parallel processing."""

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
    objective_partial = partial(objective, df_train=df_train)

    # Run optimization
    study.optimize(
        objective_partial,
        n_trials=n_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
        show_progress_bar=True
    )

    print("\nOptimization completed!")
    print(f"Best trial MSE: {study.best_value}")
    print("Best parameters:", study.best_params)

    return study


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default="RHI_j11_sigma2",
                        choices=("RHI_j11_sigma2", "RHI_j12_sigma4"))
    parser.add_argument('--rhi_data_path', type=str,
                        default="data_out/data_RHI_jitter_1_1_sigma_prop_2.npz")
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--n_jobs', type=int, default=4)
    parser.add_argument('--storage', type=str, default="sqlite:///optuna_results.db")
    args = parser.parse_args()

    # Set up paths
    cpg_path = f'results/{args.data_set}/network_inverse_kinematic/best_inverse_results.npz'

    # Load training data
    df_train = merge_training_data(
        rhi_path=args.rhi_data_path,
        cpg_path=cpg_path,
        save_name=f'{args.data_set}_training.parquet'
    )

    # Run optimization
    study = run_optimization(
        df_train=df_train,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=f"hyperparameter_optimization_{args.data_set}"
    )
