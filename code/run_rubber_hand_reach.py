import os

import numpy as np
import pandas as pd
import argparse
from typing import Optional

# import from other scripts
from mlp_utils import merge_training_data, merge_test_data
from monitoring import PopMonitor, ConMonitor
from train_rubber_hand_reach import *


def create_one_hot_encoded_column(df: pd.DataFrame,
                                  column_name: str,
                                  new_column_name: str,
                                  inplace: bool = True,
                                  scaling: float = 0.5, ) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    # Extract unique angles and sort them
    unique_angles = sorted(df[column_name].unique())
    n_angles = len(unique_angles)

    # Create angle to index mapping
    angle_to_idx = {angle: idx for idx, angle in enumerate(unique_angles)}

    # Function to create one-hot array for a single angle
    def create_onehot(angle):
        onehot = np.zeros(n_angles)
        onehot[angle_to_idx[angle]] = scaling
        return onehot

    # Apply one-hot encoding to the column
    df[new_column_name] = df[column_name].apply(create_onehot)

    return df


def normalize_list_column(df, column_name, new_column_name=None):
    # Find global maximum across all lists in the column
    global_max = max(max(lst) for lst in df[column_name])

    if global_max == 0:
        raise ValueError("Global maximum is 0, cannot normalize")

    # Function to normalize a single list
    def normalize_list(lst):
        return [val / global_max for val in lst]

    # Set output column name
    output_column = new_column_name if new_column_name else column_name

    # Apply normalization to the column
    df[output_column] = df[column_name].apply(normalize_list)

    return df


def training(df_train: pd.DataFrame,
             s1_column: str = 'r_output',
             m1_column: str = 'theta',
             m1_scaling: float = 0.5,
             wait_time: float = 50.,
             reach_time: float = 150.,
             save_path: Optional[str] = None,
             pop_monitors: Optional[PopMonitor] = None,
             con_monitors: Optional[ConMonitor] = None,
             sub_samples: Optional[int] = None,
             normalize_s1_inputs: bool = True,
             save_model: bool = True,
             shuffle: bool = True, ):
    # create one hot encoded column for movement input
    df_train = create_one_hot_encoded_column(df_train,
                                             column_name=m1_column,
                                             new_column_name='m1_input',
                                             inplace=True,
                                             scaling=m1_scaling, )
    # shuffle data
    if shuffle:
        df_train = df_train.sample(frac=1).reset_index(drop=True)

    if sub_samples is not None:
        df_train = df_train.sample(sub_samples)

    if normalize_s1_inputs:
        df_train = normalize_list_column(df_train, column_name=s1_column)

    # train the model
    if con_monitors is not None:
        con_monitors.extract_weights()

    if pop_monitors is not None:
        pop_monitors.start()

    train_over_inputs(s1_inputs=df_train[s1_column].tolist(),
                      m1_inputs=df_train['m1_input'].tolist(),
                      wait_time=wait_time,
                      reach_time=reach_time)

    if pop_monitors is not None:
        pop_monitors.stop()

    if con_monitors is not None:
        con_monitors.extract_weights()

    # debug animation of current monitors
    if sub_samples is not None:
        pop_monitors.animate_current_monitors(
            clear_monitors=False,
            plot_types=['Bar', 'Bar', 'Bar', 'Bar', 'Bar', 'Line', 'Matrix'],
            plot_order=(4, 2)
        )

    if save_path is not None:
        if save_path[-1] != '/':
            save_path += '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if pop_monitors is not None:
            pop_monitors.save(folder=save_path, delete=True)
        if con_monitors is not None:
            con_monitors.save_cons(folder=save_path)
            con_monitors.current_weight_diff(fig_size=(10, 10), save_name=save_path + 'weight_diff.pdf')
            con_monitors.reset()

        if save_model:
            ann.save(save_path + 'bg_synapses.npz')

    return df_train


def testing(df_test: pd.DataFrame,
            temperature_softmax: float,
            s1_column: str = 'r_output',
            wait_time: float = 50.,
            reach_time: float = 150.,
            save_path: Optional[str] = None,
            pop_monitors: Optional[PopMonitor] = None,
            sub_samples: Optional[int] = None,
            normalize_s1_inputs: bool = True,
            load_model_path: Optional[str] = None,
            shuffle: bool = True,
            append_sparse_goal: bool = False, ):

    # shuffle data
    if shuffle:
        df_test = df_test.sample(frac=1).reset_index(drop=True)

    if load_model_path is not None:
        ann.load(load_model_path + '/bg_synapses.npz')

    if sub_samples is not None:
        df_test = df_test.sample(sub_samples)

    if normalize_s1_inputs:
        df_test = normalize_list_column(df_test, column_name=s1_column)

    # test the model
    if pop_monitors is not None:
        pop_monitors.start()

    m1_rates, cpgs_output, angles_output = predict_over_inputs(s1_inputs=df_test[s1_column].tolist(),
                                                               reach_time=reach_time,
                                                               wait_time=wait_time,
                                                               temperature=temperature_softmax)

    if pop_monitors is not None:
        pop_monitors.stop()

    # update the dataframe
    df_test['bg_theta_output'] = angles_output
    df_test['bg_cpg_output'] = cpgs_output
    df_test['bg_m1_output'] = m1_rates

    if append_sparse_goal:
        df_test = create_one_hot_encoded_column(df_test,
                                                column_name='theta',
                                                new_column_name='sparse_goal',
                                                inplace=True,
                                                scaling=0.5)

    if sub_samples is not None and pop_monitors is not None:
        pop_monitors.animate_current_monitors(
            clear_monitors=False,
        )

    if save_path is not None:
        if save_path[-1] != '/':
            save_path += '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        df_test.to_parquet(save_path + 'test_results.parquet')
        if pop_monitors is not None:
            pop_monitors.save(folder=save_path, delete=True)

    return df_test


def plot_theta_errors(df_test: pd.DataFrame,
                      save_path: Optional[str] = None,
                      scatter_subset: Optional[int] = None, ):
    import matplotlib.pyplot as plt

    results_df = pd.DataFrame({
        'theta': df_test['theta'],
        'theta_pred': df_test['bg_theta_output'],
        'theta_diff_true': df_test['theta'] - df_test['vision_theta'],
        'theta_diff_pred': df_test['bg_theta_output'] - df_test['vision_theta']
    })

    error_stats = results_df.groupby('theta_diff_true')['theta_diff_pred'].agg(['mean', 'std']).reset_index()
    error_stats.columns = ['diff', 'mean', 'std']
    error_stats['se'] = error_stats['std'] / np.sqrt(
        len(results_df.groupby('theta_diff_true')['theta_diff_pred'].count()))

    # Calculate the upper and lower bounds for the error range
    error_stats['lower_bound'] = error_stats['mean'] - error_stats['std']
    error_stats['upper_bound'] = error_stats['mean'] + error_stats['std']

    # Plot the mean error line for this dataset
    fig = plt.figure(figsize=(10, 6))
    plt.plot(error_stats['diff'], error_stats['mean'], 'b', label='Mean Error in Theta')

    # Plot the shaded error range for this dataset
    plt.fill_between(error_stats['diff'], error_stats['lower_bound'], error_stats['upper_bound'], alpha=0.1)

    if scatter_subset is not None:
        results_df = results_df.sample(n=scatter_subset)
    plt.scatter(results_df['theta_diff_true'] + np.random.uniform(low=-0.5, high=0.5, size=len(results_df)),
                results_df['theta_diff_pred'], s=0.04,
                alpha=0.2, c='gray')

    plt.legend()
    plt.grid()
    plt.ylabel('$\\theta^{proprio}_{true} - \\theta_{pred}$')
    plt.xlabel('$\\theta^{proprio}_{true} - \\theta^{vision}_{true}$')
    if save_path is not None:
        if save_path[-1] != '/':
            save_path += '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + 'theta_error.png')

    plt.close(fig)


def plot_training_error(df_train: pd.DataFrame,
                        rhi_theta_column: str = 'theta',
                        bg_theta_column: str = 'bg_theta_output',
                        bin_size: float = 0.2,
                        save_path: Optional[str] = None):
    """
    Create a bar plot showing the frequency of differences between two theta columns.
    """

    import matplotlib.pyplot as plt

    # Calculate differences between the two columns
    theta_diff = df_train[rhi_theta_column] - df_train[bg_theta_column]

    # Create histogram
    fig = plt.figure(figsize=(10, 6))
    plt.hist(theta_diff, bins=np.arange(min(theta_diff), max(theta_diff) + bin_size, bin_size),
             edgecolor='black', alpha=0.7)

    plt.xlabel(f'Difference ({rhi_theta_column} - {bg_theta_column})')
    plt.ylabel('Frequency')
    plt.title('Distribution of Theta Differences in Training')
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        if save_path[-1] != '/':
            save_path += '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + 'training_error_dist.pdf')
        plt.close(fig)
    else:
        plt.show()


def plot_average_m1_firing_rates(df_train: pd.DataFrame,
                                 rhi_theta_column: str = 'theta',
                                 m1_column: str = 'bg_m1_output',
                                 save_path: Optional[str] = None):
    """
    Create subplots showing average M1 firing rates for each unique theta value.
    Each M1 firing rate is a list of length N=33.
    """
    import matplotlib.pyplot as plt

    # Get unique theta values and sort them
    unique_thetas = sorted(df_train[rhi_theta_column].unique())
    n_thetas = len(unique_thetas)

    # Calculate number of rows and columns for subplots
    n_cols = min(5, n_thetas)  # Maximum 5 columns
    n_rows = (n_thetas + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    fig.suptitle('Average M1 Firing Rates by Theta Value')

    # Flatten axes array for easier iteration
    if n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_cols > 1:
        axes = axes.flatten()
    elif n_rows > 1 and n_cols == 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot average firing rate for each theta
    for idx, theta in enumerate(unique_thetas):
        # Get all firing rate lists for this theta
        firing_rates = df_train[df_train[rhi_theta_column] == theta][m1_column].tolist()

        # Convert list of lists to 2D numpy array
        firing_rates = np.array(firing_rates)

        # Calculate mean and standard deviation
        mean_rates = np.mean(firing_rates, axis=0)
        std_rates = np.std(firing_rates, axis=0)

        # Create x-axis values (timesteps)
        timesteps = np.arange(len(mean_rates))

        # Plot mean and standard deviation
        axes[idx].plot(timesteps, mean_rates, 'b-', label='Mean')
        axes[idx].fill_between(timesteps,
                               mean_rates - std_rates,
                               mean_rates + std_rates,
                               alpha=0.2,
                               color='b')

        axes[idx].set_title(f'θ = {theta:.1f}')
        axes[idx].grid(True, alpha=0.3)

        if idx % n_cols == 0:  # Add y-label to leftmost plots
            axes[idx].set_ylabel('Firing Rate')

        if idx >= n_thetas - n_cols:  # Add x-label to bottom plots
            axes[idx].set_xlabel('Timestep')

    # Remove empty subplots if any
    if n_thetas < len(axes):
        for idx in range(n_thetas, len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()

    if save_path is not None:
        if save_path[-1] != '/':
            save_path += '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + 'm1_firing_rates.pdf')
        plt.close(fig)
    else:
        plt.show()


if __name__ == '__main__':
    rhi_parser = argparse.ArgumentParser()
    rhi_parser.add_argument('--data_set', type=str, default="RHI_j11_sigma2",
                            choices=("RHI_j11_sigma2", "RHI_j12_sigma4"), help="Abreviation of rhi data set")
    rhi_parser.add_argument('--rhi_data_path', type=str, default="data_out/data_RHI_jitter_1_1_sigma_prop_2.npz",
                            help="Path to Valentin's raw rhi data")
    rhi_parser.add_argument('--monitoring_training', type=bool, default=True,
                            help="Monitor the training process?")
    rhi_parser.add_argument('--monitoring_testing', type=bool, default=False,
                            help="Monitor the testing process?")
    rhi_parser.add_argument('--clean_compile', type=bool, default=True, )
    rhi_parser.add_argument('--debug', type=bool, default=False, )
    rhi_parser.add_argument('--temperature', type=float, default=parameters['temperature_softmax'], )
    rhi_parser.add_argument('--do_plots', type=bool, default=True, )
    rhi_parser.add_argument('--init_m1_scale', type=float, default=1.0, )
    rhi_args = rhi_parser.parse_args()

    # data paths
    cpg_path = f'results/{rhi_args.data_set}/network_inverse_kinematic/best_inverse_results.npz'
    rhi_raw_path = rhi_args.rhi_data_path
    df_train_name = f'{rhi_args.data_set}_training.parquet'
    df_test_name = f'{rhi_args.data_set}_test.parquet'
    # save paths
    training_save_path = f'results/{rhi_args.data_set}/bg_reach_training/'
    testing_save_path = f'results/{rhi_args.data_set}/bg_reach_testing/'

    # debug conditions
    if rhi_args.debug:
        training_save_path = f'results/{rhi_args.data_set}/bg_reach_training_debug/'
        testing_save_path = f'results/{rhi_args.data_set}/bg_reach_testing_debug/'
        # reduce number of samples in the dataframes
        n_samples = 200
        # monitoring sampling rates can be lower
        sampling_rate_training = 2.
        sampling_rate_testing = 2.
    else:
        n_samples = None
        sampling_rate_training = 100.
        sampling_rate_testing = 100.

    # compile model
    compile_folder = f'annarchy/{rhi_args.data_set}/'
    if not os.path.exists(compile_folder):
        os.makedirs(compile_folder)
    ann.compile(directory=compile_folder, clean=rhi_args.clean_compile)

    # training
    df_train = merge_training_data(rhi_path=rhi_raw_path, cpg_path=cpg_path, save_name=df_train_name)

    if rhi_args.monitoring_training:
        pop_monitors_training = PopMonitor([S1, StrD1, SNr, VL, M1, SNc, ],
                                           variables=['r', 'r', 'r', 'r', 'r', 'r', ],
                                           sampling_rate=sampling_rate_training)
        con_monitors_training = ConMonitor([S1_StrD1])
    else:
        pop_monitors_training = None
        con_monitors_training = None

    print('Beginning training...')
    training(df_train=df_train,
             pop_monitors=pop_monitors_training,
             con_monitors=con_monitors_training,
             m1_scaling=rhi_args.init_m1_scale,
             save_path=training_save_path,
             sub_samples=n_samples)

    # test training performance with congruent s1 representations
    df_train = testing(df_test=df_train,
                       reach_time=200.,
                       temperature_softmax=rhi_args.temperature,
                       save_path=training_save_path,
                       sub_samples=n_samples,
                       append_sparse_goal=False)

    if rhi_args.do_plots:
        plot_training_error(df_train=df_train, save_path=training_save_path)
        plot_average_m1_firing_rates(df_train=df_train, save_path=training_save_path)

    # test on incongruent s1 representations
    df_test = merge_test_data(rhi_path=rhi_raw_path, cpg_path=cpg_path, save_name=df_test_name)

    if rhi_args.monitoring_testing:
        pop_monitors_testing = PopMonitor([S1, StrD1, SNr, VL, M1, Brainstem, CPG_output],
                                          sampling_rate=sampling_rate_testing)
    else:
        pop_monitors_testing = None

    print('Beginning testing...')
    df_test = testing(df_test=df_test,
                      reach_time=200.,
                      pop_monitors=pop_monitors_testing,
                      save_path=testing_save_path,
                      sub_samples=n_samples,
                      temperature_softmax=rhi_args.temperature)

    if rhi_args.debug:
        print(df_test.columns)
        print(np.amax(np.array(S1_StrD1.w), axis=1))
        print(S1_StrD1.alpha)

    if rhi_args.do_plots and not rhi_args.debug:
        plot_theta_errors(df_test=df_test, save_path=testing_save_path, scatter_subset=10_000)
    elif rhi_args.do_plots and rhi_args.debug:
        plot_theta_errors(df_test=df_test, save_path=testing_save_path, scatter_subset=n_samples)
        plot_average_m1_firing_rates(df_train=df_test, save_path=testing_save_path)
