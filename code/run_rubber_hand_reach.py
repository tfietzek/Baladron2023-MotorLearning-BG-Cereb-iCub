import os
import numpy as np
import ANNarchy as ann
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
             m1_scaling: float = 0.4,
             wait_time: float = 50.,
             reach_time: float = 150.,
             save_path: Optional[str] = None,
             pop_monitors: Optional[PopMonitor] = None,
             con_monitors: Optional[ConMonitor] = None,
             sub_samples: Optional[int] = None,
             normalize_s1_inputs: bool = True,
             save_model: bool = True, ):

    # create one hot encoded column for movement input
    df_train = create_one_hot_encoded_column(df_train,
                                             column_name=m1_column,
                                             new_column_name='m1_input',
                                             inplace=True,
                                             scaling=m1_scaling, )

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


def testing(df_test: pd.DataFrame,
            s1_column: str = 'r_output',
            wait_time: float = 50.,
            reach_time: float = 150.,
            temperature_softmax: float = 0.5,
            save_path: Optional[str] = None,
            pop_monitors: Optional[PopMonitor] = None,
            sub_samples: Optional[int] = None,
            normalize_s1_inputs: bool = True,
            load_model_path: Optional[str] = None, ):

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

    if sub_samples is not None:
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
                      scatter_subset: Optional[int] = None,):
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


def plot_cpg_errors(df_test: pd.DataFrame,
                    save_path: Optional[str] = None,):
    pass


if __name__ == '__main__':
    rhi_parser = argparse.ArgumentParser()
    rhi_parser.add_argument('--data_set', type=str, default="RHI_j11_sigma2",
                            choices=("RHI_j11_sigma2", "RHI_j12_sigma4"), help="Abreviation of rhi data set")
    rhi_parser.add_argument('--rhi_data_path', type=str, default="data_out/data_RHI_jitter_1_1_sigma_prop_2.npz",
                            help="Path to Valentin's raw rhi data")
    rhi_parser.add_argument('--monitoring_training', type=bool, default=False,
                            help="Monitor the training process?")
    rhi_parser.add_argument('--monitoring_testing', type=bool, default=False,
                            help="Monitor the testing process?")
    rhi_parser.add_argument('--clean_compile', type=bool, default=True, )
    rhi_parser.add_argument('--debug', type=bool, default=False, )
    rhi_parser.add_argument('--temperature', type=float, default=0.1, )
    rhi_parser.add_argument('--do_plots', type=bool, default=True, )
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
        n_samples = 20
        # monitoring sampling rates can be lower
        sampling_rate_training = 1.
        sampling_rate_testing = 1.
    else:
        n_samples = None
        sampling_rate_training = 50.
        sampling_rate_testing = 100.

    # compile model
    compile_folder = f'annarchy/{rhi_args.data_set}/'
    if not os.path.exists(compile_folder):
        os.makedirs(compile_folder)
    ann.compile(directory=compile_folder, clean=rhi_args.clean_compile)

    # training
    df_train = merge_training_data(rhi_path=rhi_raw_path, cpg_path=cpg_path, save_name=df_train_name)

    if rhi_args.monitoring_training:
        pop_monitors_training = PopMonitor([S1, StrD1, SNr, VL, M1, SNc, S1_StrD1],
                                           variables=['r', 'r', 'r', 'r', 'r', 'r', 'w'],
                                           sampling_rate=sampling_rate_training)
        con_monitors_training = ConMonitor([S1_StrD1])
    else:
        pop_monitors_training = None
        con_monitors_training = None

    print('Beginning training...')
    training(df_train=df_train,
             pop_monitors=pop_monitors_training,
             con_monitors=con_monitors_training,
             save_path=training_save_path,
             sub_samples=n_samples)

    # testing
    df_test = merge_test_data(rhi_path=rhi_raw_path, cpg_path=cpg_path, save_name=df_test_name)

    if rhi_args.monitoring_testing:
        pop_monitors_testing = PopMonitor([S1, StrD1, SNr, VL, M1, Brainstem, CPG_output],
                                          sampling_rate=sampling_rate_testing)
    else:
        pop_monitors_testing = None

    print('Beginning testing...')
    df_test = testing(df_test=df_test,
                      pop_monitors=pop_monitors_testing,
                      save_path=testing_save_path,
                      sub_samples=n_samples,
                      temperature_softmax=rhi_args.temperature)

    if rhi_args.debug:
        print(df_test.columns)

    if rhi_args.do_plots:
        plot_theta_errors(df_test=df_test, save_path=testing_save_path, scatter_subset=10_000)