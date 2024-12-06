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
             m1_scaling: float = 0.5,
             wait_time: float = 50.,
             reach_time: float = 150.,
             save_path: Optional[str] = None,
             pop_monitors: Optional[PopMonitor] = None,
             con_monitors: Optional[ConMonitor] = None,
             sub_samples: Optional[int] = None,
             normalize_s1_inputs: bool = True,):
    # create one hot encoded column
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
            plot_types=['Bar', 'Bar', 'Bar', 'Bar', 'Bar', 'Line'],
        )

    if save_path is not None:
        if save_path[-1] != '/':
            save_path += '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pop_monitors.save(folder=save_path, delete=True)
        if con_monitors is not None:
            con_monitors.save_cons(folder=save_path)
            con_monitors.current_weight_diff(fig_size=(10, 10), save_name=save_path + 'weight_diff.pdf')
            con_monitors.reset()


def testing(df_test: pd.DataFrame,
            s1_column: str = 'r_output',
            wait_time: float = 50.,
            reach_time: float = 150.,
            temperature_softmax: float = 0.5,
            save_path: Optional[str] = None,
            pop_monitors: Optional[PopMonitor] = None,
            sub_samples: Optional[int] = None,
            normalize_s1_inputs: bool = True, ):
    if sub_samples is not None:
        df_test = df_test.sample(sub_samples)

    if normalize_s1_inputs:
        df_test = normalize_list_column(df_test, column_name=s1_column)

    # test the model
    if pop_monitors is not None:
        pop_monitors.start()

    cpgs_output, angles_output = predict_over_inputs(s1_inputs=df_test[s1_column].tolist(),
                                                     reach_time=reach_time,
                                                     wait_time=wait_time,
                                                     temperature=temperature_softmax)

    if pop_monitors is not None:
        pop_monitors.stop()

    # update the dataframe
    df_test['theta_output'] = angles_output
    df_test['cpg_output'] = cpgs_output

    if save_path is not None:
        if save_path[-1] != '/':
            save_path += '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        df_test.to_parquet(save_path + 'test_results.parquet')
        pop_monitors.save(folder=save_path, delete=True)

    return df_test


if __name__ == '__main__':
    rhi_parser = argparse.ArgumentParser()
    rhi_parser.add_argument('--data_set', type=str, default="RHI_j11_sigma2",
                            choices=("RHI_j11_sigma2", "RHI_j12_sigma4"), help="Abreviation of rhi data set")
    rhi_parser.add_argument('--rhi_data_path', type=str, default="data_out/data_RHI_jitter_1_1_sigma_prop_2.npz",
                            help="Path to Valentins raw rhi data")
    rhi_parser.add_argument('--monitoring_training', type=bool, default=True,
                            help="Monitor the training process")
    rhi_parser.add_argument('--monitoring_testing', type=bool, default=True,
                            help="Monitor the testing process")
    rhi_parser.add_argument('--clean_compile', type=bool, default=False, )
    rhi_parser.add_argument('--debug', type=bool, default=True, )
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
        n_samples = 50
    else:
        n_samples = None

    # compile model
    ann.compile(directory=f'annarchy/{rhi_args.data_set}/', clean=rhi_args.clean_compile)

    # training
    df_train = merge_training_data(rhi_path=rhi_raw_path, cpg_path=cpg_path, save_name=df_train_name)

    if rhi_args.monitoring_training:
        pop_monitors_training = PopMonitor([S1, StrD1, SNr, VL, M1, SNc], sampling_rate=5.0)
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
        pop_monitors_testing = PopMonitor([S1, StrD1, SNr, VL, M1, Brainstem, CPG_output], sampling_rate=10.0)
    else:
        pop_monitors_testing = None

    print('Beginning testing...')
    df_test = testing(df_test=df_test,
                      pop_monitors=pop_monitors_testing,
                      save_path=testing_save_path,
                      sub_samples=n_samples)

