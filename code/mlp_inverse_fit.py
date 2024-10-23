import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
from typing import Optional
import os
import gc

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error

import CPG_lib.parameter as params
from kinematic import *
from train_BG_reaching import execute_movement
from plot_error_inverse_kinematics import choose_run_with_lowest_error, load_inverse_kinematic_results


def make_reaching_error_data(
        cpg_path: Optional[str] = 'results/RHI_j11_sigma2/network_inverse_kinematic/inverse_results_run5',
        goal: np.ndarray = np.array((-0.25, 0.1, 0.15))
):

    # init data
    if cpg_path is None:
        cpg_data = choose_run_with_lowest_error()
    else:
        cpg_data = load_inverse_kinematic_results(cpg_path)

    changed_angle = cpg_data['changed_angle']
    cpgs = cpg_data['cpg_params_to_goals']

    # make column for possible reaching errors
    reaching_errors = []

    # init icub + cpg
    iCubMotor = importlib.import_module(params.iCub_joint_names)

    joint1 = iCubMotor.RShoulderPitch
    joint2 = iCubMotor.RShoulderRoll
    joint3 = iCubMotor.RShoulderYaw
    joint4 = iCubMotor.RElbow

    joints = [joint1, joint2, joint3, joint4]
    initial_angles = np.zeros(params.number_cpg)

    init_pos_arm = np.array([-49., 60., 66., 15., -50., -5., -5.])
    initial_angles[:init_pos_arm.shape[0]] = init_pos_arm

    kin_read.release_links([7, 8, 9])
    kin_read.set_jointangles(np.radians(init_pos_arm))
    kin_read.block_links([7, 8, 9])

    for cpg_param in cpgs:
        reaching_error = []
        for ang in changed_angle:
            initial_angles[3] = ang

            if cpg_param.shape != (4, 6):
                cpg_param = cpg_param.reshape(4, 6)

            # execute movement
            reached, reached_angles = execute_movement(pms=cpg_param, current_angles=initial_angles, radians=False)

            # compute error
            error = np.linalg.norm(goal - reached)
            reaching_error.append(error)

        reaching_errors.append(reaching_error)

    # make dataframe
    data = dict(cpg_data)
    data['reaching_error'] = reaching_errors

    # save
    path, _ = os.path.split(cpg_path)
    np.savez(path + '/best_inverse_results.npz', **data)

    # clear memory
    del iCubMotor
    gc.collect()


def load_rhi_data(path: str = 'data_out/data_RHI_jitter_1_1_sigma_prop_2.npz',
                  debug: bool = False,
                  normalize: bool = False) -> dict:
    data = np.load(path, allow_pickle=True)['arr_0'].item()
    if debug:
        plt.plot(np.sort(data['theta_inputs_train']), 'o')
        plt.grid()
        plt.show()

    if normalize:
        data['r_train'] = data['r_train'] / np.mean(data['r_train'], 1)[:, None]
        data['r_train'][np.isnan(data['r_train'])] = 0
        data['r_RHI'] = data['r_RHI'] / np.mean(data['r_RHI'], 1)[:, None]
        data['r_RHI'][np.isnan(data['r_RHI'])] = 0
    return data


def load_training_rhi_thetas(path: str = 'data_out/data_RHI_jitter_1_1_sigma_prop_2.npz') -> np.ndarray:
    data = load_rhi_data(path, normalize=False)
    thetas = data['theta_proprioception_inputs_test'][data['theta_proprioception_inputs_test'] == data['theta_vision_inputs_test']]
    return np.unique(thetas)


def merge_training_data(rhi_path: str = 'data_out/data_RHI_jitter_1_1_sigma_prop_2.npz',
                        cpg_path: Optional[str] = 'results/RHI_j11_sigma2/network_inverse_kinematic/best_inverse_results',
                        normalize_rhi_data: bool = False) -> pd.DataFrame:

    if not os.path.isfile(cpg_path):
        make_reaching_error_data()

    # create rhi dataframe
    rhi_data = load_rhi_data(rhi_path, normalize=normalize_rhi_data)

    # train only on thetas that are the same
    bool_mask = rhi_data['theta_proprioception_inputs_test'] == rhi_data['theta_vision_inputs_test']

    rhi_df = pd.DataFrame({
        'theta': rhi_data['theta_proprioception_inputs_test'][bool_mask],
        'r_output': rhi_data['r_RHI_norm'][bool_mask].tolist(),
    })

    # create cpg dataframe
    if cpg_path is None:
        cpg_data = choose_run_with_lowest_error()
    else:
        cpg_data = load_inverse_kinematic_results(cpg_path)

    thetas = cpg_data['changed_angle']
    cpg_df = pd.DataFrame({
        'theta': thetas,
        'reaching_error': cpg_data['reaching_error'].tolist(),
    })

    # sort by theta
    cpg_df = cpg_df.sort_values('theta')
    cpg_df['category'] = np.arange(0, thetas.shape[0])
    return pd.merge(rhi_df, cpg_df, on='theta', how='left')


def merge_test_data(rhi_path: str = 'data_out/data_RHI_jitter_1_1_sigma_prop_2.npz',
                    cpg_path: Optional[str] = 'results/RHI_j11_sigma2/network_inverse_kinematic/best_inverse_results',
                    merge_nearest_neighbor: bool = False) -> pd.DataFrame:

    if not os.path.isfile(cpg_path):
        make_reaching_error_data()

    # create rhi dataframe
    rhi_data = load_rhi_data(rhi_path)

    # test only on thetas that are not the same
    bool_mask = rhi_data['theta_proprioception_inputs_test'] != rhi_data['theta_vision_inputs_test']

    rhi_df = pd.DataFrame({
        'theta': rhi_data['theta_proprioception_inputs_test'][bool_mask],
        'vision_theta': rhi_data['theta_vision_inputs_test'][bool_mask],
        'r_output': rhi_data['r_RHI_norm'][bool_mask].tolist(),
    })

    # create cpg dataframe
    if cpg_path is None:
        cpg_data = choose_run_with_lowest_error()
    else:
        cpg_data = load_inverse_kinematic_results(cpg_path)

    thetas = cpg_data['changed_angle']
    cpg_df = pd.DataFrame({
        'theta': thetas,
        'reaching_error': cpg_data['reaching_error'].tolist(),
    })

    # Sort both DataFrames by the 'theta' column. This is required for merge_asof to work correctly.
    rhi_df = rhi_df.sort_values('theta')
    cpg_df = cpg_df.sort_values('theta')
    cpg_df['category'] = np.arange(0, thetas.shape[0])

    # Perform the merge
    if merge_nearest_neighbor:
        return pd.merge_asof(rhi_df, cpg_df, on='theta', direction='nearest')
    else:
        return pd.merge(rhi_df, cpg_df, on='theta', how='left')


def save_mlp(mlp, scaler, save_path: str = 'results/mlp_execute_movement/') -> None:
    import pickle

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the MLP
    with open(save_path + 'mlp_model.pkl', 'wb') as f:
        pickle.dump(mlp, f)

    # Save the scaler
    with open(save_path + 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


def load_mlp(save_path: str = 'results/mlp_execute_movement/'):
    import pickle

    with open(save_path + 'mlp_model.pkl', 'rb') as f:
        mlp = pickle.load(f)

    with open(save_path + 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return mlp, scaler


# TODO: Implement version with categorical MLP (one hot encoding for categorical features)
def train_mlp(trainings_df: pd.DataFrame,
              input_col: str = 'r_output',
              target_col: str = 'category',
              test_size: float = 0.2,
              hidden_layer_size: tuple = (128, 128,),
              random_state: Optional[int] = 42,
              print_mse: bool = True) -> tuple:

    X = np.array(trainings_df[input_col].tolist())
    y = np.array(trainings_df[target_col].tolist())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the MLP
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_size, max_iter=10_000, random_state=random_state)
    mlp.fit(X_train_scaled, y_train)

    # Make predictions
    y_train_pred = mlp.predict(X_train_scaled)
    y_test_pred = mlp.predict(X_test_scaled)

    # Calculate MSE
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    if print_mse:
        print(f"Hidden layer size: {hidden_layer_size}")
        print(f"Training MSE: {mse_train:.4f}")
        print(f"Test MSE: {mse_test:.4f}\n")

    return mlp, scaler, mse_test


def train_mlps(trainings_df: pd.DataFrame,
               input_col: str = 'r_output',
               target_col: str = 'category',
               test_size: float = 0.2,
               hidden_layer_size: tuple = (128, 128,),
               max_iter: int = 100) -> tuple:

    best_mlp, best_scaler, best_mse = None, None, np.inf
    for _ in range(max_iter):
        mlp, scaler, mse = train_mlp(trainings_df,
                                     input_col=input_col,
                                     target_col=target_col,
                                     test_size=test_size,
                                     hidden_layer_size=hidden_layer_size,
                                     random_state=None, print_mse=False)
        if mse < best_mse:
            best_mse = mse
            best_mlp = mlp
            best_scaler = scaler

    return best_mlp, best_scaler


def get_prediction(r_input: np.ndarray | list,
                   trained_mlp,
                   training_scaler,
                   cpg_reshape: bool = False) -> np.ndarray:
    """
    Predict CPG parameters for given input

    :param r_input: RHI input from Valentin
    :param trained_mlp: Trained MLP which maps input to CPG parameters
    :param training_scaler: The scaler used to scale the input
    :param cpg_reshape: If the CPG parameters should be reshaped or not
    :return: cpg parameters
    """
    if isinstance(r_input, list):
        r_input = np.array(r_input)

    # reshape if 1D
    if r_input.ndim == 1:
        r_input = r_input.reshape(1, -1)

    n_in = r_input.shape[0]
    r_input_scaled = training_scaler.transform(r_input)
    y_pred = trained_mlp.predict(r_input_scaled)

    if cpg_reshape:
        if n_in == 1:
            y_pred = y_pred.reshape(4, 6)
        else:
            y_pred = y_pred.reshape(n_in, 4, 6)

    return y_pred


def test_mlp(test_df: pd.DataFrame,
             trained_mlp,
             training_scaler,
             input_col: str = 'r_output',
             target_col: str = 'cpg_output',
             test_id: Optional[int] = None,
             print_mse: bool = True) -> tuple:

    from sklearn.metrics import mean_squared_error

    r_test = np.array(test_df[input_col].tolist())
    cpg_test = np.array(test_df[target_col].tolist())

    if test_id is not None:
        # shape of r_test: (179685, 50)
        assert test_id < r_test.shape[0], "test_id out of range!"
        r_test = r_test[test_id]
        cpg_test = cpg_test[test_id].reshape(1, -1)

    cpg_pred = get_prediction(r_test, trained_mlp, training_scaler)
    mse = mean_squared_error(cpg_test, cpg_pred)
    if print_mse:
        print(f"MSE: {mse:.4f}")

    return cpg_pred, mse


if __name__ == '__main__':
    make_reaching_error_data()
