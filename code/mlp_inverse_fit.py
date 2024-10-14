import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from plot_error_inverse_kinematics import choose_run_with_lowest_error


def load_rhi_data(path: str = 'RHI/data.npz',
                  debug: bool = False) -> dict:
    data = np.load(path, allow_pickle=True)['arr_0'].item()
    if debug:
        plt.plot(np.sort(data['theta_inputs_train']), 'o')
        plt.grid()
        plt.show()
    return data


def load_rhi_thetas(path: str = 'RHI/data.npz') -> np.ndarray:
    data = load_rhi_data(path)
    return np.unique(data['theta_inputs_train'])


def merge_training_data(rhi_path: str = 'RHI/data.npz',
                        cpg_path: str = 'results/network_inverse_kinematic/') -> pd.DataFrame:
    rhi_data = load_rhi_data(rhi_path)

    # create rhi dataframe
    rhi_df = pd.DataFrame({
        'theta': rhi_data['theta_inputs_train'],
        'r_output': rhi_data['r_train'].tolist(),
    })

    # create cpg dataframe
    cpg_data = choose_run_with_lowest_error(cpg_path)
    thetas = cpg_data['changed_angle']
    cpgs = cpg_data['cpg_params_to_goals'].reshape(thetas.shape[0], -1)

    cpg_df = pd.DataFrame({
        'theta': thetas,
        'cpg_output': cpgs.tolist(),
    })

    return pd.merge(rhi_df, cpg_df, on='theta', how='left')


def merge_test_data(rhi_path: str = 'RHI/data.npz',
                    cpg_path: str = 'results/network_inverse_kinematic/') -> pd.DataFrame:
    # create rhi dataframe
    rhi_data = load_rhi_data(rhi_path)
    rhi_df = pd.DataFrame({
        'theta': rhi_data['theta_proprioception_inputs_test'],
        'vision_theta': rhi_data['theta_vision_inputs_test'],
        'r_output': rhi_data['r_test'].tolist(),
    })

    # create cpg dataframe
    cpg_data = choose_run_with_lowest_error(cpg_path)
    thetas = cpg_data['changed_angle']
    cpgs = cpg_data['cpg_params_to_goals'].reshape(thetas.shape[0], -1)

    cpg_df = pd.DataFrame({
        'theta': thetas,
        'cpg_output': cpgs.tolist(),
    })

    # Sort both DataFrames by the 'theta' column. This is required for merge_asof to work correctly.
    rhi_df = rhi_df.sort_values('theta')
    cpg_df = cpg_df.sort_values('theta')

    # Perform the merge
    return pd.merge_asof(rhi_df, cpg_df, on='theta', direction='nearest')


def train_mlp(trainings_df: pd.DataFrame,
              input_col: str = 'r_output',
              target_col: str = 'cpg_output',
              test_size: float = 0.2,
              hidden_layer_size: tuple = (64, 64,),
              random_state: Optional[int] = 42,
              print_mse: bool = True) -> tuple:

    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    X = np.array(trainings_df[input_col].tolist())
    y = np.array(trainings_df[target_col].tolist())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

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

    # load data
    train_df = merge_training_data()
    test_df = merge_test_data()

    hidden_layer_sizes = (
        (32,), (32, 32), (64,), (64, 64), (128,), (128, 128), (256,), (256, 256),
    )

    for hidden_layer_size in hidden_layer_sizes:
        mlp, scaler, _ = train_mlp(train_df, hidden_layer_size=hidden_layer_size, random_state=None)

    test_mlp(test_df, mlp, scaler, test_id=0)