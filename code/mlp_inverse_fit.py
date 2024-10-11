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


def merge_data(rhi_path: str = 'RHI/data.npz',
               cpg_path: str = 'results/network_inverse_kinematic/') -> pd.DataFrame:

    rhi_data = load_rhi_data(rhi_path)

    # create rhi dataframe
    rhi_df = pd.DataFrame({
        'theta': np.round(rhi_data['theta_inputs_train']),
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


def train_mlp(df: pd.DataFrame,
              input_col: str = 'r_output',
              target_col: str = 'cpg_output',
              test_size: float = 0.2,
              hidden_layer_size: tuple = (64, 64,),
              random_state: Optional[int] = 42,
              print_mse: bool = True):

    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    X = np.array(df[input_col].tolist())
    y = np.array(df[target_col].tolist())

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

    return mlp, scaler


if __name__ == '__main__':

    # load data
    df = merge_data()

    hidden_layer_sizes = (
        (64,), (64, 64), (128,), (128, 128), (256,), (256, 256), (512,), (512, 512),
    )

    for hidden_layer_size in hidden_layer_sizes:
        train_mlp(df, hidden_layer_size=hidden_layer_size)