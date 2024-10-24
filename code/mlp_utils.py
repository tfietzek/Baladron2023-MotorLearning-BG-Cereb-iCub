import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
from typing import Optional
import os


from plot_error_inverse_kinematics import choose_run_with_lowest_error, load_inverse_kinematic_results


def load_rhi_data(path: str,
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
    thetas = data['theta_proprioception_inputs_test'][
        data['theta_proprioception_inputs_test'] == data['theta_vision_inputs_test']]
    return np.unique(thetas)


def get_cpg_data(cpg_path: str) -> pd.DataFrame:
    cpg_data = np.load(cpg_path)

    thetas = cpg_data['changed_angle']
    cpg_df = pd.DataFrame({
        'theta': thetas,
        'reaching_error': cpg_data['reaching_error'].tolist(),
    })

    # sort by theta
    cpg_df = cpg_df.sort_values('theta')
    cpg_df['error_id'] = np.arange(cpg_df.shape[0])

    return cpg_df


def merge_training_data(rhi_path: str,
                        cpg_path: str,
                        normalize_rhi_data: bool = False) -> pd.DataFrame:
    save_name = "/training_data.parquet"
    save_path, _ = os.path.split(cpg_path)
    if os.path.isfile(save_path + save_name):
        return pd.read_parquet(save_path + save_name)

    if not os.path.isfile(cpg_path):
        print("make_reaching_error_data() must be run before merge_training_data()")

    # create rhi dataframe
    rhi_data = load_rhi_data(rhi_path, normalize=normalize_rhi_data)

    # train only on thetas that are the same
    bool_mask = rhi_data['theta_proprioception_inputs_test'] == rhi_data['theta_vision_inputs_test']

    rhi_df = pd.DataFrame({
        'theta': rhi_data['theta_proprioception_inputs_test'][bool_mask],
        'r_output': rhi_data['r_RHI_norm'][bool_mask].tolist(),
        'r_gains': rhi_data['input_gains_RHI'][bool_mask].tolist(),
    })

    # create cpg dataframe
    cpg_df = get_cpg_data(cpg_path)
    merge_df = pd.merge(rhi_df, cpg_df, on='theta', how='left')

    # save
    path, _ = os.path.split(cpg_path)
    merge_df.to_parquet(path + save_name)

    return merge_df


def merge_test_data(rhi_path: str,
                    cpg_path: str,
                    merge_nearest_neighbor: bool = False) -> pd.DataFrame:
    if not os.path.isfile(cpg_path):
        print("make_reaching_error_data() must be run before merge_training_data()")

    # create rhi dataframe
    rhi_data = load_rhi_data(rhi_path)

    # test only on thetas that are not the same
    bool_mask = rhi_data['theta_proprioception_inputs_test'] != rhi_data['theta_vision_inputs_test']

    rhi_df = pd.DataFrame({
        'theta': rhi_data['theta_proprioception_inputs_test'][bool_mask],
        'vision_theta': rhi_data['theta_vision_inputs_test'][bool_mask],
        'r_output': rhi_data['r_RHI_norm'][bool_mask].tolist(),
        'r_gains': rhi_data['input_gains_RHI'][bool_mask].tolist(),
    })

    # create cpg dataframe
    cpg_df = get_cpg_data(cpg_path)

    # Sort rhi DataFrames by the 'theta' column, because the 'merge_asof' function requires this.
    rhi_df = rhi_df.sort_values('theta')

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


def train_mlp(trainings_df: pd.DataFrame,
              input_col: str = 'r_output',
              target_col: str = 'theta',
              test_size: float = 0.2,
              hidden_layer_size: tuple = (128,),
              random_state: Optional[int] = 42,
              verbose: bool = True,
              do_loss_plot: bool = True) -> tuple:
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    X = np.array(trainings_df[input_col].tolist())
    y = trainings_df[target_col].values  # make

    # Scale the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train the classifier
    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layer_size,  # Two hidden layers
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=random_state,
        verbose=verbose
    )

    classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if do_loss_plot:
        # Plot learning curve
        plt.figure(figsize=(10, 5))
        plt.plot(classifier.loss_curve_)
        plt.title('Learning Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    return classifier, scaler, accuracy_score(y_test, y_pred)


def train_mlps(trainings_df: pd.DataFrame,
               input_col: str = 'r_output',
               target_col: str = 'theta',
               test_size: float = 0.2,
               hidden_layer_size: tuple = (128, 128,),
               max_iter: int = 100) -> tuple:

    best_mlp, best_scaler, best_accuracy = None, None, -np.inf
    for _ in range(max_iter):
        mlp, scaler, accuracy = train_mlp(trainings_df,
                                          input_col=input_col,
                                          target_col=target_col,
                                          test_size=test_size,
                                          hidden_layer_size=hidden_layer_size,
                                          random_state=None,
                                          verbose=False)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mlp = mlp
            best_scaler = scaler

    print(f"Best Test Accuracy: {best_accuracy:.4f}")
    return best_mlp, best_scaler


def get_prediction(r_input: np.ndarray | list,
                   trained_mlp,
                   training_scaler) -> np.ndarray:
    """
    Predict class for given input

    :param r_input: RHI input from Valentin
    :param trained_mlp: Trained MLP which maps input to one-hot encoded outputs
    :param training_scaler: The scaler used to scale the input
    :return: predicted classes or one-hot encoded predictions
    """
    if isinstance(r_input, list):
        r_input = np.array(r_input)

    # reshape if 1D
    if r_input.ndim == 1:
        r_input = r_input.reshape(1, -1)

    r_input_scaled = training_scaler.transform(r_input)
    y_pred_onehot = trained_mlp.predict(r_input_scaled)

    return y_pred_onehot


def test_mlp(test_df: pd.DataFrame,
             trained_mlp,
             training_scaler,
             input_col: str = 'r_output',
             target_col: str = 'theta',
             error_col: str = 'reaching_error',
             angle_id_col: str = 'error_id',
             test_id: Optional[int] = None,
             print_accuracy: bool = True) -> tuple:
    from sklearn.metrics import accuracy_score, classification_report

    r_test = np.array(test_df[input_col].tolist())
    y_test = np.array(test_df[target_col].values)

    if test_id is not None:
        assert test_id < r_test.shape[0], "test_id out of range!"
        r_test = r_test[test_id]
        y_test = y_test[test_id].reshape(1, -1)

    y_pred = get_prediction(r_test, trained_mlp, training_scaler)
    accuracy = accuracy_score(y_test, y_pred)

    if print_accuracy:
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    return y_pred, accuracy
