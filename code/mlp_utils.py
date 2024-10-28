import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os
import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE

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

    cpg_df = pd.DataFrame({
        'theta': cpg_data['changed_angle'],
        'cpg': cpg_data['cpg_params_to_goals'].tolist(),
        'reaching_error': cpg_data['reaching_error'].tolist(),
    })

    # sort by theta
    cpg_df = cpg_df.sort_values('theta')
    cpg_df['error_id'] = np.arange(cpg_df.shape[0])

    return cpg_df


def merge_training_data(rhi_path: str,
                        cpg_path: str,
                        normalize_rhi_data: bool = False,
                        save_name: Optional[str] = None) -> pd.DataFrame:
    if save_name is None:
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
    merged_df = pd.merge(rhi_df, cpg_df, on='theta', how='left')

    # save
    merged_df.to_parquet(save_path + save_name)

    return merged_df


def merge_test_data(rhi_path: str,
                    cpg_path: str,
                    merge_nearest_neighbor: bool = False,
                    save_name: Optional[str] = None) -> pd.DataFrame:
    if save_name is None:
        save_name = "/test_data.parquet"
    save_path, _ = os.path.split(cpg_path)
    if os.path.isfile(save_path + save_name):
        return pd.read_parquet(save_path + save_name)

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
        merged_df = pd.merge_asof(rhi_df, cpg_df, on='theta', direction='nearest')
    else:
        merged_df = pd.merge(rhi_df, cpg_df, on='theta', how='left')

    # save
    merged_df.to_parquet(save_path + save_name)

    return merged_df


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
              tolerance: float = 1e-6,
              scaler_type: str = 'robust',
              use_smote: bool = False,
              verbose: bool = True,
              save_classification_report: Optional[str] = None,
              save_loss_plot: Optional[str] = None) -> Tuple[BaseEstimator, BaseEstimator, float]:

    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

    from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
    from sklearn.metrics import accuracy_score, classification_report

    X = np.array(trainings_df[input_col].tolist())
    y = trainings_df[target_col].values  # make

    # Choose scaler based on parameter
    scalers = {
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'power': PowerTransformer(method='yeo-johnson'),
        'quantile': QuantileTransformer(output_distribution='normal'),
    }

    # Scale the input features
    scaler = scalers.get(scaler_type, StandardScaler())
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    # Apply SMOTE for balanced sampling
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Initialize and train the classifier
    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layer_size,  # Two hidden layers
        activation='relu',
        solver='adam',
        max_iter=1_000,
        random_state=random_state,
        verbose=verbose,
        tol=tolerance,
        validation_fraction=0.1,
        n_iter_no_change=10,
        batch_size='auto',
        learning_rate_init=0.001,
        learning_rate='adaptive',
        alpha=0.0001,  # L2 penalty
    )

    classifier.fit(X_train, y_train)

    # Perform cross-validation
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
    if verbose:
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Evaluate the model
    y_pred = classifier.predict(X_test)
    if save_classification_report is not None:
        content = []
        content.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("\nModel Configuration:")
        content.append(f"Scaler: {scaler_type}")
        content.append(f"SMOTE: {use_smote}")
        content.append("\nCross-validation scores:")
        content.append(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        content.append("\nClassification Report:")
        content.append(classification_report(y_test, y_pred))
        with open(save_classification_report, 'w') as f:
            f.write('\n'.join(content))

    if save_loss_plot is not None:
        # Plot learning curve
        fig = plt.figure(figsize=(10, 5))
        plt.plot(classifier.loss_curve_)
        plt.title('Learning Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(save_loss_plot, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return classifier, scaler, accuracy_score(y_test, y_pred)


def train_mlps(trainings_df: pd.DataFrame,
               input_col: str = 'r_output',
               target_col: str = 'theta',
               test_size: float = 0.2,
               hidden_layer_size: tuple = (128, 128,),
               max_iter: int = 10,
               scaler_type: str = 'robust',
               print_accuracy: bool = True,
               plot_path: Optional[str] = None) -> Tuple[BaseEstimator, BaseEstimator]:

    best_mlp, best_scaler, best_accuracy = None, None, -np.inf
    for i in range(max_iter):
        mlp, scaler, accuracy = train_mlp(trainings_df,
                                          input_col=input_col,
                                          target_col=target_col,
                                          test_size=test_size,
                                          hidden_layer_size=hidden_layer_size,
                                          random_state=None,
                                          verbose=False,
                                          scaler_type=scaler_type,
                                          save_classification_report=plot_path + f'report_{i}.txt',
                                          save_loss_plot=plot_path + f'loss_curve_{i}.png')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mlp = mlp
            best_scaler = scaler
    if print_accuracy:
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


def extract_accuracy(report_content: str):
    """Extract the accuracy score from a report file content."""
    import re
    # Look for the accuracy line in the classification report
    accuracy_match = re.search(r'accuracy\s+([0-9.]+)', report_content)
    if accuracy_match:
        return float(accuracy_match.group(1))
    return None


def find_best_model(root_dir: str):
    """
    Search through all directories and report files to find the best performing model.
    Returns tuple of (best_folder, best_report_file, best_accuracy)
    """
    from pathlib import Path

    best_accuracy = 0
    best_folder = None
    best_report = None

    # Use glob to find all relevant directories
    # This will match power_mlp_shape*, quantile_mlp_shape*, and robust_mlp_shape*
    for scaler in ['power', 'quantile', 'robust', 'standard']:
        for folder in Path(root_dir).glob(f"{scaler}_mlp_shape*"):
            if not folder.is_dir():
                continue

            # Check all report files in the folder
            for report_file in folder.glob("report_*.txt"):
                try:
                    with open(report_file, 'r') as f:
                        content = f.read()
                        accuracy = extract_accuracy(content)

                        if accuracy and accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_folder = folder.name
                            best_report = report_file.name

                except Exception as e:
                    print(f"Error reading {report_file}: {e}")

    return best_folder, best_report, best_accuracy


if __name__ == '__main__':
    print(find_best_model(root_dir='results/RHI_j11_sigma2'))
    print(find_best_model(root_dir='results/RHI_j12_sigma4'))