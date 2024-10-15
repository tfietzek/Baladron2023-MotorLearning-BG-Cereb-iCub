import importlib
import os.path

import pandas as pd
import numpy as np

# Model
from kinematic import *
from cpg import *
from train_BG_reaching import execute_movement, random_goal2_iCub
from mlp_inverse_fit import load_rhi_thetas

# CPG
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *
from scipy.optimize import minimize

from mlp_inverse_fit import train_mlp, test_mlp, merge_training_data, merge_test_data, get_prediction


def plot_reaching_error_on_test_data(test_path: str = 'results/mlp_execute_movement/results_on_test_set.npz',
                                     show_plot: bool = False) -> None:
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f'No data found at {test_path}')

    test_data = np.load(test_path)

    df = pd.DataFrame({
        'proprio': test_data['proprio_angles'],
        'vision': test_data['vision_angles'],
        'diff': test_data['proprio_angles'] - test_data['vision_angles'],
        'error': test_data['errors']
    })

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['proprio'], df['error'], c=np.abs(df['diff']), cmap='RdBu_r')
    plt.colorbar(scatter, label='diff')

    plt.xlabel('Proprio theta in [°]')
    plt.ylabel('Error in [m]')

    plt.tight_layout()
    plt.savefig('results/mlp_execute_movement/reaching_error_on_test_set.pdf', dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()


def plot_reaching_error_on_train_data(train_path: str = 'results/mlp_execute_movement/results_on_train_set.npz',
                                      show_plot: bool = False) -> None:
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f'No data found at {train_path}')

    train_data = np.load(train_path)

    df = pd.DataFrame({
        'theta': train_data['proprio_angles'],
        'error': train_data['errors']
    })

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['theta'], df['error'], c=np.abs(df['error']), cmap='RdBu_r')
    plt.colorbar(scatter, label='Error in [m]')

    plt.xlabel('Proprio + vis theta in [°]')
    plt.ylabel('Error in [m]')

    plt.tight_layout()
    plt.savefig('results/mlp_execute_movement/reaching_error_on_train_set.pdf', dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()


def ols_reach_error(test_data_path: str = 'results/mlp_execute_movement/results_on_test_set.npz',
                    show_plot: bool = False,
                    use_abs_diff: bool = True) -> tuple[float, float]:

    from scipy import stats
    import statsmodels.api as sm

    if not os.path.isfile(test_data_path):
        raise FileNotFoundError(f'No data found at {test_data_path}')

    test_data = np.load(test_data_path)

    df = pd.DataFrame({
        'diff': test_data['proprio_angles'] - test_data['vision_angles'],
        'error': test_data['errors']
    })

    if use_abs_diff:
        df['diff'] = np.abs(df['diff'])

    # Add a constant to the independent variable
    X = sm.add_constant(df['diff'])

    # Fit the model
    model = sm.OLS(df['error'], X).fit()

    # Calculate the correlation coefficient
    correlation_coefficient, p_value = stats.pearsonr(df['diff'], df['error'])

    # Create a scatter plot with the regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(df['diff'], df['error'], alpha=0.5)
    plt.plot(df['diff'], model.predict(X), color='red', linewidth=2)

    plt.xlabel('Diff')
    plt.ylabel('Error')
    plt.title(f'OLS Regression, r={correlation_coefficient:.3f}, p={p_value:.3f}')

    plt.tight_layout()
    plt.savefig('results/mlp_execute_movement/ols_reach_error.pdf', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

    return correlation_coefficient, p_value


def save_mlp(mlp, scaler, save_path: str = 'results/mlp_execute_movement/' ) -> None:
    import pickle

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the MLP
    with open(save_path + 'mlp_model.pkl', 'wb') as f:
        pickle.dump(mlp, f)

    # Save the scaler
    with open(save_path + 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


def load_mlp(save_path: str = 'results/mlp_execute_movement/' ):
    import pickle

    with open(save_path + 'mlp_model.pkl', 'rb') as f:
        mlp = pickle.load(f)

    with open(save_path + 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return mlp, scaler


if __name__ == '__main__':
    # Parameters
    n_samples: None | int = None  # number of samples to test the movement. If None, all samples are used
    save_best_model: bool = True
    print_error: bool = False

    # load data
    train_df = merge_training_data()

    # Initialize robot connection
    sys.path.append('../../CPG_lib/MLMPCPG')
    sys.path.append('../../CPG_lib/icubPlot')
    iCubMotor = importlib.import_module(params.iCub_joint_names)

    # init icub + cpg
    joint1 = iCubMotor.RShoulderPitch
    joint2 = iCubMotor.RShoulderRoll
    joint3 = iCubMotor.RShoulderYaw
    joint4 = iCubMotor.RElbow

    joints = [joint1, joint2, joint3, joint4]
    initial_angles = np.zeros(params.number_cpg)

    init_pos_arm = np.array([-49., 60., 66., 15., -60., -5., -5.])
    initial_angles[:init_pos_arm.shape[0]] = init_pos_arm

    kin_read.release_links([7, 8, 9])
    kin_read.set_jointangles(init_pos_arm)
    kin_read.block_links([7, 8, 9])

    goal = np.array([-0.25, 0.1, 0.15])

    # train mlp
    print("Training MLP")
    max_iter = 50
    best_mlp, best_scaler, best_mse = None, None, np.inf
    for _ in range(max_iter):
        mlp, scaler, mse = train_mlp(train_df, hidden_layer_size=(256, 256), random_state=None, print_mse=False)
        if mse < best_mse:
            best_mse = mse
            best_mlp = mlp
            best_scaler = scaler

    if save_best_model:
        save_mlp(best_mlp, best_scaler)
    print(f"Best MLP MSE: {best_mse}\n")

    # test mlp with train input
    print("Testing MLP with train data...")
    if n_samples is None:
        df_inputs = train_df
    else:
        df_inputs = train_df.sample(n=n_samples)

    results_training_set = {
        'goal': goal,
        'initial_angles': [],
        'proprio_angles': [],
        'reached_pos': [],
        'reached_angles': [],
        'errors': [],
    }

    for index, row in df_inputs.iterrows():
        initial_angles[3] = row['theta']

        r_input = np.array(row['r_output'])
        cpg_pred = get_prediction(r_input, best_mlp, best_scaler, cpg_reshape=True)

        # execute movement
        reached, reached_angles = execute_movement(pms=cpg_pred, current_angles=initial_angles, radians=False)

        results_training_set['initial_angles'].append(initial_angles)
        results_training_set['reached_pos'].append(reached)
        results_training_set['reached_angles'].append(reached_angles)
        results_training_set['proprio_angles'].append(row['theta'])

        # calculate error
        results_training_set['errors'].append(np.linalg.norm(goal - reached))

        if print_error:
            print(f'Error: {results_training_set["errors"][-1]}')

    folder = 'results/mlp_execute_movement/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.savez(folder + 'results_on_training_set.npz', **results_training_set)

    plot_reaching_error_on_train_data(train_path=folder + 'results_on_training_set.npz', show_plot=False)

    # test mlp with test inputs
    print("Testing MLP with test data...")
    if n_samples is None:
        df_inputs = merge_test_data()
    else:
        df_inputs = merge_test_data().sample(n=n_samples)

    results_test_set = {
        'goal': goal,
        'initial_angles': [],
        'proprio_angles': [],
        'vision_angles': [],
        'reached_pos': [],
        'reached_angles': [],
        'errors': [],
    }

    for index, row in df_inputs.iterrows():
        initial_angles[3] = row['theta']

        r_input = np.array(row['r_output'])
        cpg_pred = get_prediction(r_input, best_mlp, best_scaler, cpg_reshape=True)

        # execute movement
        reached, reached_angles = execute_movement(pms=cpg_pred, current_angles=initial_angles, radians=False)

        results_test_set['initial_angles'].append(initial_angles)
        results_test_set['reached_pos'].append(reached)
        results_test_set['reached_angles'].append(reached_angles)
        results_test_set['vision_angles'].append(row['vision_theta'])
        results_test_set['proprio_angles'].append(row['theta'])

        # calculate error
        results_test_set['errors'].append(np.linalg.norm(goal - reached))

        if print_error:
            print(f'Error: {results_test_set["errors"][-1]}')

    np.savez(folder + 'results_on_test_set.npz', **results_test_set)

    plot_reaching_error_on_test_data(folder + 'results_on_test_set.npz', show_plot=False)
    ols_reach_error(folder + 'results_on_test_set.npz', show_plot=False)
