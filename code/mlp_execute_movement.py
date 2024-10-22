import importlib
import os.path
from typing import Optional
import pandas as pd
import numpy as np
import gc

# Model
from kinematic import *
from cpg import *
from train_BG_reaching import execute_movement, random_goal2_iCub
from mlp_inverse_fit import load_training_rhi_thetas, save_mlp

# CPG
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *
from scipy.optimize import minimize

from mlp_inverse_fit import train_mlp, test_mlp, train_mlps, merge_training_data, merge_test_data, get_prediction


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
    scatter = plt.scatter(df['proprio'] + np.random.uniform(low=-0.34, high=0.34, size=len(df)), df['error'],
                          c=np.abs(df['diff']), cmap='RdBu_r', s=2.)
    plt.colorbar(scatter, label='diff')

    plt.xlabel('Proprio theta in [°]')
    plt.ylabel('Error in [m]')

    plt.tight_layout()

    path, _ = os.path.split(test_path)
    plt.savefig(path + '/reaching_error_on_test_set.pdf', dpi=300, bbox_inches='tight')

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

    path, _ = os.path.split(train_path)
    plt.savefig(path + '/reaching_error_on_train_set.pdf', dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()


def ols_reach_error(test_data_path: str = 'results/mlp_execute_movement/results_on_test_set.npz',
                    show_plot: bool = False,
                    use_abs_diff: bool = False) -> tuple[float, float]:
    from scipy import stats
    import statsmodels.api as sm

    if not os.path.isfile(test_data_path):
        raise FileNotFoundError(f'No data found at {test_data_path}')

    test_data = np.load(test_data_path)

    df = pd.DataFrame({
        'proprio': test_data['proprio_angles'],
        'diff': test_data['proprio_angles'] - test_data['vision_angles'],
        'error': test_data['errors']
    })

    if use_abs_diff:
        df['diff'] = np.abs(df['diff'])

    #df = df[df['proprio'] == 48.]

    # Add a constant to the independent variable
    X = sm.add_constant(df['diff'])

    # Fit the model
    model = sm.OLS(df['error'], X).fit()

    # Calculate the correlation coefficient
    correlation_coefficient, p_value = stats.pearsonr(df['diff'], df['error'])

    # Create a scatter plot with the regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(df['diff'], df['error'], alpha=0.5, s=2.)
    plt.plot(df['diff'], model.predict(X), color='red', linewidth=2)

    plt.xlabel('Diff')
    plt.ylabel('Error')
    plt.title(f'OLS Regression, r={correlation_coefficient:.3f}, p={p_value:.3f}')

    plt.tight_layout()
    path, _ = os.path.split(test_data_path)
    plt.savefig(path + '/ols_reach_error.pdf', dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

    return correlation_coefficient, p_value


def execute_movement_with_mlp_on_test_set(
        df_inputs: pd.DataFrame,
        mlp,
        scaler,
        goal: np.ndarray = np.array([-0.25, 0.1, 0.15]),
        n_samples: Optional[int] = None,
        folder: str = 'results/mlp_execute_movement/'
) -> None:
    if n_samples is not None:
        df_inputs = df_inputs.sample(n_samples)

    # get cpg prediction
    r_input = df_inputs['r_output'].tolist()
    df_inputs['cpg_pred'] = get_prediction(r_input, mlp, scaler).tolist()

    # save results in this dict
    results_test_set = {
        'goal': goal,
        'initial_angles': [],
        'proprio_angles': [],
        'vision_angles': [],
        'reached_pos': [],
        'reached_angles': [],
        'errors': [],
    }

    # init icub + cpg
    iCubMotor = importlib.import_module(params.iCub_joint_names)

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

    for index, row in df_inputs.iterrows():
        initial_angles[3] = row['theta']
        cpg_pred = np.array(row['cpg_pred']).reshape(4, 6)

        # execute movement
        reached, reached_angles = execute_movement(pms=cpg_pred, current_angles=initial_angles, radians=False)

        results_test_set['initial_angles'].append(initial_angles)
        results_test_set['reached_pos'].append(reached)
        results_test_set['reached_angles'].append(reached_angles)
        results_test_set['vision_angles'].append(row['vision_theta'])
        results_test_set['proprio_angles'].append(row['theta'])

        # calculate error
        results_test_set['errors'].append(np.linalg.norm(goal - reached))

    # save results
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savez(folder + 'results_on_test_set.npz', **results_test_set)

    plot_reaching_error_on_test_data(test_path=folder + 'results_on_test_set.npz', show_plot=False)
    ols_reach_error(test_data_path=folder + 'results_on_test_set.npz', show_plot=False, use_abs_diff=True)

    # clean
    del results_test_set
    gc.collect()


if __name__ == '__main__':
    data_set = (0, 1)[int(sys.argv[1])]
    hidden_layer_sizes = ((64, 64,), (128, 128,), (128, 64,), (256, 256,), (256, 128,), (256, 64,),)[int(sys.argv[2])]

    # data paths
    inv_path = ('results/RHI_j11_sigma2/network_inverse_kinematic/inverse_results_run1',
                'results/RHI_j12_sigma4/network_inverse_kinematic/inverse_results_run7')[data_set]
    rhi_path = ('data_out/data_RHI_jitter_1_1_sigma_prop_2.npz',
                'data_out/data_RHI_jitter_1_2_sigma_prop_4.npz')[data_set]
    data_name = ('RHI_j11_sigma2', 'RHI_j12_sigma4')[data_set]

    df_train = merge_training_data(rhi_path=rhi_path, cpg_path=inv_path)
    df_test = merge_test_data(rhi_path=rhi_path, cpg_path=inv_path)

    for hidden_layer_size in hidden_layer_sizes:
        # save results in this folder
        folder = f'results/{data_name}/mlp_shape[{hidden_layer_size[0], hidden_layer_size[1]}]/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # train mlp
        print(f'Training MLP with hidden layer size: {hidden_layer_size}')
        mlp, scaler = train_mlps(df_train,
                                 hidden_layer_size=hidden_layer_size,
                                 max_iter=5,
                                 test_size=0.2)

        save_mlp(mlp, scaler, save_path=folder)

        # test mlp on test set
        print('Test MLP on test set')
        execute_movement_with_mlp_on_test_set(df_test, mlp, scaler,
                                              n_samples=None,
                                              folder=folder)
