import importlib

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


if __name__ == '__main__':
    # load data
    train_df = merge_training_data()
    n_samples: int = 30  # number of samples to test the movement
    print_error: bool = False
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
    max_iter = 10
    best_mlp, best_scaler, best_mse = None, None, np.inf
    for _ in range(max_iter):
        mlp, scaler, mse = train_mlp(train_df, hidden_layer_size=(256, 256), random_state=None, print_mse=False)
        if mse < best_mse:
            best_mse = mse
            best_mlp = mlp
            best_scaler = scaler
    print(f"Best MLP MSE: {best_mse}\n")

    # test mlp with train input
    print("Testing MLP with train data...")
    df_inputs = train_df.sample(n=n_samples)

    results_training_set = {
        'goal': goal,
        'initial_angles': [],
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

        # calculate error
        results_training_set['errors'].append(np.linalg.norm(goal - reached))

        if print_error:
            print(f'Error: {results_training_set["errors"][-1]}')

    folder = 'results/mlp_execute_movement/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.savez(folder + 'results_on_training_set.npz', **results_training_set)

    # test mlp with test inputs
    print("Testing MLP with test data...")
    df_inputs = merge_test_data()

    results_test_set = {
        'goal': goal,
        'initial_angles': [],
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

        # calculate error
        results_test_set['errors'].append(np.linalg.norm(goal - reached))

        if print_error:
            print(f'Error: {results_test_set["errors"][-1]}')

    np.savez(folder + 'results_on_test_set.npz', **results_test_set)