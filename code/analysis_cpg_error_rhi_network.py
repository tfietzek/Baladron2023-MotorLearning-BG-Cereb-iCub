import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import gc
import importlib

# Model
from kinematic import *
from cpg import *
from train_BG_reaching import execute_movement, random_goal2_iCub

# CPG
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *


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


def execute_cpg_params(df: pd.DataFrame,
                       goal: np.ndarray = np.array([-0.25, 0.1, 0.15]),
                       n_samples: Optional[int] = None,
                       folder: str = 'results/RHI_j11_sigma2/bg_reach_testing/',
                       do_plot: bool = False):

    if n_samples is not None:
        df = df.sample(n_samples)

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

    init_pos_arm = np.array([-49., 60., 66., 15., -50., -5., -5.])
    initial_angles[:init_pos_arm.shape[0]] = init_pos_arm

    kin_read.release_links([7, 8, 9])
    kin_read.set_jointangles(np.radians(init_pos_arm))
    kin_read.block_links([7, 8, 9])

    for index, row in df.iterrows():
        initial_angles[3] = row['theta']
        cpg_pred = np.array(row['bg_cpg_output']).reshape(4, 6)

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
    np.savez(folder + 'cpg_error_on_rhi_data.npz', **results_test_set)

    if do_plot:
        plot_reaching_error_on_test_data(test_path=folder + 'cpg_error_on_rhi_data.npz', show_plot=False)

    # clean
    del results_test_set
    gc.collect()


if __name__ == '__main__':
    df = pd.read_parquet('results/RHI_j11_sigma2/bg_reach_testing/test_results.parquet')
    execute_cpg_params(df=df, folder='results/RHI_j11_sigma2/bg_reach_testing/', do_plot=True)
