import numpy as np
import importlib
import gc
import os
from typing import Optional

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


if __name__ == '__main__':
    cpg_files = (
        'results/RHI_j11_sigma2/network_inverse_kinematic/inverse_results_run1.npz',
        'results/RHI_j12_sigma4/network_inverse_kinematic/inverse_results_run9.npz')

    for cpg_file in cpg_files:
        make_reaching_error_data(cpg_path=cpg_file)
