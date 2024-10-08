"""
Code for the paper:

Baladron, J., Vitay, J., Fietzek, T. and Hamker, F. H.
The contribution of the basal ganglia and cerebellum to motor learning: a neuro-computational approach.

Copyright the Authors (License MIT)
"""


# Basic imports
import importlib
import sys
import time
import numpy as np
from pathlib import Path

# ANNarchy
from ANNarchy import *
setup(num_threads=2)

# Model
from kinematic import *
from cpg import *
from train_BG_reaching import execute_movement, random_goal2_iCub

# CPG
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *
from scipy.optimize import minimize

# find cpg parameters to how many random targets?
num_trials = 66
debug: bool = False

# Prepare save directory
folder_net = './results/network_inverse_kinematic'
Path(folder_net).mkdir(parents=True, exist_ok=True)

# Compile the network
compile_folder = './annarchy/inverse_kinematic'
Path(compile_folder).mkdir(parents=True, exist_ok=True)
compile(directory=compile_folder)

# Initialize robot connection
sys.path.append('../../CPG_lib/MLMPCPG')
sys.path.append('../../CPG_lib/icubPlot')
iCubMotor = importlib.import_module(params.iCub_joint_names)

# CPG layers
RG_Layer_E = []
RG_Layer_F = []
PF_Layer_E = []
PF_Layer_F = []
MN_Layer_E = []
MN_Layer_F = []
myT = fSetTiming()

# Create list of CPG objects
myCont = fnewMLMPcpg(params.number_cpg)
# Instantiate the CPG list with iCub robot data
myCont = fSetCPGNet(myCont, params.my_iCub_limits, params.positive_angle_dir)

# List of joints in the iCub
"""
    NeckPitch, NeckRoll, NeckYaw, EyesTilt, EyesVersion, EyesVergence, TorsoYaw, TorsoRoll, TorsoPitch, RShoulderPitch, RShoulderRoll, \
    RShoulderYaw, RElbow, RWristProsup, RWristPitch, RWristYaw, RHandFinger, RThumbOppose, RThumbProximal, RThumbDistal, RIndexProximal, \
    RIndexDistal, RMiddleProximal, RMiddleDistal, RPinky, RHipPitch, RHipRoll, RHipYaw, RKnee, RAnklePitch, RAnkleRoll, \
    LShoulderPitch, LShoulderRoll, LShoulderYaw, LElbow, LWristProsup, LWristPitch, LWristYaw, LHandFinger, LThumbOppose, LThumbProximal, \
    LThumbDistal, LIndexProximal, LIndexDistal, LMiddleProximal, LMiddleDistal, LPinky, LHipPitch, LHipRoll, LHipYaw, LKnee, \
    LAnklePitch, LAnkleRoll
"""

# Initiate PF and RG patterns for the joints
joint1 = iCubMotor.RShoulderPitch
joint2 = iCubMotor.RShoulderRoll
joint3 = iCubMotor.RShoulderYaw
joint4 = iCubMotor.RElbow

joints = [joint1, joint2, joint3, joint4]


AllJointList = joints
num_joints = 4
angles = np.zeros(params.number_cpg)

angles[iCubMotor.RShoulderPitch] = 10
angles[iCubMotor.RShoulderRoll] = 15.
angles[iCubMotor.RElbow] = 15.
initial_angles = np.copy(angles)
initial_position = wrist_position_icub(np.radians(initial_angles[joints]))[0:3]

# Update CPG initial position (reference position)
for i in range(0, len(myCont)):
    myCont[i].fUpdateInitPos(angles[i])

# Update all joints CPG, it is important to update all joints
# at least one time, otherwise, non used joints will be set to
# the default init position in the CPG which is 0
for i in range(0, len(myCont)):
    myCont[i].fUpdateLocomotionNetwork(myT, angles[i])

for ff in range(num_joints):
    VelRG_Pat1[ff].disable_learning()
    VelRG_Pat2[ff].disable_learning()
    VelRG_Pat3[ff].disable_learning()
    VelRG_Pat4[ff].disable_learning()
    VelPF_Pat1[ff].disable_learning()
    VelPF_Pat2[ff].disable_learning()
    VelInjCurr[ff].disable_learning()
#VelInter.disable_learning()

RG_Pat1.factor_exc = 1.0
RG_Pat2.factor_exc = 1.0
RG_Pat3.factor_exc = 1.0
RG_Pat4.factor_exc = 1.0
PF_Pat1.factor_exc = 1.0
PF_Pat2.factor_exc = 1.0
Inj_Curr.factor_exc = 1.0


def readout_CPG(intermediate_id: int | None = None,
                num_intermediate_neurons: int = Intermediate.geometry[0]) -> np.ndarray:

    if intermediate_id is None:
        intermediate_id = np.random.randint(0, num_intermediate_neurons - 1)

    Intermediate[intermediate_id].baseline = 1.0
    simulate(150)

    pms = np.zeros((4, 6))
    for j in range(4):
        RG1_joint = 5 + parameter_readout(RG_Pat1[j, :], 0, 5)
        RG2_joint = 5 + parameter_readout(RG_Pat2[j, :], 0, 5)
        RG3_joint = 0.001 + parameter_readout(RG_Pat3[j, :], -4, 4)
        RG4_joint = 5 + parameter_readout(RG_Pat4[j, :], 0, 10)

        PF1_joint = 0.001 + parameter_readout(PF_Pat1[j, :], 0, 2.0)
        PF2_joint = 0.001 + parameter_readout(PF_Pat2[j, :], 0, 2.0)

        pms[j] = [RG1_joint, RG2_joint, RG3_joint, RG4_joint, PF1_joint, PF2_joint]

    return pms


def inverse_kinematics(goal: np.ndarray,
                       initial_cpg_params: np.ndarray,
                       initial_angles: np.ndarray,
                       abort_criterion: float = 1e-4,
                       max_iterations: int = 1_000,
                       radians: bool = True) -> np.ndarray:
    """
    Returns CPG Parameters to a specific goal position

    :param goal: cartesian position (x, y, z) of the end-effector
    :param initial_angles: initial angles of the iCub robot
    :param abort_criterion: criterion to stop the optimization in [m]
    :param max_iterations: maximum number of iterations
    :param radians: if the initial angles are in radians
    :return: CPG parameters to goal position
    """

    if not radians:
        initial_angles = np.radians(initial_angles)

    pms_shape = initial_cpg_params.shape

    def objective_function(cpg_params: np.ndarray,
                           initial_angles: np.ndarray = initial_angles,
                           cpg_params_shape: tuple = pms_shape) -> float:
        """
        Objective function to minimize
        """

        if cpg_params.shape != cpg_params_shape:
            cpg_params = cpg_params.reshape(cpg_params_shape)

        final_pos = execute_movement(cpg_params, initial_angles)
        position_error = np.linalg.norm(goal - final_pos)

        # TODO: Implement angle penalty?
        return position_error

    if objective_function(initial_cpg_params) < abort_criterion:
        print('Initial position is already close enough')
        return initial_cpg_params

    # Run the optimization
    result = minimize(
        objective_function,
        initial_cpg_params.reshape(-1),  # must be 1D array
        method='L-BFGS-B',
        # bounds=bounds,
        options={'ftol': abort_criterion, 'maxiter': max_iterations},
    )

    if result.success:
        return result.x
    else:
        print('Optimization failed')
        return initial_cpg_params


inverse_results = {
    'initial_position': [],
    'initial_angle': [],
    'goals': [],
    'cpg_params_to_goals': [],
    'cpg_params_init': [],
}

# for t in range(num_trials):
#     # Choose the goal
#     goal = random_goal2_iCub(initial_position)
#     init_pms = readout_CPG()

#     # Run the optimization
#     pms = inverse_kinematics(goal=goal,
#                              initial_cpg_params=init_pms,
#                              initial_angles=initial_angles, radians=False)

#     inverse_results['goals'].append(goal)
#     inverse_results['cpg_params_to_goals'].append(pms)
#     inverse_results['cpg_params_init'].append(init_pms)

#     if debug:
#         print('Goal:', goal)
#         print('CPG:', pms)
#         print('Difference CPG:', pms.reshape(-1) - init_pms.reshape(-1))

# np.savez(folder_net + 'inverse_results', **inverse_results)


initial_angles = np.zeros(params.number_cpg)

init_pos_arm = np.array([-49., 60., 66., 15., -60., -5., -5.])
initial_angles[:init_pos_arm.shape[0]] = init_pos_arm

kin_read.release_links([7, 8, 9])
kin_read.set_jointangles(init_pos_arm)
kin_read.block_links([7, 8, 9])

goal = [-0.25, 0.1, 0.15]

min_angle = 15
max_angle = 81
step = 2
angles = [i for i in range(min_angle, max_angle, step)]

for ang in angles:
    initial_angles[3] = ang
    init_pms = readout_CPG()
    initial_position = wrist_position_icub(np.radians(initial_angles[joints]))[0:3]


    # Run the optimization
    pms = inverse_kinematics(goal=goal,
                            initial_cpg_params=init_pms,
                            initial_angles=initial_angles,
                            max_iterations=5000,
                            radians=False)

    reached = execute_movement(np.reshape(pms, (4,6)), initial_angles)
    print('Goal:', goal, reached, np.linalg.norm(goal - reached))

    if debug:
        print('Goal:', goal)
        print('CPG:', pms)
        print('Difference CPG:', pms.reshape(-1) - init_pms.reshape(-1))

    inverse_results['initial_position'].append(initial_position)
    inverse_results['initial_angle'].append(ang)
    inverse_results['goals'].append(goal)
    inverse_results['cpg_params_to_goals'].append(pms)
    inverse_results['cpg_params_init'].append(init_pms)

np.savez(folder_net + '/inverse_results', **inverse_results)
