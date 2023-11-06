#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for the paper:

Baladron, J., Vitay, J., Fietzek, T. and Hamker, F. H.
The contribution of the basal ganglia and cerebellum to motor learning: a neuro-computational approach.

Copyright the Authors (License MIT)

Script for the reaching task.

> python run_adaptation.py
"""

# Parameters
num_goals = 2 # Number of goals. 2 or 8 in the manuscript
num_goals_per_trial = 300 # Number of trials per goal
num_trials_test = 100 # Number of test trials with the reservoir


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
from reservoir import *
from kinematic import *
from train_BG_reaching import *

# CPG
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *


# Prepare save directory
folder_net = './results/network_g' + str(num_goals) + '_run'
if len(sys.argv) > 1:
    folder_net += '_' + sys.argv[1]
Path(folder_net).mkdir(parents=True, exist_ok=True)

# Compile the network
compile(directory="./annarchy/run_" + sys.argv[1])

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
#angles = np.radians(angles)


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

def gaussian_input(x,mu,sig):
    return np.exp(-np.power(x-mu,2.)/(2*np.power(sig,2)))

distance_history = np.zeros(num_trials_test)
goal_history= np.zeros((num_trials_test,3))
parameter_history = np.zeros((num_trials_test,4,6))
final_pos_history = np.zeros((num_trials_test,3))

max_angle = 0
num_tests = 0
a = [0,0,0]

pop.enable()
num_trials = num_goals * num_goals_per_trial
error_history = np.zeros(num_trials)
dh = np.zeros(num_trials)

###################
# BG controller
###################
print('Training BG')
goal_history, parameter_history = train_bg(num_goals)

# Compute the mean reward per trial
R_mean = np.zeros(num_goals)
alpha = 0.33 #0.75 0.33

###################
# Reservoir
###################
print('Training reservoir...')
parameter = []
goal_per_trial = []
fin_pos_trials = []
init_pos_trials = []
init_angles = []
for t in range(num_trials):

    # Select the goal
    current_goal =  goal_history[t%num_goals]
    current_params =  np.copy(parameters_per_goal(current_goal))
    goal_per_trial.append(current_goal)

    # Reinitialize the reservoir
    pop.x = Uniform(-0.01, 0.01).get_values(N)
    pop.r = np.tanh(pop.x)
    pop[1].r = np.tanh(1.0)
    pop[10].r = np.tanh(1.0)
    pop[11].r = np.tanh(-1.0)

    # Set the input
    inp[(t%num_goals)].r = 1.0
    simulate(200)

    # ISI
    inp.r = 0.0
    simulate(200)

    # Retrieve recordings
    rec = m.get()

    # Compute output
    output = rec['r'][-200:,-24:]
    output = np.mean(output,axis=0) * 2


    if(t > -1):
        current_params += output.reshape((4,6))

    can = np.copy(angles)
    parameter.append(current_params)
    final_pos = execute_movement(current_params,can)
    distance = np.linalg.norm(final_pos-current_goal)

    fin_pos_trials.append(final_pos)
    init_pos_trials.append(initial_position)
    init_angles.append(can)

    error = 0
    initial_distance = np.linalg.norm(initial_position-current_goal)
    if(t>-1):
        error = distance

    dh[t] = distance

    if(t > 10):
        # Apply the learning rule
        Wrec.learning_phase = 1.0
        Wrec.error = error
        Wrec.mean_error = R_mean[t%num_goals]
        # Learn for one step
        step()
        # Reset the traces
        Wrec.learning_phase = 0.0
        Wrec.trace = 0.0
        _ = m.get()

    # Update mean reward
    R_mean[t%num_goals] = alpha * R_mean[t%num_goals] + (1.- alpha) * error
    error_history[t] = error



## Save network data
np.save(folder_net + '/error_' + str(num_goals) + '.npy', error_history)

# Save data
np.save(folder_net + '/parameter_' + str(num_goals) + '.npy' ,parameter)
np.save(folder_net + '/goals.npy', goal_history)
np.save(folder_net + '/goal_per_trial.npy', goal_per_trial)
# np.save(folder_net + '/fin_pos_trials.npy', fin_pos_trials)
# np.save(folder_net + '/init_pos_trials.npy', init_pos_trials)
# np.save(folder_net + '/init_angles_trials.npy', init_angles)

# # Save network connectivity
# for proj in projections():
#     proj.save_connectivity(filename=folder_net + '/weights_' + proj.name + '.npz')
