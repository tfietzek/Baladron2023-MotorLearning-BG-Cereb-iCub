#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for the paper:

Baladron, J., Vitay, J., Fietzek, T. and Hamker, F. H.
The contribution of the basal ganglia and cerebellum to motor learning: a neuro-computational approach.

Copyright the Authors (License MIT)

Training procedure of the basal ganglia module in the reaching task.
"""

import importlib
import sys
import time
import numpy as np

# ANNarchy
from ANNarchy import *

# Model
from basalganglia import *
from kinematic import *

# CPG
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *


# Initialize robot connection
sys.path.append('../../CPG_lib/MLMPCPG')
sys.path.append('../../CPG_lib/icubPlot')
iCubMotor = importlib.import_module(params.iCub_joint_names)


global All_Command
global All_Joints_Sensor
global myCont, angles, myT
All_Command = []
All_Joints_Sensor = []

myT = fSetTiming()

# Create list of CPG objects
myCont = fnewMLMPcpg(params.number_cpg)
# Instantiate the CPG list with iCub robot data
myCont = fSetCPGNet(myCont, params.my_iCub_limits, params.positive_angle_dir)

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


initial_position = wrist_position_icub(np.radians(angles[joints]))[0:3]

def gaussian_input(x,mu,sig):
             return np.exp(-np.power(x-mu,2.)/(2*np.power(sig,2)))

def execute_movement(pms,current_angles):
    myCont = fnewMLMPcpg(params.number_cpg)
    myCont = fSetCPGNet(myCont, params.my_iCub_limits, params.positive_angle_dir)


    an = np.zeros((120,4))

    for j in range(4):
        myCont[joints[j]].fSetPatternRG(RG_Patterns(pms[j,0], pms[j,1], pms[j,2], pms[j,3]))
        myCont[joints[j]].fSetPatternPF(PF_Patterns(pms[j,4], pms[j,5]))

        myCont[joints[j]].RG.F.InjCurrent_value = 1.0 * myCont[joints[j]].RG.F.InjCurrent_MultiplicationFactor
        myCont[joints[j]].RG.E.InjCurrent_value = -1.0 * myCont[joints[j]].RG.E.InjCurrent_MultiplicationFactor

    #current_angles = np.copy(angles)
    current_angles = np.radians(current_angles)

    # Update CPG initial position (reference position)
    for i in range(0, len(myCont)):
        myCont[i].fUpdateInitPos(current_angles[i])

    # Execute a movement
    for i in AllJointList:
            myCont[i].fUpdateLocomotionNetwork(myT, current_angles[i])
    for idx, controller in enumerate(myCont):
            iCubMotor.MotorCommand[idx] = controller.joint.joint_motor_signal
    #iCub_robot.iCub_set_angles(iCubMotor.MotorCommand)
    All_Command.append(iCubMotor.MotorCommand[:])
    All_Joints_Sensor.append(current_angles)
    I=0
    while I<120:
        I+=1
        for i in AllJointList:
            myCont[i].fUpdateLocomotionNetwork(myT, current_angles[i])
        for idx, controller in enumerate(myCont):
            iCubMotor.MotorCommand[idx] = controller.joint.joint_motor_signal
        #iCub_robot.iCub_set_angles(iCubMotor.MotorCommand)
        All_Command.append(iCubMotor.MotorCommand[:])
        All_Joints_Sensor.append(current_angles)


    mc_a = np.array(iCubMotor.MotorCommand[:])
    final_pos = wrist_position_icub(mc_a[joints])[0:3]
    return final_pos

clip = lambda x, l, u: l if x < l else u if x > u else x
def random_goal2_iCub(initial_position):

    counter = 0
    nvd = 0
    goal = [0,0,0]
    #current_angles = np.copy(angles)
    current_angles = np.zeros(params.number_cpg)
    while(nvd<0.5): #(nvd<0.15): 0.15 or 0.5
        current_angles[iCubMotor.RShoulderPitch] = clip(angles[iCubMotor.RShoulderPitch] + np.random.normal(0,50), -95., 10.)
        current_angles[iCubMotor.RShoulderRoll] = clip(angles[iCubMotor.RShoulderRoll] + np.random.normal(0,80), 0., 160.8)
        current_angles[iCubMotor.RShoulderYaw] = clip(angles[iCubMotor.RShoulderYaw] + np.random.normal(0,30), -37., 80.)
        current_angles[iCubMotor.RElbow] =  clip(angles[iCubMotor.RElbow] + np.random.normal(20,40), 15.5, 106.)
        current_angles = np.radians(current_angles)
        goal = wrist_position_icub(current_angles[joints])[0:3]
        nvd = np.linalg.norm(goal-initial_position)
        #counter+=1

    if(counter<100):
        return goal
    else:
        return [0,0,0]

def random_goal_iCub(initial_position):

    counter = 0
    nvd = 0
    goal = [0,0,0]
    #current_angles = np.copy(angles)
    current_angles = np.zeros(params.number_cpg)
    while(nvd<0.5): #(nvd<0.15): 0.15 or 0.5
        current_angles[iCubMotor.RShoulderPitch] = np.random.uniform(-95., 10.)
        current_angles[iCubMotor.RShoulderRoll] = np.random.uniform(0., 160.8)
        current_angles[iCubMotor.RShoulderYaw] = np.random.uniform(-37., 80.)
        current_angles[iCubMotor.RElbow] =  np.random.uniform(15.5, 106.)
        current_angles = np.radians(current_angles)
        goal = wrist_position_icub(current_angles[joints])[0:3]
        nvd = np.linalg.norm(goal-initial_position)
        #counter+=1

    if(counter<100):
        return goal
    else:
        return [0,0,0]

def random_goal2(initial_position):

    counter = 0
    nvd = 0
    goal = [0,0,0]
    #current_angles = np.copy(angles)
    current_angles = np.zeros(params.number_cpg)
    while(nvd<0.5): #(nvd<0.15): 0.15 or 0.5
        current_angles[iCubMotor.LShoulderPitch] = angles[iCubMotor.LShoulderPitch] + np.random.normal(0,20) #np.random.uniform(40,80) #angles[iCubMotor.LShoulderPitch]>
        current_angles[iCubMotor.LShoulderRoll] = angles[iCubMotor.LShoulderRoll] + np.random.normal(0,20) #np.random.uniform(0,90) #np.random.uniform(0,90) #angles[iCu>
        current_angles[iCubMotor.LShoulderYaw] = angles[iCubMotor.LShoulderYaw] + np.random.normal(0,20) #np.random.uniform(-120,0) #np.random.uniform(-120,0) #angles[i>
        current_angles[iCubMotor.LElbow] =  angles[iCubMotor.LElbow] + np.random.normal(0,20) #np.random.uniform(-90,-10) #np.random.uniform(-90,-10) #angles[iCubMotor.>
        current_angles = np.radians(current_angles)
        goal = wrist_position(current_angles[joints])[0:3]
        nvd = np.linalg.norm(goal-initial_position)
        #counter+=1

    if(counter<100):
        return goal
    else:
        return [0,0,0]


def random_goal(initial_position):

    counter = 0
    nvd = 0
    goal = [0,0,0]
    #current_angles = np.copy(angles)
    current_angles = np.zeros(params.number_cpg)
    while(nvd<0.5 and counter<100):#(nvd<0.15): 0.15 or 0.5
        current_angles[iCubMotor.LShoulderPitch] = np.random.uniform(40,80) #angles[iCubMotor.LShoulderPitch] + np.random.normal(0,20)
        current_angles[iCubMotor.LShoulderRoll] = np.random.uniform(0,90) #angles[iCubMotor.LShoulderRoll] + np.random.normal(0,20)
        current_angles[iCubMotor.LShoulderYaw] = np.random.uniform(-120,0) #angles[iCubMotor.LShoulderYaw] + np.random.normal(0,20)
        current_angles[iCubMotor.LElbow] = np.random.uniform(-90,-10) #angles[iCubMotor.LElbow] + np.random.normal(0,20)
        current_angles = np.radians(current_angles)
        goal = wrist_position(current_angles[joints])[0:3]
        nvd = np.linalg.norm(goal-initial_position)
        counter+=1

    if(counter<100):
        return goal
    else:
        return [0,0,0]


def train_bg(nt):

    num_trials_test = 400


    error_history = np.zeros(num_trials_test+nt)


    counter = 0
    goals = np.zeros((nt,3))
    parameter_history = np.zeros((nt,4,6))
    distance_history = np.zeros(nt)

    for trial in range(num_trials_test+nt):
        print('trial '+str(trial))


        RG_Pat1.noise = 0.0
        RG_Pat2.noise = 0.0
        RG_Pat3.noise = 0.0
        RG_Pat4.noise = 0.0
        PF_Pat1.noise = 0.0
        PF_Pat2.noise = 0.0
        Inj_Curr.noise = 0.0
        RG_Pat1.trace = 0.0
        RG_Pat2.trace = 0.0
        RG_Pat3.trace = 0.0
        RG_Pat4.trace = 0.0
        PF_Pat1.trace = 0.0
        PF_Pat2.trace = 0.0
        Inj_Curr.trace = 0.0
        RG_Pat1.baseline = 0.0
        RG_Pat2.baseline = 0.0
        RG_Pat3.baseline = 0.0
        RG_Pat4.baseline = 0.0
        PF_Pat1.baseline = 0.0
        PF_Pat2.baseline = 0.0
        Inj_Curr.noise = 0.0
        RG_Pat1.r = 0.0
        RG_Pat2.r = 0.0
        RG_Pat3.r = 0.0
        RG_Pat4.r = 0.0
        PF_Pat1.r = 0.0
        PF_Pat2.r = 0.0
        Inj_Curr.r = 0.0
        Intermediate.baseline=0.0
        StrD1_putamen.baseline=0.0
        StrD1_putamen.r= 0.0
        PM.baseline = 0.0
        SNr_putamen.r = 1.1
        StrThal_putamen.r = 0.0
        VA_putamen.r = 0.0

        Cortical_input.baseline=0.0


        #inter trial
        simulate(700) #650


        goal = [0,0,0]
        while(np.array_equal(goal,[0,0,0])):

            current_angles = np.copy(angles)
            initial_position = wrist_position_icub(np.radians(current_angles[joints]))[0:3]
            goal = random_goal2_iCub(initial_position)


        neuron_update(Cortical_input,goal, 10.0, 0.5) #0.2,1 // 0.005,1.0 IN ORIGINAL
        simulate(200)


        ran_prim = 0
        if(np.max(PM.r)<0.05):
            ran_prim = np.random.randint(120)
            Intermediate[ran_prim].baseline = 1.0
            PM[ran_prim].baseline = 0.5
            simulate(150)
        else:
            ran_prim = np.argmax(PM.r)
            Intermediate[ran_prim].baseline = 1.0
            PM[ran_prim].baseline = 0.5
            simulate(150)
            if(counter<nt):
                goals[counter] = goal
                counter+=1


        pms = np.zeros((4,6))
        for j in range(4):
            RG1_joint = 5+parameter_readout(RG_Pat1[j,:],0,5)
            RG2_joint = 5+parameter_readout(RG_Pat2[j,:],0,5)
            RG3_joint =  0.001+parameter_readout(RG_Pat3[j,:],-4,4)
            RG4_joint = 5+parameter_readout(RG_Pat4[j,:],0,10)


            PF1_joint = 0.001+parameter_readout(PF_Pat1[j,:],0,2.0)
            PF2_joint = 0.001+parameter_readout(PF_Pat2[j,:],0,2.0)


            pms[j] = [RG1_joint,RG2_joint,RG3_joint,RG4_joint,PF1_joint,PF2_joint]



        #execute a movement
        final_pos = execute_movement(pms,current_angles)
        vel_final = final_pos-initial_position

        nvf = np.linalg.norm(vel_final)
        nn= 0.0

        Cortical_input.baseline=0.0
        neuron_update(Cortical_input, final_pos, 10.0, 0.5)
        simulate(100)
        print(final_pos)

        if(nvf>0.3): #(nvf>0.3):
            reward.baseline = 1.0
            SNc_put.firing = 1.0
            simulate(100)
            SNc_put.firing = 0.0
            reward.baseline=0.0


        Intermediate.baseline = 0
        Cortical_input.baseline=0.0

        error_history[trial] = np.linalg.norm(final_pos-goal)


    np.save('error_history_bg_reach_' + str(nt) + '_' + str(num_trials_test) + '.npy',error_history)

    return goals, parameter_history



def parameters_per_goal(goal):


    RG_Pat1.noise = 0.0
    RG_Pat2.noise = 0.0
    RG_Pat3.noise = 0.0
    RG_Pat4.noise = 0.0
    PF_Pat1.noise = 0.0
    PF_Pat2.noise = 0.0
    Inj_Curr.noise = 0.0
    RG_Pat1.trace = 0.0
    RG_Pat2.trace = 0.0
    RG_Pat3.trace = 0.0
    RG_Pat4.trace = 0.0
    PF_Pat1.trace = 0.0
    PF_Pat2.trace = 0.0
    Inj_Curr.trace = 0.0
    RG_Pat1.baseline = 0.0
    RG_Pat2.baseline = 0.0
    RG_Pat3.baseline = 0.0
    RG_Pat4.baseline = 0.0
    PF_Pat1.baseline = 0.0
    PF_Pat2.baseline = 0.0
    Inj_Curr.noise = 0.0
    RG_Pat1.r = 0.0
    RG_Pat2.r = 0.0
    RG_Pat3.r = 0.0
    RG_Pat4.r = 0.0
    PF_Pat1.r = 0.0
    PF_Pat2.r = 0.0
    Inj_Curr.r = 0.0
    Intermediate.baseline=0.0
    StrD1_putamen.baseline=0.0
    StrD1_putamen.r= 0.0
    PM.baseline = 0.0
    SNr_putamen.r = 1.1
    StrThal_putamen.r = 0.0
    VA_putamen.r = 0.0

    Cortical_input.baseline=0.0

    # inter-trial
    simulate(700)

    neuron_update(Cortical_input, goal, 10.0, 0.5) #0.2,1 // 0.005,1.0 IN ORIGINAL
    simulate(200)


    ran_prim = np.argmax(PM.r)
    Intermediate[ran_prim].baseline = 1.0
    #PM[ran_prim].baseline = 0.5
    simulate(150)
    #print(np.argmax(PM.r))
    print(str(ran_prim)+' '+str(goal))


    pms = np.zeros((4,6))
    for j in range(4):
            RG1_joint = 5+parameter_readout(RG_Pat1[j,:],0,5)
            RG2_joint = 5+parameter_readout(RG_Pat2[j,:],0,5)
            RG3_joint =  0.001+parameter_readout(RG_Pat3[j,:],-4,4)
            RG4_joint = 5+parameter_readout(RG_Pat4[j,:],0,10)


            PF1_joint = 0.001+parameter_readout(PF_Pat1[j,:],0,2.0)
            PF2_joint = 0.001+parameter_readout(PF_Pat2[j,:],0,2.0)


            pms[j] = [RG1_joint,RG2_joint,RG3_joint,RG4_joint,PF1_joint,PF2_joint]

    print(pms)
    return pms





