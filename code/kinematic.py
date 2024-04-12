#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for the paper:

Baladron, J., Vitay, J., Fietzek, T. and Hamker, F. H.
The contribution of the basal ganglia and cerebellum to motor learning: a neuro-computational approach.

Copyright the Authors (License MIT)

Definition of the kinematics module.
"""

import numpy as np
import math
import sys
# import icub

from ANN_iCub_Interface.iCub import iCub_Interface, Kinematic_Reader, Kinematic_Writer
import ANN_iCub_Interface.Vocabs as iCub_const
from scipy import linalg


def G(a, d, alph, theta):

    t1 = np.array([ [np.cos(theta), - np.sin(theta), 0,  0],[np.sin(theta), np.cos(theta), 0,  0],[0, 0, 1, 0],[0, 0, 0, 1]])
    bt = np.c_[ np.identity(3), np.array([0,0,d])]
    t2 = np.vstack([bt,[0,0,0,1]])

    ct =  np.c_[ np.identity(3), np.array([a,0,0])]
    t3 = np.vstack([ct,[0,0,0,1]])

    t4 = np.array([[1, 0, 0, 0],[ 0, np.cos(alph), -np.sin(alph), 0],[ 0, np.sin(alph), np.cos(alph), 0],[ 0, 0, 0, 1]])

    return np.matmul(np.matmul(np.matmul(t1,t2),t3),t4)

def wrist_position(a):
    L1 = 0.05
    L2 = 0.22
    L3 = 0.16

    theta2 = a[0] #np.radians(a1)#Shoulder yaw
    theta1 = a[1] #np.radians(a2)#Shoulder pitch
    theta3 = a[2] #np.radians(a3)#Shoulder roll
    theta4 = a[3] #np.radians(a4)#Elbow

    G_34 = G(L3, 0, 0, np.pi/2+theta4)
    G_23 = G(0, L2, np.pi/2, theta3+np.pi/2)
    G_12 = G(L1, 0, -np.pi/2, theta2+np.pi/2)
    G_01 = G(0, 0, -np.pi/2, theta1)

    G_02 = np.matmul(G_01,G_12)
    G_03 = np.matmul(G_02,G_23)
    G_04 = np.matmul(G_03,G_34)
    return G_04.dot(np.array([0, 0, 0, 1]).T)

def wrist_position2(a1,a2,a3,a4):
    L1 = 0.05
    L2 = 0.22
    L3 = 0.16

    theta2 = np.radians(a1)#Shoulder yaw
    theta1 = np.radians(a2)#Shoulder pitch
    theta3 = np.radians(a3)#Shoulder roll
    theta4 = np.radians(a4)#Elbow

    G_45 = G(-L3, 0, 0, np.pi/2+theta4)
    G_34 = G(0, L2, -np.pi/2, theta3+np.pi/2)
    G_23 = G(0, 0, np.pi/2, theta2/2+3*np.pi/4)
    G_12 = G(L1, 0, 0, theta2/2+3*np.pi/4)
    G_01 = G(0, 0, -np.pi/2, theta1+np.pi)

    G_02 = np.matmul(G_01, G_12)
    G_03 = np.matmul(G_02, G_23)
    G_04 = np.matmul(G_03, G_34)
    G_05 = np.matmul(G_04, G_45)
    return G_04.dot(np.array([0, 0, 0, 1]).T)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


## iCub kinematic

def yarpvec_2_npvec(yarp_vec):
    '''
        convert a YARP vector into a 1D numpy array

        params: yarp_vec    -- 1D YARP vector

        return: vector      -- 1D Numpy array, result of conversion
    '''
    vector = np.zeros(yarp_vec.length(), dtype=np.float64)

    for i in range(yarp_vec.length()):
        vector[i] = yarp_vec.get(i)

    return vector


iCub = iCub_Interface.ANNiCub_wrapper()
kin_write = Kinematic_Writer.PyKinematicWriter()
kin_read = Kinematic_Reader.PyKinematicReader()

if not kin_write.init(iCub, "invkin", part=iCub_const.PART_KEY_RIGHT_ARM, version=2, ini_path="./", offline_mode=True):
    sys.exit("Initialization failed")

if not kin_read.init(iCub, "fwkin", part=iCub_const.PART_KEY_RIGHT_ARM, version=2, ini_path="./", offline_mode=True):
    sys.exit("Initialization failed")

initangles = np.zeros(10)
initangles[3] = 10
initangles[4] = 15.
initangles[6] = 15.

blocked_joints = [0, 1, 2, 7, 8, 9]

kin_read.set_jointangles(np.deg2rad(initangles))
kin_write.set_jointangles(np.deg2rad(initangles))

kin_read.block_links(blocked_joints)
kin_write.block_links(blocked_joints)


def wrist_position_icub(angles):
    kin_read.set_jointangles(angles)
    return kin_read.get_handposition()

def check_joint_position_icub(goal):
    kin_read.set_jointangles(kin_write.solve_InvKin(goal))
    kin_goal = kin_read.get_handposition()
    if np.allclose(goal, kin_goal, atol=0.05):
        return True
    else:
        return False


print(linalg.norm( - wrist_position_icub(initangles)))