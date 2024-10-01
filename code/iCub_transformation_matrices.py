"""
Created on Mon May 04 2020

@author: Torsten Fietzek

Transformation matrices for the iCub in Gazebo or the iCub-Simulator.
"""

import numpy as np

######################################################################
# Simulation parameter
gazebo_sim = True

######################################################################
# Transformation matrices
######################################################################

# iCub_Sim coordinate system <-> gazebo coordinate system
Transfermat_gazebo2iCubSim = np.array([[0.,  1., 0.,  0.0000],
                                       [0.,  0., 1.,  0.0000],
                                       [1.,  0., 0.,  0.0000],
                                       [0.,  0., 0.,  1.0000]])
Transfermat_iCubSim2gazebo = np.linalg.inv(Transfermat_gazebo2iCubSim)

# robot coordinate system <-> gazebo coordinate system
Transfermat_robot2gazebo = np.array([[-1.,  0., 0., 0.0000],
                                     [0., -1., 0., 0.0000],
                                     [0., 0., 1., 0.6000],
                                     [0., 0., 0., 1.0000]])
Transfermat_gazebo2robot = np.linalg.inv(Transfermat_robot2gazebo)

# robot coordinate system <-> iCub_Sim coordinate system
Transfermat_robot2iCubSim = np.array([[0., -1., 0., 0.0000],
                                      [0., 0., 1., 0.5976],
                                      [-1., 0., 0., -0.0260],
                                      [0., 0., 0., 1.0000]])
Transfermat_iCubSim2robot = np.linalg.inv(Transfermat_robot2iCubSim)


# Set world to robot transformation based on internal state
if gazebo_sim:
    Transfermat_robot2world = Transfermat_robot2gazebo
    Transfermat_world2robot = Transfermat_gazebo2robot
else:
    Transfermat_robot2world = Transfermat_robot2iCubSim
    Transfermat_world2robot = Transfermat_iCubSim2robot


######################################################################
# Initial hand position and orientation
pos_hand_world_coord = np.array([-0.15, 0.8, 0.2, 1.0])   # in world coordinate system
init_hand_pos_robot_rf = np.dot(Transfermat_iCubSim2robot, pos_hand_world_coord.reshape((4, 1)))     # in robot coordinate system
orientation_robot_hand = np.array([0.0022, -0.993, -0.1178, 2.0423])  # in robot coordinate system


######################################################################
# Transformation of a 3D-Cartesian position from source to target ref. frame
def transform_position(pos, mat):
    """
    Transform the given position with the given transformation matrix, from the source reference frame to the target reference frame.

    Parameters
    ----------
    pos : list/NDarray
        3D-cartesian position, which need to be transformed (could also the 1 as 4th dimension value)
    mat : NDarray
        Transformation matrx from source ref. fram to target ref. frame.

    Returns
    -------
    NDarray
        Position in the target ref. frame.
    """
    pos = np.array(pos)
    if pos.shape[0] == 3:
        shape = (1,)
        if len(pos.shape) == 2:
            if pos.shape[1] > 0:
                shape = (1, pos.shape[1])
        pos = np.concatenate((pos, np.array([1.]).reshape(shape)))
        pos = pos.reshape((4, 1))
        return np.dot(mat, pos)[0:3, 0]
    elif pos.shape[0] == 4:
        pos = pos.reshape((4, 1))
        return np.dot(mat, pos)[0:3, 0]
    else:
        print("Check position vector dimension!", pos.shape)


######################################################################
# Set either gazebo or iCub_Sim as world and return the transformation
# matrices for world to robot and robot to world
def retrieve_world_transform(use_gazebo=gazebo_sim):
    """
    Returns the transformation matrices for robot to world and world to robot transformations either for gazebo or iCub-simulator.

    Parameters
    ----------
    use_gazebo : bool, optional
        Set to True, if gazebo is used and False if iCub-simulator is used, by default gazebo_sim

    Returns
    -------
    Tuple[NDarray, NDarray]
        Returns the transformation matrices as tuple in the form: (robot2world, world2robot)
    """
    if use_gazebo:
        Transfermat_robot2world = Transfermat_robot2gazebo
        Transfermat_world2robot = Transfermat_gazebo2robot
    else:
        Transfermat_robot2world = Transfermat_robot2iCubSim
        Transfermat_world2robot = Transfermat_iCubSim2robot
    return Transfermat_robot2world, Transfermat_world2robot
