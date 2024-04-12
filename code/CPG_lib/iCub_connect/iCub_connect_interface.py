"""

@author: Torsten Fietzek

Class to control the iCub joints
"""

import os

import numpy as np
from ANN_iCub_Interface.iCub import iCub_Interface


class iCub():
    """
        class handling the iCub control for the CPG
    """

    _parts_in_use = []
    _num_joints = 0
    _JReader = {}
    _JWriter = {}

    def __init__(self, parts, icub_wrapper=None):
        self._sequence = {"head": 6, "torso": 3, "right_arm": 16, "right_leg": 6, "left_arm": 16, "left_leg": 6}

        if icub_wrapper == None:
            self._iCub_wrap = iCub_Interface.ANNiCub_wrapper()
        else:
            self._iCub_wrap = icub_wrapper

        file_path = os.path.dirname(os.path.abspath(__file__))
        # print(os.path.abspath(__file__))
        # print(file_path + "/CPG_robot.xml")
        ret, name_dict = self._iCub_wrap.init_robot_from_file(file_path + "/../supportive/CPG_robot.xml")
        if not ret:
            print("Error while interface initialization!")

        for key in self._sequence:
            if parts.count(key) > 0:
                self._parts_in_use.append(key)
                self._num_joints += self._sequence[key]
                self._JReader[key] = self._iCub_wrap.get_jreader_by_part(key)
                self._JWriter[key] = self._iCub_wrap.get_jwriter_by_part(key)

    def __del__(self):
        self._iCub_wrap.clear()


    def iCub_get_angles(self):
        """
            Return robot joint angles in radians -> read joint encoders
        """

        angles = np.empty(0)
        for part in self._parts_in_use:
            angles = np.append(angles, self._JReader[part].read_double_all())
        return np.round(np.radians(angles), 4)


    def iCub_set_angles(self, motor_command):
        """
            Set robot angles -> move joints

            motor_command -> target angles in radians
        """

        if len(motor_command) != self._num_joints:
            print("[Error] Joint count in motor_command not matches used joints!")
            return False
        current_pos = self.iCub_get_angles()
        start = 0
        for part in self._parts_in_use:
            idx = start + self._sequence[part]
            position = np.degrees(motor_command[start:idx])
            if not np.allclose(current_pos[start:idx], motor_command[start:idx], atol=0.01):
                self._JWriter[part].write_double_all(position, mode="abs", blocking=True)
            start = idx
        return True

