"""


@author: Torsten Fietzek

This file creates automatically the iCub-based config files needed for the CPG
"""

import hashlib
import os
import pickle
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from ..supportive.joint_limits import joint_limits, joint_names
from ..supportive.joint_positive_dir import posdir_dict


# get checksum from filename
def generate_checksum(filename):
    file_hash = hashlib.blake2b()
    try:
        with open(filename, "rb") as f:
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)

        return file_hash.hexdigest()
    except:
        return 0

# create MyiCub file
def create_joint_limits(parts, path_prefix, name_postfix, j_names, j_limits):
    """
        create config file for joint limits
    """

    filepath = path_prefix + "/ICUBROBOT/Conf_Limits/"
    if not os.path.isdir(filepath):
        os.mkdir(os.path.abspath(filepath))
    filename = os.path.abspath(filepath) + "/MyiCub" + name_postfix + ".txt"

    names = []
    limits_min = []
    limits_max = []

    joint_count = 0
    for part in parts:
        for i in range(len(j_limits[part])//2):
            names.append(joint_names[part][i])
            limits_min.append(j_limits[part]['joint_' + str(i) + '_min'])
            limits_max.append(j_limits[part]['joint_' + str(i) + '_max'])

            joint_count += 1

    text = np.zeros(len(names), dtype=[('names', 'U25'), ('min', float), ('max', float)])
    text['names'] = names
    text['min'] = np.radians(limits_min)
    text['max'] = np.radians(limits_max)

    np.savetxt(filename, text, fmt="%s %.4f %.4f")
    return joint_count, filename

# create MyiCubPositiveAngle file
def create_pos_angle(parts, path_prefix, name_postfix, j_names, j_pos_dir):
    """
        create config file for joint positive direction
    """

    filepath = path_prefix + "/ICUBROBOT/Conf_PosDir/"
    if not os.path.isdir(filepath):
        os.mkdir(os.path.abspath(filepath))
    filename = os.path.abspath(filepath) + "/MyiCubPositiveAngle_E_or_F" + name_postfix + ".txt"

    names = []
    joint_count = 0
    for part in parts:
        for name in j_names[part]:
            names.append(name)
            joint_count += 1

    with open(filename, 'w') as f:
        for joint in names:
            f.write(joint + " " + posdir_dict[joint] + "\n")
    return joint_count, filename

# create iCubMotor
def create_joint_mapping(parts, path_prefix, name_postfix, j_names):
    """
        create config file for joint names
    """
    filepath = path_prefix + "/ICUBROBOT/JointNames/"
    filename = os.path.abspath(filepath) + "/iCubMotor" + name_postfix + ".py"
    names = []

    joint_count = 0
    for part in parts:
        for name in j_names[part]:
            names.append(name)
            joint_count += 1

    global_cmd = "global  "
    indx_list = ""

    for idx, name in enumerate(names):
        global_cmd += name
        indx_list  += '{:18s}'.format(name) + " = " + '{:2d}'.format(idx) + "\n"
        if idx < (len(names) - 1):
            global_cmd += ", "
        if idx%10 == 0 and idx > 0:
            global_cmd += "\\" "\n" "    "

    str_mtr_cmd = "global MotorCommand\n"
    str_set_mtr_cmd = "MotorCommand = [0 for x in range(" + str(len(names)) + ")]\n"

    with open(filename, 'w') as file:
        file.write("\n")
        file.write(global_cmd)
        file.write("\n\n")
        file.write(indx_list)
        file.write("\n")
        file.write(str_mtr_cmd)
        file.write(str_set_mtr_cmd)

    return joint_count, filename

# call all file creation methods
def create_conf_files(parts_used, path_prefix, filename_postfix=""):
    """
        create all config files with given parameters
    """
    # Create ordered list of used parts
    part_list = []
    sequence = {"head": 6, "torso": 3, "right_arm": 16, "right_leg": 6, "left_arm": 16, "left_leg": 6}
    for key in sequence:
        if parts_used.count(key) > 0:
            part_list.append(key)

    print("Search configuration files...")
    filepath = path_prefix + "/supportive/Check_Conf/"
    if not os.path.isdir(filepath):
        os.mkdir(os.path.abspath(filepath))

    name = "check_config" + filename_postfix +  ".pkl"
    pickle_filename = filepath + name

    if os.path.isfile(pickle_filename):
        print("Found existing config files.")
        with open(pickle_filename, "rb") as checkfile:
            content = pickle.load(checkfile)
            if 'part_list' in content:
                if part_list == content['part_list']:
                    files_equal = True
                    if 'check_limits' in content.keys():
                        if generate_checksum(content['file_limits']) != content['check_limits']:
                            files_equal = False
                        if generate_checksum(content['file_posdir']) != content['check_posdir']:
                            files_equal = False
                        if generate_checksum(content['file_motor']) != content['check_motor']:
                            files_equal = False
                    else:
                        files_equal = False
                    if files_equal:
                        print("Existing config files matches given configuration. Skipped file creation.")
                        return content['jcount_limits'], content['file_limits'], content['file_posdir'], content['module_motor']
                    else:
                        print("Existing config files are outdated! Start config file creation.")
                else:
                    print("Existing config files are outdated! Start config file creation.")
            else:
                print("Existing config files are outdated! Start config file creation.")
    else:
        print("Did not found existing config files.")

    print("Create configuration files...")
    jcount_limits, file_limits = create_joint_limits(part_list, path_prefix, filename_postfix, joint_names, joint_limits)
    jcount_posdir, file_posdir = create_pos_angle(part_list, path_prefix, filename_postfix, joint_names, posdir_dict)
    jcount_motor, file_motor = create_joint_mapping(part_list, path_prefix, filename_postfix, joint_names)

    CPG_lib = os.path.basename(os.path.normpath(path_prefix))
    idx = path_prefix.find(CPG_lib)
    tmp_string = file_motor.replace(path_prefix[:idx], "")
    tmp_string = tmp_string.replace("/", ".")
    module_motor = tmp_string.replace(".py", "")

    data_dict = {'part_list': part_list, 'jcount_limits': jcount_limits, 'file_limits': file_limits, 'file_posdir': file_posdir, 'file_motor': file_motor, 'module_motor': module_motor}
    data_dict['check_limits'] = generate_checksum(file_limits)
    data_dict['check_posdir'] = generate_checksum(file_posdir)
    data_dict['check_motor'] = generate_checksum(file_motor)

    if jcount_limits == jcount_posdir == jcount_motor:
        with open(pickle_filename, "wb") as checkfile:
            pickle.dump(data_dict, checkfile)

        print("File creation successful.")
        return jcount_limits, file_limits, file_posdir, module_motor
    else:
        print("ATTENTION!! Different joint count in the config files!")
        return 0, file_limits, file_posdir, module_motor

# create xml config file for ANNarchy iCub interface
def create_interface_xml_file(parts, path_prefix, params_interface, grpc):
    """
        create robot xml file for ANN-iCub-Interface
    """

    xml_file = path_prefix + "supportive/CPG_robot.xml"
    description="robot config for CPG \n    -> autogenerated file, changes will be overwritten by code execution"
    root = ET.Element("robot")
    root.append(ET.Comment(description))

    for part in parts:
        name_reader = "JR_" + part
        if grpc:
            args_jreader = {"part": part, 'sigma': 1., 'popsize': 5, 'ini_path': params_interface["ini_path"], 'ip_address': params_interface["ip_address_client"], 'port': params_interface["port_reader"]}
        else:
            args_jreader = {"part": part, 'sigma': 1., 'popsize': 5, 'ini_path': params_interface["ini_path"]}
        JReader = ET.SubElement(root, "JReader", attrib={"name": name_reader})
        for key, value in args_jreader.items():
            ET.SubElement(JReader, key).text = str(value)
        params_interface["port_reader"] += 1

        name_writer = "JW_" + part
        if grpc:
            args_jwriter0 = {'part': part, 'popsize': 5, 'speed': params_interface["speed"], 'ini_path': params_interface["ini_path"], 'ip_address': params_interface["ip_address_server"], 'port': params_interface["port_writer"], 'mode': 'abs', 'blocking': True, 'joints': []}
        else:
            args_jwriter0 = {'part': part, 'popsize': 5, 'speed': params_interface["speed"], 'ini_path': params_interface["ini_path"]}
        JWriter = ET.SubElement(root, "JWriter", attrib={"name": name_writer})
        for key, value in args_jwriter0.items():
            ET.SubElement(JWriter, key).text = str(value)
        params_interface["port_writer"] += 1

    XML_file_string = minidom.parseString(ET.tostring(root, 'utf-8')).toprettyxml(indent="    ")
    with open(xml_file, "w") as files :
        files.write(XML_file_string)

    return xml_file

# create ini config file for ANNarchy iCub interface
def create_interface_ini_file(params_interface):
    """Create ini-file for ANNarchy-iCub-Interface.

    Parameters
    ----------
    params_interface : dict
        dictionary containing parameter for the interface
    """
    ini_file = params_interface["ini_path"] + "/interface_param.ini"
    INI_file_string = "; configuration file for the interface, auto-generated from the values given in parameter.py\n\n[general]\n"

    INI_file_string += "simulator = " + params_interface["simulator"].lower() + "                ; use simulator (true)/ real robot (false)\n"
    INI_file_string += "robot_port_prefix = " + params_interface["robot_port_prefix"] + "    ; set to robot name\n"
    INI_file_string += "client_port_prefix = " + params_interface["client_port_prefix"] + "    ; set to client name, to avoid address conflicts\n"

    with open(ini_file, "w") as files :
        files.write(INI_file_string)

# create Python file handling the import of the iCUb connection method
def create_icub_connect(path_prefix, connect: str):
    """Create iCub_connect based on selected connection method

    Parameters
    ----------

    """
    
    connect_file = path_prefix + "iCub_connect/iCub_connect.py"

    if os.path.isfile(connect_file):
        from ..iCub_connect.iCub_connect import _connect_type
        if _connect_type == connect.lower():
            print("Skip creation of connect file, file already exists!")
            return
    
    if connect.lower()=="pyyarp":
        connection = "iCub_connect_py"
    elif connect.lower()=="interface":
        connection = "iCub_connect_interface"
    else:
        print("No valid connection variant given! Fallback to default -> PyYarp")

    iCub_connect_string = "# file to select the connection method (e.g. YARP-Python-bindings, ANN_iCub_interface) with the robot, auto-generated from the values given in parameter.py\n\n"
    iCub_connect_string += f"from .{connection} import iCub\n"
    iCub_connect_string += f"_connect_type = \"{connect.lower()}\"\n"

    with open(connect_file, "w") as files :
        files.write(iCub_connect_string)


if __name__ == "__main__":
    path_prefix = "../"
    filename_postfix = "_larm"
    # parts: ["head", "torso", "right_arm", "right_leg", "left_arm", "left_leg"]
    parts_used = ["left_arm"]

    create_conf_files(parts_used, path_prefix, filename_postfix)
