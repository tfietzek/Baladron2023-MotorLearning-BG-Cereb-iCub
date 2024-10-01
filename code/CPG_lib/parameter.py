"""
Created on Wed July 22 2020

@author Torsten Fietzek

parameterfile for the CPG with the iCub robot
-> handles the creation of the config files

Set used_parts, path_prefix and filename_postfix denpendent on the setup
"""
from pathlib import Path

from .supportive.create_motor_config_files import create_conf_files, create_interface_xml_file, create_interface_ini_file, create_icub_connect

# parts from the iCub to be controlled by the CPG
# possible parts are: ["head", "torso", "right_arm", "right_leg", "left_arm", "left_leg"]
used_parts = ["right_arm"]

# absolute path parameter file -> default behavior works with CPG_lib as subdirectory beside the main file, e.g. iCub_drawin.py
path_prefix = str(Path(__file__).resolve().parents[0]) + "/"

# filename postfix for the autogenerated configuration files
filename_postfix = "_right_arm"

## type of iCub connection
connect = "interface" # pyyarp; interface
create_icub_connect(path_prefix, connect)

# grpc flag -> only for interface connection
grpc = False

# interface robot config
if connect == "interface":
    params_interface = {}
    params_interface["ini_path"] = path_prefix + "supportive/"
    params_interface["ip_address_server"] = "0.0.0.0"
    params_interface["ip_address_client"] = "0.0.0.0"
    params_interface["port_reader"] = 50100
    params_interface["port_writer"] = 50110
    params_interface["speed"] = 50.0
    xml_filename = create_interface_xml_file(used_parts, path_prefix, params_interface, grpc)
    params_interface["simulator"] = "false"                    # use simulator (true)/ real robot (false)
    params_interface["robot_port_prefix"] = "/icubSim"      # set robot name -> port prefix of iCub robot devices
    params_interface["client_port_prefix"] = "/CPG"      # set client name, to avoid address conflicts
    create_interface_ini_file(params_interface)


## automatically created parameter
number_cpg, my_iCub_limits, positive_angle_dir, iCub_joint_names = create_conf_files(used_parts, path_prefix, filename_postfix)
