"""
    @author: Torsten Fietzek

    Class to control the environment in the gazebo simulator, for example simple object manipulation.
    The basis is given for object manipulation in: http://www.icub.org/software_documentation/icub_python_simworld_control.html (Marek Rucinski).
    This is adapted to the gazebo simulator.
"""


import numpy as np
import yarp

yarp.Network.init()  # Initialise YARP

####################################################
"""
    TODO:
        attach
        detach
"""

########################################################


class WorldController:
    """Class for controlling iCub simulator via its RPC world port."""

    def __init__(self):
        self._rpc_client = yarp.RpcClient()
        self._port_name = "/WorldController_" + str(id(self)) + "/commands"
        self._rpc_client.open(self._port_name)
        self._rpc_client.addOutput("/world_input_port")

    def _execute(self, cmd):
        """Execute an RPC command, returning obtained answer bottle."""
        ans = yarp.Bottle()
        self._rpc_client.write(cmd, ans)
        return ans

    def _is_success(self, ans):
        """Check if RPC call answer Bottle indicates successfull execution."""
        return ans.size() == 1 and ans.get(0).asVocab32() == 27503  # Vocab for '[ok]'

    def _prepare_del_all_command(self):
        """Prepare the "world del all" command bottle."""
        result = yarp.Bottle()
        result.clear()
        list(map(result.addString, ["deleteAll"]))
        return result

    def del_all(self):
        """Delete all objects and models from the simulator"""
        result = self._is_success(self._execute(
            self._prepare_del_all_command()))

        return result

########################################################
# object creation and manipulation
########################################################

    ########################################################
    # create object inside the simulator
    def _prepare_create_obj_command(self, obj_func, size, location, orientation, color):
        """
            Prepare an RPC command for creating an object in the gazebo simulator environment.

            See Simulator Readme section 'Object Creation'

            Parameters:
                obj_func    -- function name for the creation of the different objects: 'sphere', 'box', 'cylinder'.
                size        -- list of values specifying the size of an object. Parameters depend on object type:
                                (s)box: [ x, y, z ]
                                (s)sph: [ radius ]
                                (s)cyl: [ radius, length ]
                location    -- coordinates of the object location, [ x, y, z ]
                orientation -- object rotation in rad
                color       -- object color in RGB (values: [0...255]), [ r, g, b ]

            Returns:
                yarp.Bottle with the command, ready to be sent to the rpc port of the simulator
        """
        yarp_cmd = yarp.Bottle()
        yarp_cmd.clear()
        yarp_cmd.addString(obj_func)
        list(map(yarp_cmd.addFloat64, size))
        list(map(yarp_cmd.addFloat64, location))
        list(map(yarp_cmd.addFloat64, orientation))
        list(map(yarp_cmd.addInt32, color))
        return yarp_cmd

    def create_object(self, obj, size, location, orientation, color):
        """
            Create an object of a specified type, size, location, orientation and color, returning internal object ID or -1 on error.

            Parameters:
                obj         -- object type string: 'sphere', 'box', 'cylinder', frame.
                size        -- list of values specifying the size of an object. Parameters depend on object type:
                                box: [ x, y, z ]
                                sphere: [ radius ]
                                cylinder: [ radius, length ]
                                frame: [ scale ]
                location    -- coordinates of the object location, [ x, y, z ]
                orientation -- object rotation in rad
                color       -- object color in RGB (values: [0...255]), [ r, g, b ]

            Returns:
                object name as string; returns "" at error
        """
        if obj == "sphere":
            obj_func = "makeSphere"
        elif obj == "box":
            obj_func = "makeBox"
        elif obj == "cylinder":
            obj_func = "makeCylinder"
        elif obj == "frame":
            obj_func = "makeFrame"
        else:
            print("Error: No correct object identifier given!")
            return -1

        cmd = self._prepare_create_obj_command(
            obj_func, size, location, orientation, color)
        ans = self._execute(cmd)
        if ans.toString() == "[fail]":
            print('Error: Object creation failed!')
            return None  # error

        return ans.toString()

    ########################################################
    # set object pose inside the simulator
    def _prepare_set_pose_command_obj(self, obj_id, location, rotation):
        """
            Prepare the "world set <obj> <xyz>" command bottle.

            Parameters:
                obj         -- object descriptor string
                obj_id      -- simulator object ID
                location    -- coordinates of the object location, [ x, y, z ]
                orientation -- object rotation in rad

            Returns:
                yarp.Bottle with the command, ready to be sent to the rpc port of the simulator
        """
        yarp_cmd = yarp.Bottle()
        yarp_cmd.clear()
        yarp_cmd.addString("setPose")
        yarp_cmd.addString(obj_id)
        list(map(yarp_cmd.addFloat64, location))
        list(map(yarp_cmd.addFloat64, rotation))
        return yarp_cmd

    def set_pose(self, obj_id, location, rotation):
        """
            Set a new pose (location and rotation) for the given object.

            Parameters:
                obj_id      -- internal model ID to reference the object
                location    -- coordinates of the object location, [ x, y, z ]
                orientation -- object rotation in rad

            Returns:
                True/False dependent on success/failure.
        """
        return self._is_success(self._execute(self._prepare_set_pose_command_obj(obj_id, location, rotation)))

    ########################################################
    # get object pose inside the simulator
    def _prepare_get_pose_command_obj(self, obj_id):
        """
            Prepare the "world get <obj> <id>" command bottle.

            Parameters:
                obj         -- object descriptor string
                obj_id      -- simulator object ID

            Returns:
                yarp.Bottle with the command, ready to be sent to the rpc port of the simulator
        """
        result = yarp.Bottle()
        result.clear()
        list(map(result.addString, ["getPose", obj_id]))
        return result

    def get_pose(self, obj_id):
        """
            Obtain the object location from the simulator.

            Parameters:
                obj_id      -- internal object ID

            Returns:
                object location coordinates [x, y, z]; None on failure.

        """
        result = self._execute(self._prepare_get_pose_command_obj(obj_id))
        if result.size() == 6:
            # 6-element list with xyz coordinates
            return np.array([result.get(i).asDouble() for i in range(6)])

        return None  # An error occured

    ########################################################
    # set object name inside the simulator
    def _prepare_set_name_command_obj(self, obj_id, new_name):
        """
            Prepare the "world set <obj> <xyz>" command bottle.

            Parameters:
                obj_id      -- simulator object ID
                new_name    --

            Returns:
                yarp.Bottle with the command, ready to be sent to the rpc port of the simulator
        """
        yarp_cmd = yarp.Bottle()
        yarp_cmd.clear()
        list(map(yarp_cmd.addString, ["rename", obj_id, new_name]))
        return yarp_cmd

    def set_name(self, obj_id, new_name):
        """
            !!! not fully implemented in gazebo !!!
            Set a new name for the given object.

            Parameters:
                obj_id      -- internal model ID to reference the object
                new_name    -- new model name

            Returns:
                True/False dependent on success/failure.
        """
        result = self._is_success(self._execute(
            self._prepare_set_name_command_obj(obj_id, new_name)))
        return result, new_name

    ########################################################
    # change object color inside the simulator
    def _prepare_change_color_command_obj(self, obj_id, new_color):
        """
            Prepare the "world set <obj> <xyz>" command bottle.

            Parameters:
                obj_id      -- simulator object ID
                new_name    --

            Returns:
                yarp.Bottle with the command, ready to be sent to the rpc port of the simulator
        """
        yarp_cmd = yarp.Bottle()
        yarp_cmd.clear()
        list(map(yarp_cmd.addString, ["changeColor", obj_id]))
        list(map(yarp_cmd.addInt32, new_color))

        return yarp_cmd

    def change_color(self, obj_id, new_color):
        """
            !!! not fully implemented in plugins!!!
            Set a new color for the given object.

            Parameters:
                obj_id      -- internal model ID to reference the object
                new_color   --

            Returns:
                True/False dependent on success/failure.
        """
        return self._is_success(self._execute(self._prepare_change_color_command_obj(obj_id, new_color)))

    ########################################################
    # enable gravity for the given obejct
    def _prepare_enable_gravity_command_obj(self, obj_id, enable):
        """
            Prepare the "world set <obj> <xyz>" command bottle.

            Parameters:
                obj_id      -- internal model ID to reference the object
                enable      -- Integer -> >0 enable gravity; <=0 disable gravity

            Returns:
                yarp.Bottle with the command, ready to be sent to the rpc port of the simulator
        """
        yarp_cmd = yarp.Bottle()
        yarp_cmd.clear()
        yarp_cmd.addString("enableGravity")
        yarp_cmd.addString(obj_id)
        yarp_cmd.addInt32(enable)
        return yarp_cmd

    def enable_gravity(self, obj_id, enable):
        """
            Enable gravity for the given object.

            Parameters:
                obj_id      -- internal model ID to reference the object
                enable      -- Integer -> >0 enable gravity; <=0 disable gravity

            Returns:
                True/False dependent on success/failure.
        """
        return self._is_success(self._execute(self._prepare_enable_gravity_command_obj(obj_id, enable)))

    ########################################################
    # enable collision for the given obejct
    def _prepare_enable_collision_command_obj(self, obj_id, enable):
        """
            Prepare the command bottle.

            Parameters:
                obj_id      -- internal model ID to reference the object
                enable      -- Integer -> >0 enable collision; <=0 disable collision

            Returns:
                yarp.Bottle with the command, ready to be sent to the rpc port of the simulator
        """
        yarp_cmd = yarp.Bottle()
        yarp_cmd.clear()
        yarp_cmd.addString("enableCollision")
        yarp_cmd.addString(obj_id)
        yarp_cmd.addInt32(enable)
        return yarp_cmd

    def enable_collision(self, obj_id, enable):
        """
            Enable collision for the given object.

            Parameters:
                obj_id      -- internal model ID to reference the object\n
                enable      -- Integer -> >0 enable collision; <=0 disable collision

            Returns:
                True/False dependent on success/failure.
        """
        return self._is_success(self._execute(self._prepare_enable_collision_command_obj(obj_id, enable)))

    ########################################################
    # get list with object names
    def _prepare_get_list_command_obj(self):
        """
            Prepare the command bottle.

            Parameters:

            Returns:
                yarp.Bottle with the command, ready to be sent to the rpc port of the simulator
        """
        result = yarp.Bottle()
        result.clear()
        result.addString("getList")
        return result

    def get_list(self):
        """
            Return a list with the names of the objects created inside the simulator.

            Parameters:

            Returns:
                list with strings of object names; None on failure.

        """
        result = self._execute(self._prepare_get_list_command_obj())
        if result.size() < 1 or result.get(0).toString() == "[fail]":
            return None  # An error occured

        return result.pop().toString().split(' ')

    ########################################################
    # delete a single Object in the simulator
    def _prepare_del_object_command_obj(self, obj_id):
        """
            Prepare the "world get <obj> <id>" command bottle.

            Parameters:
                obj         -- object descriptor string
                obj_id      -- simulator object ID

            Returns:
                yarp.Bottle with the command, ready to be sent to the rpc port of the simulator
        """
        result = yarp.Bottle()
        result.clear()
        list(map(result.addString, ["deleteObject", obj_id]))
        return result

    def del_object(self, obj_id):
        """
            Obtain the object location from the simulator.

            Parameters:
                obj_id      -- internal object ID

            Returns:
                True/False dependent on success/failure.

        """
        result = self._is_success(self._execute(
            self._prepare_del_object_command_obj(obj_id)))
        return result

    ########################################################
    # load 3D model into the simulator
    def _prepare_create_model_command_model(self, filename, location, orientation):
        """
            Prepare an RPC command for importing a model in the simulator environment.

            Parameters:
                filename    -- absolute path to the model
                location    -- coordinates of the object location, [ x, y, z ]
                orientation -- object rotation in rad

            Returns:
                yarp.Bottle with the command, ready to be sent to the rpc port of the simulator
        """
        result = yarp.Bottle()
        result.clear()
        result.addString("loadModelFromFileWithPose")
        result.addString(filename)
        list(map(result.addFloat64, location))
        list(map(result.addFloat64, orientation))
        return result

    def create_model(self, filename, location, orientation):
        """
            Import a model from a sdf file.

            Parameters:
                filename    -- absolute path to the model
                location    -- coordinates of the object location, [ x, y, z ]
                orientation -- object rotation in rad

            Returns:
                string as internal model ID to reference the model. In case of an error an empty string is returned
        """
        cmd = self._prepare_create_model_command_model(filename, location, orientation)
        ans = self._execute(cmd)
        if ans.toString() == "[fail]":
            print('error')
            return ""  # error

        return ans.toString()

    ########################################################
    # call destructor
    def __del__(self):
        try:
            if self._rpc_client is not None:
                self.del_all()
            self._rpc_client.close()
            del self._rpc_client
        except AttributeError:
            pass
