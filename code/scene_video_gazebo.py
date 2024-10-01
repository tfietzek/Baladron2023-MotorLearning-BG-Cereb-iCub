"""

@author: Torsten Fietzek


"""

######################################################################
########################## Import modules  ###########################
######################################################################


import matplotlib.pylab as plt
import numpy as np
import yarp

yarp.Network.init()
if not yarp.Network.checkNetwork():
    print('[ERROR] Please try running yarp server')


class scene_cam():
    def __init__(self, prefix="client", cam_port="/gazebo/cam/world", shape=(720,480)):

        self._port = yarp.Port()
        self._port_name = f"/{prefix}/scene_{id(self)}"
        self._cam = cam_port
        if not self._port.open(self._port_name):
            print("[ERROR] Could not open scene camera port")
        if not yarp.Network.connect(self._cam, self._port_name):
            print(f"[ERROR] Could not connect {self._port_name} to {self._cam}")

        self._img_array = np.ones((shape[1], shape[0], 3), np.uint8)
        self._yarp_image = yarp.ImageRgb()
        self._yarp_image.resize(shape[0], shape[1])

        self._yarp_image.setExternal(self._img_array.data, self._img_array.shape[1], self._img_array.shape[0])

    def __del__(self):
        # disconnect the ports
        yarp.Network.disconnect(self._cam, self._port_name)
        # close the ports
        self._port.close()

    def read_image(self):
        self._port.read(self._yarp_image)
        self._port.read(self._yarp_image)

        if self._yarp_image.getRawImage().__int__() != self._img_array.__array_interface__['data'][0]:
            print("read() reallocated self._yarp_image!")

        return self._img_array.copy()


