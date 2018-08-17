import sys
import argparse
from math import pi, sin, cos

import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import Point3, NodePath
from panda3d.core import AmbientLight, DirectionalLight, PointLight
from panda3d.core import loadPrcFileData

loadPrcFileData('', 'win-size 50 50')

class MyApp(ShowBase):
    def __init__(self, start_pos, finish_pos=4., rot_vel=1., direction=0.):
        ShowBase.__init__(self)

        # Disable the camera trackball controls.
        self.disableMouse()
        self.accept("escape", sys.exit)

        # This empty node is this node that we rotate and position
        # etc. in order to position the object.

        # The object is added as a child node. Movement of the child
        # is relative to this (its parents) which allows the center of
        # rotation to be changed.

        obj_handle = NodePath("empty")
        obj_handle.reparentTo(self.render)


        # This model came from the Sketch Up website IIRC.
        self.obj = self.loader.loadModel('./cube.egg')
        self.obj.reparentTo(obj_handle)

        self.obj.setPos(-0.5,-0.5,-0.5) # Adjust position so that rotation is about the center.
        self.obj.setScale(1./173) # This can from inspecting the egg file.



        # The object moves along, and rotates around, the x axis.
        obj_handle.setPos(start_pos, 0, 0)

        obj_handle.posInterval(2, (finish_pos, 0, 0)).loop()

        obj_handle.hprInterval(3./rot_vel, (0, 360, 0)).loop()

        # We the position the camera so that the object is moving
        # towards to camera, slightly off axis, but still looking at
        # the origin. (`radius` controls by how much the camera is off
        # axis.)

        radius = 6
        theta_deg = direction
        theta = (theta_deg / 360.0) * 2 * pi
        x = cos(theta) * radius
        y = sin(theta) * radius


        camX = 8

        self.camera.setPos(camX,x,y)
        self.camera.lookAt(0,0,0)

        # The lighting also coincides with the camera. This gives the
        # effect of a fixed camera/lighting, and a varying object
        # path.

        dlight = DirectionalLight('light')
        dlnp = render.attachNewNode(dlight)
        self.render.setLight(dlnp)

        dlnp.setPos(camX,x,y)
        dlnp.lookAt(0,0,0)


    def exit_after(self, duration):
        taskMgr.add(lambda task: exitTask(task, duration), 'exitTask')

        
def exitTask(task, duration):
    if (task.frame > 30 * duration):
        sys.exit()
    else:
        return task.cont


def main(out_path):
    start_pos = np.random.uniform(low=-12.0, high=-8.0)
    finish_pos = np.random.uniform(low=0.8, high=4.0)
    rot_vel = np.random.uniform(low=0.33, high=1.0)
    direction = np.random.uniform(0, 366)

    #print((start_pos, finish_pos, rot_vel, direction))
    app = MyApp(start_pos, finish_pos, rot_vel, direction)

    # http://www.panda3d.org/reference/1.9.4/python/direct.showbase.ShowBase.ShowBase#aaa33db419e3252da60a92e824d88b7f0
    app.movie(namePrefix='{}/frame'.format(out_path), format='png', duration=2.0, fps=10)
    app.exit_after(2)

    app.run()


# These can be turned into movies with something like this:
# ffmpeg -framerate 10 -i movie/frame_%4d.png -r 25 out.mpg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str)
    args = parser.parse_args()
    assert args.o is not None, 'argument error'
    print('cube args: {}'.format(args))
    main(args.o)
