import numpy as np
from envs.diffdart_env import DiffDartEnv
import nimblephysics as dart
import torch
from .utils import ComputeCostGrad
import os

class AtlasEnv(DiffDartEnv):
    def __init__(self, FD = False):
        frame_skip = 1
        DiffDartEnv.__init__(self,None, frame_skip, dt = 0.002, FD=FD)
        self.dart_world = dart.simulation.World()
        self.dart_world.setGravity([0, -9.81, 0])
        atlas = self.dart_world.loadSkeleton(os.path.join(os.path.dirname(__file__), 
                "./assets/atlas/atlas_v3_no_head.urdf"))
        atlas.setPosition(0, -0.5 * 3.14159)

        ground = self.dart_world.loadSkeleton(os.path.join(os.path.dirname(__file__),
                 "./assets/atlas/ground.urdf"))
        floorBody = ground.getBodyNode(0)
        floorBody.getShapeNode(0).getVisualAspect().setCastShadows(False)
        self.state_dofs = np.arange(atlas.getNumDofs())
        self.control_dofs = np.arange(6,atlas.getNumDofs())
        forceLimits = np.ones(atlas.getNumDofs()) * 500
        forceLimits[0:6] = 0
        atlas.setControlForceUpperLimits(forceLimits)
        atlas.setControlForceLowerLimits(forceLimits * -1)
        self.target = np.array([0.0, 0.8 -1.0])
        self.target_mask = np.array([0, 1, 2])
        # Add IK mapping for loss computation
        # TODO: IK Currently cannot be handled properly
        self.ikmap = dart.neural.IKMapping(self.dart_world)
        self.ikmap.addLinearBodyNode(atlas.getBodyNode('l_hand'))
        self.ikmap_name = 'ik'
        self.robot = atlas

    def running_cost(self, x, u, compute_grads = False):
        pass

    def terminal_cost(self, x, compute_grads = False):
        pass

    def run_cost(self):
        return self.running_cost

    def ter_cost(self):
        return self.terminal_cost
        
    

