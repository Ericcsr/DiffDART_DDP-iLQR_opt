import numpy as np
from envs.diffdart_env import DiffDartEnv
import nimblephysics as dart
import torch
from .utils import ComputeCostGrad

class CustomDroneEnv(DiffDartEnv):
    def __init__(self,FD=False):
        frame_skip = 1
        DiffDartEnv.__init__(self,None,frame_skip,dt=0.01,FD=FD)
        # Create a world calling same name
        self.dart_world = dart.simulation.World()
        self.dart_world.setGravity([0, -9.81, 0])
        droneSkel = dart.dynamics.Skeleton()

        joint, drone = droneSkel.createPrismaticJointAndBodyNodePair()
        drone.setMass(2.0)
        joint.setAxis([0, 1, 0])
        droneShape = drone.createShapeNode(dart.dynamics.BoxShape([1.0, 1.0, 1.0]))
        droneShape.createCollisionAspect()
        droneVisual = droneShape.createVisualAspect()
        droneVisual.setColor([0.5, 0.5, 0.5])
        joint.setPositionUpperLimit(0, 10)
        joint.setPositionLowerLimit(0, -10)
        joint.setControlForceUpperLimit(0 ,20)
        joint.setControlForceLowerLimit(0 ,-20)

        self.dart_world.addSkeleton(droneSkel)

        floorSkel = dart.dynamics.Skeleton()
        floorJoint, floor = floorSkel.createWeldJointAndBodyNodePair()
        floorOffset = dart.math.Isometry3()
        floorOffset.set_translation([0, -0.55, 0])
        floorJoint.setTransformFromParentBodyNode(floorOffset)
        floorShape = floor.createShapeNode(dart.dynamics.BoxShape([10.0, 0.1, 10.0]))
        floorShape.createCollisionAspect()
        floorVisual = floorShape.createVisualAspect()
        floorVisual.setColor([0.2, 0.2, 0.2])
        self.dart_world.addSkeleton(floorSkel)

        self.dart_world.setTimeStep(self.timestep)
        self.robot_skeleton = droneSkel
        self.control_dofs = np.array([0])

    def running_cost(self, x, u, compute_grads = False):
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor(u, requires_grad=True)
        x_target = torch.FloatTensor([1.0, 0.])
        coeff = torch.FloatTensor([0.5, 0.5,])

        run_cost = torch.sum(0.01 * torch.mul(u,u))
        
        run_cost += torch.sum(torch.mul(coeff, torch.mul(x - x_target, x - x_target)))

        if compute_grads:
            run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu = ComputeCostGrad(run_cost, x, u=u)
            return run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu
        else:
            return run_cost.detach().numpy()

    def terminal_cost(self, x, compute_grads = False):
        x = torch.tensor(x, requires_grad=True)

        x_target = torch.FloatTensor([1.0, 0.])
        coeff = torch.FloatTensor([50, 50])
        ter_cost = torch.sum(torch.mul(coeff, torch.mul(x - x_target, x - x_target)))

        if compute_grads:
            ter_cost, grad_x, Hess_xx = ComputeCostGrad(ter_cost, x)
            return ter_cost, grad_x, Hess_xx
        else:
            return ter_cost.detach().numpy()
    
    def run_cost(self):
        return self.running_cost

    def ter_cost(self):
        return self.terminal_cost



