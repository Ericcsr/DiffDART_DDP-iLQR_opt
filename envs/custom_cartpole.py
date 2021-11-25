import numpy as np
from envs.diffdart_env import DiffDartEnv
import nimblephysics as dart
import torch
from .utils import ComputeCostGrad

class CustomCartPoleEnv(DiffDartEnv):
    def __init__(self,FD=False):
        frame_skip = 1
        DiffDartEnv.__init__(self,None,frame_skip,dt=0.01,FD=FD)
        # Create a world calling same name
        self.dart_world = dart.simulation.World()
        self.dart_world.setGravity([0, -9.81, 0])
        cartpole = dart.dynamics.Skeleton()

        cartRail, cart = cartpole.createPrismaticJointAndBodyNodePair()
        cartRail.setAxis([1, 0, 0])
        cartShape = cart.createShapeNode(dart.dynamics.BoxShape([0.5, 0.1, 0.1]))
        cartVisual = cartShape.createVisualAspect()
        cartVisual.setColor([0.5, 0.5, 0.5])
        cartRail.setPositionUpperLimit(0, 10)
        cartRail.setPositionLowerLimit(0, -10)
        cartRail.setControlForceUpperLimit(0 ,20)
        cartRail.setControlForceLowerLimit(0 ,-20)

        poleJoint, pole = cartpole.createRevoluteJointAndBodyNodePair(cart)
        poleJoint.setAxis([0, 0, 1])
        poleShape = pole.createShapeNode(dart.dynamics.BoxShape([0.1, 1.0, 0.1]))
        poleVisual = poleShape.createVisualAspect()
        poleVisual.setColor([0.7 , 0.7, 0.7])
        poleJoint.setControlForceUpperLimit(0, 0)
        poleJoint.setControlForceLowerLimit(0, 0)

        poleOffset = dart.math.Isometry3()
        poleOffset.set_translation([0, -0.5, 0])
        poleJoint.setTransformFromChildBodyNode(poleOffset)

        self.dart_world.addSkeleton(cartpole)

        self.dart_world.setTimeStep(self.timestep)
        self.robot_skeleton = cartpole
        self.control_dofs = np.array([0])

    def running_cost(self, x, u, compute_grads = False):
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor(u, requires_grad=True)

        #x_target = torch.FloatTensor([0.5, 0., 0., 0.])
        #coeff = torch.FloatTensor([0.1, 0.5, 0.06, 0.1])

        run_cost = torch.sum(0.01 * torch.mul(u,u))
        
        #run_cost += torch.sum(torch.mul(coeff, torch.mul(x - x_target, x - x_target)))

        if compute_grads:
            run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu = ComputeCostGrad(run_cost, x, u=u)
            return run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu
        else:
            return run_cost.detach().numpy()

    def terminal_cost(self, x, compute_grads = False):
        x = torch.tensor(x, requires_grad=True)

        x_target = torch.FloatTensor([0.5, 0., 0., 0.])
        coeff = torch.FloatTensor([10, 50, 6, 10])
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



