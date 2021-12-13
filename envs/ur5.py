import numpy as np
from envs.diffdart_env import DiffDartEnv
import nimblephysics as dart
import torch
from .utils import ComputeCostGrad
import os

class UR5Env(DiffDartEnv):
    # TODO: Eric Collision need to be removed
    def __init__(self,FD=False):
        frame_skip = 1
        DiffDartEnv.__init__(self,None, frame_skip, dt = 0.01, FD=FD)
        self.dart_world = dart.simulation.World()
        #self.dart_world.setGravity([0, -9.81, 0])
        ur5 = self.dart_world.loadSkeleton(os.path.join(os.path.dirname(__file__), 
                "./assets/ur5/ur5.urdf"))
        self.robot_skeleton = ur5

        # Set Joint Torque Limits
        forceLimits = np.ones(ur5.getNumDofs()) * 50
        ur5.setControlForceUpperLimits(forceLimits)
        ur5.setControlForceLowerLimits(-forceLimits)

        # Set Controllable Joint
        self.state_dofs = np.arange(self.dart_world.getNumDofs())
        self.control_dofs = np.arange(self.dart_world.getNumDofs())
        self.target = np.ones(2 * self.dart_world.getNumDofs()) * 5
        self.target[6:] = 0
        print(self.target)

    def running_cost(self, x, u, compute_grads = False):
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor(u, requires_grad=True)
        x_target = torch.from_numpy(self.target)
        state_weight = torch.ones_like(x).float()  * 0.00001
        action_weight = torch.ones_like(u).float() * 0.00001

        run_cost = torch.sum(state_weight*((x - x_target)**2))
        run_cost += torch.sum(action_weight * (u**2))

        if compute_grads:
            run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu = ComputeCostGrad(run_cost, x, u=u)
            return run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu
        else:
            return run_cost.detach().numpy()

    def terminal_cost(self, x, compute_grads = False):
        x = torch.tensor(x, requires_grad=True)

        x_target = torch.from_numpy(self.target)
        final_weight = torch.ones_like(x) * 10
        ter_cost = torch.sum(final_weight * ((x - x_target)**2))

        if compute_grads:
            ter_cost, grad_x, Hess_xx = ComputeCostGrad(ter_cost, x)
            return ter_cost, grad_x, Hess_xx
        else:
            return ter_cost.detach().numpy()

    def run_cost(self):
        return self.running_cost

    def ter_cost(self):
        return self.terminal_cost
