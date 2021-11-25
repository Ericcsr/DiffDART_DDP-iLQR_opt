import numpy as np
#from gym import utils
from envs.diffdart_env import DiffDartEnv
from pdb import set_trace as bp
import torch
from .utils import ComputeCostGrad

class DartCartPoleEnv(DiffDartEnv):#, utils.EzPickle):
    def __init__(self,FD=False):
        #control_bounds = np.array([[1.0],[-1.0]])
        frame_skip = 1
        DiffDartEnv.__init__(self, 'cartpole.skel', frame_skip, dt=0.001,FD=FD) #4, control_bounds, dt=0.02)#, disableViewer=False)
        #utils.EzPickle.__init__(self)
        #self.action_scale = 100
        self.control_dofs = np.array([0])



    def running_cost(self, x, u, compute_grads = False): #define the running cost
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor(u, requires_grad=True)

        #---------------------------Enter running cost:-----------------------------------------------------------
        run_cost = torch.sum(0.01*torch.mul(u,u)) #example of quadratic cost
        #---------------------------------------------------------------------------------------------------------

        #Autodiff gradient and Hessian calculation
        if compute_grads:
            run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu = ComputeCostGrad(run_cost, x, u=u)
            return run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu
        else:
            return run_cost.detach().numpy()

    def terminal_cost(self, x, compute_grads = False): #define the terminal cost
        x = torch.tensor(x, requires_grad=True)

        #---------------------------Enter terminal cost:-----------------------------------------------------------
        x_target = torch.FloatTensor([0., 0., 0., 0.])
        coeff = torch.FloatTensor([0.0, 500., 60., 100.])
        ter_cost = torch.sum(torch.mul(coeff,torch.mul(x-x_target,x-x_target))) #example cT*(x-x_target)*2
        #--------------------------------------------------------------------------------------------------------- 

        #Autodiff gradient and Hessian calculation
        if compute_grads:
            ter_cost, grad_x, Hess_xx = ComputeCostGrad(ter_cost, x)
            
            return ter_cost, grad_x, Hess_xx
        else:
            return ter_cost.detach().numpy()

    def run_cost(self):
        return self.running_cost
    def ter_cost(self):
        return self.terminal_cost














#-----------------------------------------------------------------------------------------------------
#------ WARNING: THE FOLLOWING ARE NOT USED. ONLY RETAINED IN CASE WE WANT TO TRAIN AN RL POLICY -----
#-----------------------------------------------------------------------------------------------------


    def step(self, a):
        reward = 1.0

        tau = np.zeros(self.robot_skeleton.getNumDofs())
        tau[0] = a[0] * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}


    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.getPositions(), self.robot_skeleton.getVelocities()]).ravel()

    def reset_model(self,state):
        self.dart_world.reset()

        #qpos = self.robot_skeleton.getPositions() + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.getNumDofs())
        #qvel = self.robot_skeleton.getVelocities() + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.getNumDofs())
        self.set_state_vector(state)
        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0