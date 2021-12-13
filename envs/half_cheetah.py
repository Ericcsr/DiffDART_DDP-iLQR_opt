import numpy as np
from envs.diffdart_env import DiffDartEnv
from pdb import set_trace as bp
import torch
from .utils import ComputeCostGrad

class DartHalfCheetahEnv(DiffDartEnv):
    def __init__(self,FD=False):
        #self.control_bounds = np.array([[1.0]*6,[-1.0]*6])
        #self.action_scale = np.array([120, 90, 60, 120, 60, 30]) * 1.0
        #obs_dim = 17

        #self.velrew_weight = 1.0

        #self.t = 0

        #self.total_dist = []
        frame_skip = 1
        DiffDartEnv.__init__(self, ['half_cheetah.skel'], frame_skip,dt=0.002,FD=FD)# obs_dim, self.control_bounds, disableViewer=True, dt=0.01)
        self.ndofs = self.robot_skeleton.getNumDofs()
        self.dart_world.removeDofFromActionSpace(0)
        self.dart_world.removeDofFromActionSpace(1)
        self.dart_world.removeDofFromActionSpace(2)
        
        self.dart_world2 = self.dart_world.clone()
        self.dart_world2.getSkeleton('ground skeleton').getBodyNode(0).removeAllShapeNodes()
        self.robot_skeleton_no_contact = self.dart_world2.getSkeleton(1)
        
        self.control_dofs = np.arange(6)
        self.state_dofs = np.arange(self.ndofs)

    def running_cost(self, x, u, compute_grads = False): #define the running cost
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor(u, requires_grad=True)
        #bp()
        mask = torch.zeros(self.ndofs*2)
        mask[self.ndofs] = 1.0
        #mask[self.ndofs+2]=0.001
        x_target = torch.zeros(self.ndofs*2)
        x_target[self.ndofs] = 10.0



        #---------------------------Enter running cost:-----------------------------------------------------------
        run_cost = torch.sum(0.1*torch.mul(u,u)) #example of quadratic cost
        run_cost += torch.sum(torch.mul(mask,torch.mul(x-x_target,x-x_target))) #cost = (v0-10)^2: make v0 "big", i.e., close to 10
        #---------------------------------------------------------------------------------------------------------
        #bp()
        #Autodiff gradient and Hessian calculation
        if compute_grads:
            run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu = ComputeCostGrad(run_cost, x, u=u)
            return run_cost, grad_x, Hess_xx, grad_u, Hess_uu, Hess_ux, Hess_xu
        else:
            return run_cost.detach().numpy()

    def terminal_cost(self, x, compute_grads = False): #define the terminal cost
        x = torch.tensor(x, requires_grad=True)

        #---------------------------Enter terminal cost:-----------------------------------------------------------
        #x_target = torch.FloatTensor([0., 0., 0., 0.])
        #coeff = torch.FloatTensor([0.0, 1000., 60., 100.])
        #ter_cost = torch.sum(torch.mul(coeff,torch.mul(x-x_target,x-x_target))) #example cT*(x-x_target)*2
        mask = torch.zeros(self.ndofs*2)
        mask[0] = 10.0
        #mask[2]=0.1
        x_target = 0.5*torch.ones(self.ndofs*2)
        x_target[0] = 5.0
        ter_cost = torch.sum(torch.mul(mask,torch.mul(x-x_target,x-x_target))) #cost = 10(x0-10)^2+0.1 x2^2 : make x0 "big", i.e., close to 10
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






    def advance(self, a):
        self.posbefore = self.robot_skeleton.q[0]
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]

        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale
        self.do_simulation(tau, self.frame_skip)

    def terminated(self):
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and np.abs(s[2]) < 1.3)
        return done

    def pre_advance(self):
        self.posbefore = self.robot_skeleton.q[0]

    def reward_func(self, a, step_skip=1):
        posafter = self.robot_skeleton.q[0]
        alive_bonus = 1.0
        reward = (posafter - self.posbefore) / self.dt * self.velrew_weight
        reward += alive_bonus * step_skip
        reward -= 1e-1 * np.square(a).sum()

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all())

        if done:
            reward = 0

        return reward

    def step(self, a):
        self.t += self.dt
        self.pre_advance()
        self.advance(a)
        reward = self.reward_func(a)

        done = self.terminated()

        ob = self._get_obs()

        self.cur_step += 1

        envinfo = {}

        return ob, reward, done, envinfo

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)

        self.cur_step = 0

        self.height_threshold_low = 0.56*self.robot_skeleton.bodynodes[2].com()[1]
        self.t = 0

        self.fall_on_ground = False

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -4

    def state_vector(self):
        s = np.copy(np.concatenate([self.robot_skeleton.q, self.robot_skeleton.dq]))
        return s

    def set_state_vector(self, s):
        snew = np.copy(s)
        self.robot_skeleton.q = snew[0:len(self.robot_skeleton.q)]
        self.robot_skeleton.dq = snew[len(self.robot_skeleton.q):]

