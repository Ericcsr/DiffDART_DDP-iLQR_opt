import numpy as np
from envs.diffdart_env import DiffDartEnv
from pdb import set_trace as bp
import torch
from .utils import ComputeCostGrad, dart_map_pos
import nimblephysics as dart

class DartReacher2dEnv(DiffDartEnv):
    def __init__(self,FD=False):
        self.target = np.array([0.1, 0.01, -0.1])
        #self.action_scale = np.array([200, 200])
        #self.control_bounds = np.array([[1.0, 1.0],[-1.0, -1.0]])
        frame_skip = 1
        DiffDartEnv.__init__(self, 'reacher2d.skel', frame_skip,FD=FD)#, 11, self.control_bounds, dt=0.01, disableViewer=False)
        self.ndofs = self.robot_skeleton.getNumDofs()
        self.control_dofs = np.arange(0,self.ndofs) #TODO needs to change!
        #for s in self.dart_world.skeletons:
        #    s.set_self_collision_check(False)
        #    for n in s.bodynodes:
        #        n.set_collidable(False)
        #utils.EzPickle.__init__(self)

    def running_cost(self, x, u, compute_grads = False): #define the running cost
        x = torch.tensor(x, requires_grad=True)
        u = torch.tensor(u, requires_grad=True)
        
        end_eff_node = self.robot_skeleton.getBodyNode(self.robot_skeleton.getNumBodyNodes() - 1)
        ikMap: dart.neural.IKMapping = dart.neural.IKMapping(self.dart_world)
        ikMap.addLinearBodyNode(end_eff_node)
        x_end_eff=dart_map_pos(self.dart_world, ikMap, x[0:self.ndofs])

        #x_end_eff = dart.convert_to_world_space_positions_linear(self.dart_world,end_eff_node,x[:self.ndofs])
        #bp()
        x_target = torch.tensor(self.target)



        #---------------------------Enter running cost:-----------------------------------------------------------
        run_cost = torch.sum(0.01*torch.mul(u,u)) #example of quadratic cost
        run_cost += torch.sum(torch.mul(x_end_eff-x_target,x_end_eff-x_target))
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
        
        end_eff_node = self.robot_skeleton.getBodyNode(self.robot_skeleton.getNumBodyNodes() - 1)
        ikMap: dart.neural.IKMapping = dart.neural.IKMapping(self.dart_world)
        ikMap.addLinearBodyNode(end_eff_node)
        x_end_eff=dart_map_pos(self.dart_world, ikMap, x[0:self.ndofs])

        #x_end_eff = dart.convert_to_world_space_positions_linear(self.dart_world,end_eff_node,x[:self.ndofs])

        x_target = torch.tensor(self.target)
        ter_cost = torch.sum(torch.mul(x_end_eff-x_target,x_end_eff-x_target))
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
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        vec = self.robot_skeleton.bodynodes[-1].com() - self.target

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()#*0.1
        reward = reward_dist + reward_ctrl

        s = self.state_vector()
        #done = not (np.isfinite(s).all() and (-reward_dist > 0.02))
        done = False

        return ob, reward, done, {}

    def _get_obs(self):
        theta = self.robot_skeleton.q
        vec = self.robot_skeleton.bodynodes[-1].com() - self.target
        return np.concatenate([np.cos(theta), np.sin(theta), [self.target[0], self.target[2]], self.robot_skeleton.dq, vec]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        while True:
            self.target = self.np_random.uniform(low=-.2, high=.2, size=3)
            self.target[1] = 0.0
            if np.linalg.norm(self.target) < .2: break
        self.target[1] = 0.01

        self.dart_world.skeletons[1].q=[0, 0, 0, self.target[0], self.target[1], self.target[2]]


        return self._get_obs()


    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -1.0
        self._get_viewer().scene.tb._set_theta(-45)
        self.track_skeleton_id = 0
