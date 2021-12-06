import nimblephysics as dart
import numpy as np
import torch
import time

SHOT_LENGTH = 20

class MultiShot_Traj_Optimizer():
    def __init__(self,
                 Env,
                 T,
                 X0 = None,
                 FD = False):
        self.Env = Env(FD=FD)
        self.world = self.Env.dart_world
        self.dt = self.Env.dt
        self.N = int(T / self.dt)
        self.init_gui = False
        self.init_state = self.world.getState()
        # Set the init state as required
        self.active_dofs = np.concatenate([self.Env.state_dofs, self.Env.state_dofs + self.world.getNumDofs()])
        self.init_state[self.active_dofs] = X0
        print(self.init_state)
        self.reset()
        self.target = self.Env.target
        # Initialize Loss function with Environment
        # Here for efficiency the loss is different from iLQR loss
        self.lossFn = dart.trajectory.LossFn(self.loss)

        self.problem = dart.trajectory.MultiShot(
            self.world, 
            self.lossFn,
            self.N,
            SHOT_LENGTH,
            False)
        self.problem.setParallelOperationsEnabled(True)

        self.optimizer = dart.trajectory.IPOptOptimizer()
        self.optimizer.setLBFGSHistoryLength(5)
        self.optimizer.setCheckDerivatives(False)

    def loss(self,rollout):
        poses = rollout.getPoses('identity')
        vels = rollout.getVels('identity')
        last_pos = poses[:, -1]
        last_vel = vels[:, -1]
        state = np.concatenate([last_pos, last_vel])
        final_loss = ((self.target - state[self.active_dofs])**2).sum()
        return final_loss
        
    def optimize(self, maxIter, thresh, silent = False):
        if thresh == None:
            thresh = 1e-4
        self.optimizer.setTolerance(thresh)
        self.optimizer.setIterationLimit(maxIter)
        self.optimizer.setSilenceOutput(silent)
        start_time = time.time()
        self.result = self.optimizer.optimize(self.problem)
        print("The optimization terminated in: ", time.time() - start_time)
        
        steps = self.result.getNumSteps()
        self.rollout = self.result.getStep(steps - 1).rollout
        X, U = self.getStateActionFromRollout(self.rollout)
        cost = self.compute_cost(X, U)
        return X, U, cost

    def compute_cost(self, X, U):
        cost = 0
        for i in range(len(X)-1):
            state = X[self.active_dofs, i]
            action = U[self.Env.control_dofs, i]
            cost += float(self.Env.running_cost(state, action))
        cost += float(self.Env.terminal_cost(X[self.active_dofs,len(X)-1]))
        return cost

    def createLossFn(self, rollout):
        X, U = self.getStateActionFromRollout(rollout)
        return self.compute_cost(X, U)
        

    def getStateActionFromRollout(self,rollout):
        poses = rollout.getPoses('identity')
        vels = rollout.getVels('identity')
        X = np.vstack([poses, vels])
        U = rollout.getControlForces('identity')
        return X, U

    def simulate_traj(self, X, U, render = False, iter_num = None):
        cost = self.compute_cost(X, U)
        if render:
            if not self.init_gui:
                self.init_gui = True
                self.gui = dart.NimbleGUI(self.world)
                self.gui.serve(8080)
            self.gui.displayState(torch.from_numpy(X[:,0]))
            if not iter_num == None:
                print(f"Iteration: {iter_num} begin")
            input("Press enter to begin rendering")
            self.reset()
            for i in range(X.shape[1]):
                self.gui.displayState(torch.from_numpy(X[:,i]))
                time.sleep(self.dt)
        return cost

    def reset(self):
        self.world.setState(self.init_state)


    



        