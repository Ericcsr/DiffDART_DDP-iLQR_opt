from matplotlib import use
import numpy as np
import copy
import torch
import nimblephysics as dart
from pdb import set_trace as bp
import time
from pyinstrument import Profiler


class DDP_Traj_Optimizer():
	def __init__(self,
				 Env, #diffDart Env
				 T,     # Time horizon in seconds
				 X0 = None, # State initial condition (optional; if not provided, it is obtained from world)
				 U_guess = None, # Control initial guess (optional)
				 FD = False, #whether to use finite differencing for gradients
				 lr = 1.0, # initial lr value (will be reduced with linesearch)
				 patience = 8, # linesearch/regularization patience
				 visualize_all = False, # If we need to save and visualize all trajectories
				 alter_strategy = False,
				 perturb_state = False,
				 use_heuristic = False,
				 weights = [0.1, 0.9]):
		#From the diffDart environment, obtain the world object and the running and terminal cost functions
		self.Env = Env(FD=FD)
		self.world = self.Env.dart_world
		self.robot = self.Env.robot_skeleton
		self.running_cost = self.Env.run_cost()
		self.terminal_cost = self.Env.ter_cost()
		self.T = T
		self.dt = self.Env.dt #this includes frameskip!
		self.N = int(T / self.dt) #number of timesteps
		self.dofs =  np.concatenate([self.Env.state_dofs,
									 self.Env.state_dofs + self.world.getNumDofs()])
		print(self.dofs, self.world.getNumDofs())
		self.n_states = len(self.dofs)
		self.control_dofs = self.Env.control_dofs
		self.n_controls = len(self.control_dofs)
		self.frame_skip = self.Env.frame_skip
		self.visualize_all = visualize_all
		self.init_socket = False
		self.alter_strategy = alter_strategy
		self.threshold = 0.5
		self.gamma = 0.5
		self.action_clip = 100
		self.weights = weights
		self.perturb_state = perturb_state
		self.use_heuristic = use_heuristic

		self.idx_x = self.dofs.repeat(self.n_states)
		self.idx_y = np.tile(self.dofs, self.n_states)
		
		# These two buffers are used to store intermediate trajectories
		self.x_buffer = []
		self.u_buffer = []

		assert self.frame_skip == 1 #DDP currently supports no frame skip!
		self.profiler = Profiler()
		
		#----- Initializing numpy arrays------
		# Initialize state and control traj
		self.X = np.zeros((self.N,self.n_states))
		if X0 is not None:
			self.X[0,:] = X0
		else:
			self.X0 = np.concatenate((self.robot.getPositions(), self.robot.getVelocities()))

		self.robot.setPositions(self.X[0,0:self.n_states//2])
		self.robot.setVelocities(self.X[0,self.n_states//2:])
		self.init_state = self.world.getState()
		#self.Env.gui.stateMachine().renderWorld(self.world)
		#bp()
		if U_guess is not None:
			if U_guess == 'random':
				np.random.seed(1001)
				self.U = 0.2*np.random.random(size=(self.N-1,self.n_controls))-0.1
			else:
				self.U = U_guess.copy()
		else:
			self.U = np.zeros((self.N-1,self.n_controls))


		# initialize Cost derivative matrices
		self.Lx = np.zeros((self.N-1,self.n_states))
		self.Lu = np.zeros((self.N-1,self.n_controls))
		self.Lxx = np.zeros((self.N-1,self.n_states,self.n_states))
		self.Luu = np.zeros((self.N-1,self.n_controls,self.n_controls))
		self.Lux = np.zeros((self.N-1,self.n_states,self.n_controls))

		# initialize Jacobians of dynamics, w.r.t. state x and control u
		self.Fx = np.zeros((self.N-1,self.n_states,self.n_states))   
		self.Fu = np.zeros((self.N-1,self.n_states,self.n_controls))

		if self.alter_strategy:
			self.Fx_alter = np.zeros((self.N-1,self.n_states,self.n_states))   
			self.Fu_alter = np.zeros((self.N-1,self.n_states,self.n_controls))
	
		# initialize feedback and feedforward control updates
		self.K = np.zeros((self.N-1,self.n_controls,self.n_states))
		self.k = np.zeros((self.N-1,self.n_controls))


		self.Costs = []
		#self.DeltaJ = 1.0 #Initialize DeltaJ
		self.alpha_reset_value = lr # initial learning rate value
		self.alpha = self.alpha_reset_value #initialize learning rate
		self.patience_reset_value = patience #linesearch patience
		self.patience = self.patience_reset_value 
		self.early_termination = False
		
		#regularization schedule:
		self.DELTA0 = 2. 
		self.DELTA = self.DELTA0
		self.mu_min = 1e-6
		self.mu = 100.*self.mu_min

	def optimize(self,maxIter,thresh=None):
		t = time.time()
		self.cost = self.simulate_traj(self.X, self.U)
		prev_cost = np.inf
		i = 0
		while i < maxIter and not self.early_termination:
			self.reset()
			self.forward_pass()
			self.backward_pass()
			self.threshold *= self.gamma
			if self.visualize_all:
				self.x_buffer.append(copy.deepcopy(self.X))
				self.u_buffer.append(copy.deepcopy(self.U))
			
			self.Costs.append(self.cost)

			print('Iteration: ', i+1, ', trajectory cost: ', self.cost)

			if thresh is not None:
				if abs(prev_cost-self.cost)<thresh:
					print('Optimization threshold met, exiting...')
					break
				else:
					prev_cost = self.cost.copy()
			i += 1
		print('---Optimization completed in ',time.time()-t,'sec---')
		return self.X, self.U, self.Costs

	# ========== Helper Functions begin ==========
	def forward_pass(self):
		Xnew = self.X.copy()
		Unew = self.U.copy()
		cost = 0.0
		for j in range(self.N-1):
        	
			Unew[j,:] = Unew[j,:] + self.alpha*self.k[j,:]+self.K[j,:,:].dot(Xnew[j,:]-self.X[j,:]) #update control (zero in first iteration)

			l0, l_x, l_xx, l_u, l_uu, l_ux, l_xu = self.running_cost(Xnew[j,:],Unew[j,:],compute_grads=True)
			
			cost+=l0*self.dt
			self.Lx[j,:] = l_x*self.dt
			self.Lu[j,:] = l_u*self.dt
			self.Lxx[j,:] = l_xx*self.dt
			self.Luu[j,:,:] = l_uu*self.dt
			self.Lux[j,:,:] = l_ux*self.dt
		
			if self.alter_strategy:
				Xnew[j+1,:], self.Fx[j,:,:], self.Fu[j,:,:], self.Fx_alter[j,:,:], self.Fu_alter[j,:,:] \
					= self.dynamics(Xnew[j,:],Unew[j,:],compute_grads=True, perturb_state=self.perturb_state)
			else:
				Xnew[j+1,:], self.Fx[j,:,:], self.Fu[j,:,:] \
					= self.dynamics(Xnew[j,:], Unew[j,:], compute_grads=True, perturb_state=self.perturb_state)
		#print("Process Cost: ", cost)
		
		cost+= self.terminal_cost(Xnew[-1,:])
		
		#Linesearch back-tracking
		if  self.check_inf_nan(cost) or self.check_inf_nan(Xnew[-1,:]):
			print(f"NaN Detected ...Cost: {cost} Last x: {Xnew[-1,:]}")
			if self.patience == 0:
				print('Linesearch patience limit met, exiting... ')
				self.early_termination = True
			else:
				self.alpha *= 0.5
				#print('Linesearch: decreasing alpha to ', self.alpha)
				self.patience -= 1	
				self.forward_pass() #retry with smaller learning rate
		elif (not (self.cost - cost) >= 0 and not self.perturb_state):
			#print(f"Current Cost:{self.cost} Candidate Cost: {cost}")
			if self.patience == 0:
				print('Linesearch patience limit met, exiting... ')
				self.early_termination = True
			else:
				self.alpha *= 0.5
				#print('Linesearch: decreasing alpha to ', self.alpha)
				self.patience -= 1	
				self.forward_pass() #retry with smaller learning rate
		else:
			self.X=Xnew
			self.U=Unew
			self.cost = cost
			self.patience = self.patience_reset_value #reset linesearch patience
			self.alpha = self.alpha_reset_value #reset learning rate
			#return cost
	
	def backward_pass(self):
		# initialize backward pass:
		#self.DeltaJ = 0.0
		_, Vx, Vxx = self.terminal_cost(self.X[-1,:],compute_grads=True)
		j = self.N-2
		while j >= 0 and not self.early_termination:
			Fu = self.Fu[j,:,:]
			Fx = self.Fx[j,:,:]
			if self.alter_strategy:
				Fu = self.weights[0] * Fu + self.weights[1] * self.Fu_alter[j,:,:]
				Fx = self.weights[0] * Fx + self.weights[1] * self.Fx_alter[j,:,:]
			if self.use_heuristic and not self.alter_strategy:
				# Need to know the Jacobian wrt contact velocity
				# Which is Contact Jacobian
				dV_dx = Vxx @ self.X[j,:] + Vx # Should be a vector considered as upstream gradient
				pass


			Qx = self.Lx[j,:] + Fx.T.dot(Vx)
			Qu = self.Lu[j,:] + Fu.T.dot(Vx)
			Qxx = self.Lxx[j,:,:] + Fx.T.dot(Vxx.dot(Fx))
			Qux = self.Lux[j,:,:].T + Fu.T.dot(Vxx.dot(Fx))
			Quu = self.Luu[j,:,:] + Fu.T.dot(Vxx.dot(Fu))
			Quubar = self.Luu[j,:,:] + Fu.T.dot((Vxx+self.mu*np.eye(Vxx.shape[0])).dot(Fu))
			Quxbar = self.Lux[j,:,:] + Fu.T.dot((Vxx+self.mu*np.eye(Vxx.shape[0])).dot(Fx)).T
			
			if not self.is_invertible(Quubar):
				if self.patience == 0:
					self.early_termination = True
					print('Regularization patience limit met, exiting... ')
					break
				else: 	
					print('Warning: singular Quu, iteration: ', j ,'- repeating backward pass with increased mu.')
					break
					self.increase_mu()
					self.patience -= 1
					self.backward_pass() #retry with larger regularization
			else:
				Quubar_inv = np.linalg.inv(Quubar)

		
				self.K[j,:,:] = -Quubar_inv.dot(Quxbar.T) 
				self.k[j,:] = -Quubar_inv.dot(Qu)

				#DeltaV =  Qu.T.dot(self.k[j,:])+0.5*self.k[j,:].T.dot(Quu.dot(self.k[j,:])) #not needed?
				Vx = Qx + self.K[j,:,:].T.dot(Quu.dot(self.k[j,:])) + self.K[j,:,:].T.dot(Qu) + Qux.T.dot(self.k[j,:]) 
				Vxx = Qxx + self.K[j,:,:].T.dot(Quu.dot(self.K[j,:,:])) + self.K[j,:,:].T.dot(Qux) + Qux.T.dot(self.K[j,:,:]) 
				#self.DeltaJ+=self.alpha*self.k[j,:].T.dot(Qu) + 0.5*self.alpha**2 *self.k[j,:].T.dot(Quu.dot(self.k[j,:]))
				j -= 1
		if not self.early_termination:
			self.decrease_mu() # decrease mu if backward pass was successful
			self.patience = self.patience_reset_value #reset patience value
		else:
			return

	def dynamics(self,x,u,compute_grads = False, perturb_state = False):
		if perturb_state:
			epsilon = np.random.random()
			if epsilon > (1-self.threshold): # self.threshold = 0.5 * 0.9^n
				x += np.random.random() * 0.02 - 0.01
		pos = x[:self.n_states//2]
		vel = x[self.n_states//2:]
		self.robot.setPositions(pos)
		self.robot.setVelocities(vel)
	
		a = np.zeros(self.world.getNumDofs())
		a[self.control_dofs] =  u
		for _ in range(self.frame_skip):
			self.world.setControlForces(a.clip(min=-self.action_clip, max=self.action_clip))
			snapshot = dart.neural.forwardPass(self.world)
			
		x_next = np.concatenate((self.robot.getPositions(), self.robot.getVelocities()))
		if compute_grads:
			actionJacobian = self.getActionJacobian(snapshot, self.world)
			stateJacobian = self.getStateJacobian(snapshot, self.world)
			if self.alter_strategy:
				Fu_alter = self.getContactFreeActionJacobian(snapshot,self.world)				
				Fx_alter = self.getContactFreeStateJacobian(snapshot,self.world)

			Fx = stateJacobian			
			Fu = actionJacobian
			if self.alter_strategy:
				return x_next, Fx, Fu, Fx_alter, Fu_alter
			else:
				return x_next, Fx, Fu
		else:
			return x_next

	def getStateJacobian(self, snapshot, world):
		stateJacobian = snapshot.getStateJacobian(world)
		return stateJacobian[self.idx_x, self.idx_y].reshape(self.n_states, self.n_states)

	def getContactFreeStateJacobian(self, snapshot, world):
		stateJacobian = snapshot.getContactFreeStateJacobian(world)
		return stateJacobian[self.idx_x, self.idx_y].reshape(self.n_states, self.n_states)

	def getActionJacobian(self, snapshot, world):
		actionJacobian = snapshot.getActionJacobian(world)
		return actionJacobian[self.dofs,:]

	def getContactFreeActionJacobian(self, snapshot, world):
		actionJacobian = snapshot.getContactFreeActionJacobian(world)
		return actionJacobian[self.dofs,:]

	def is_invertible(self,M):
		if M.shape[0] == 1 and abs(M)<1e-6:
			return False
		else:
			return M.shape[0]==M.shape[1] and np.linalg.matrix_rank(M) == M.shape[0]

	def check_inf_nan(self,x):
		if np.isnan(x).any() or (abs(x)>1e6).any():
			return True
		else: 
			return False

	def increase_mu(self):
		self.DELTA = max(self.DELTA0,self.DELTA*self.DELTA0)
		self.mu = max(self.mu_min,self.mu*self.DELTA)

	def decrease_mu(self):
		self.DELTA = min(1.0/self.DELTA0, self.DELTA/self.DELTA0)
		self.mu = self.mu*self.DELTA if self.mu*self.DELTA>self.mu_min else 0.0
	# ========== Helper Function End ============
	
	def simulate_traj(self, X, U, render = False, iter_num=None):
		cost = 0
		if render:
			if self.init_socket == False:
				self.init_socket = True
				self.gui = dart.NimbleGUI(self.world)
				self.gui.serve(8080)
			self.gui.displayState(torch.from_numpy(X[0,:]))
			if not iter_num == None:
				print(f"Iteration: {iter_num} begin")
			input('Press enter to begin rendering')
		self.reset()
		for j in range(self.N-1):
			X[j+1,:] = self.dynamics(X[j,:], U[j,:])
			if render:
				self.gui.displayState(torch.from_numpy(self.world.getState()))
				time.sleep(self.dt)
			cost += self.running_cost(X[j,:],U[j,:])*self.dt
		cost += self.terminal_cost(X[-1,:])

		if render: # repeat animation 5 times
			for _ in range(1):
				time.sleep(10*self.dt)
				self.reset()
				for j in range(self.N-1):
					X[j+1,:] = self.dynamics(X[j,:], U[j,:])
					#print("Before x: ", X[j,:], "Action: ", U[j,:], "After x: ", X[j+1,:])
					self.gui.displayState(torch.from_numpy(self.world.getState()))
					time.sleep(self.dt)

		return cost

	def reset(self):
		self.world.setState(self.init_state)


	