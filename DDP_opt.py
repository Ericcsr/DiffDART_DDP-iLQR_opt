import numpy as np
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
				 patience = 8 # linesearch/regularization patience
				 ):
		#From the diffDart environment, obtain the world object and the running and terminal cost functions
		self.Env = Env(FD=FD)
		self.world = self.Env.dart_world
		self.robot = self.Env.robot_skeleton
		self.running_cost = self.Env.run_cost()
		self.terminal_cost = self.Env.ter_cost()
		self.T = T
		self.dt = self.Env.dt #this includes frameskip!
		self.N = int(T / self.dt) #number of timesteps
		self.dofs =  self.robot.getNumDofs()
		self.n_states = self.dofs*2
		self.control_dofs = self.Env.control_dofs
		self.n_controls = len(self.control_dofs) 
		self.frame_skip = self.Env.frame_skip
		assert self.frame_skip == 1 #DDP currently supports no frame skip!
		self.profiler = Profiler()
		
		#----- Initializing numpy arrays------
		# Initialize state and control traj
		self.X = np.zeros((self.N,self.n_states))
		if X0 is not None:
			self.X[0,:] = X0
		else:
			self.X0 = np.concatenate((self.robot.getPositions(), self.robot.getVelocities()))

		self.robot.setPositions(self.X[0,0:self.dofs])
		self.robot.setVelocities(self.X[0,self.dofs:])
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
			self.forward_pass()
			self.backward_pass()
			
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
		
			Xnew[j+1,:], self.Fx[j,:,:], self.Fu[j,:,:] = self.dynamics(Xnew[j,:],Unew[j,:],compute_grads=True)
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
		elif not (self.cost - cost) >= 0:
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

			Qx = self.Lx[j,:] + self.Fx[j,:,:].T.dot(Vx)
			Qu = self.Lu[j,:] + self.Fu[j,:,:].T.dot(Vx)
			Qxx = self.Lxx[j,:,:] + self.Fx[j,:,:].T.dot(Vxx.dot(self.Fx[j,:,:]))
			Qux = self.Lux[j,:,:].T + self.Fu[j,:,:].T.dot(Vxx.dot(self.Fx[j,:,:]))
			Quu = self.Luu[j,:,:] + self.Fu[j,:,:].T.dot(Vxx.dot(self.Fu[j,:,:]))
			Quubar = self.Luu[j,:,:] + self.Fu[j,:,:].T.dot((Vxx+self.mu*np.eye(Vxx.shape[0])).dot(self.Fu[j,:,:]))
			Quxbar = self.Lux[j,:,:] + self.Fu[j,:,:].T.dot((Vxx+self.mu*np.eye(Vxx.shape[0])).dot(self.Fx[j,:,:])).T
			#bp()

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



	def dynamics(self,x,u,compute_grads = False):
		pos = x[:self.dofs]
		vel = x[self.dofs:]
		self.robot.setPositions(pos)
		self.robot.setVelocities(vel)
	
		a = np.zeros(self.dofs)
		a[self.control_dofs] =  u
		for _ in range(self.frame_skip):
			self.world.setControlForces(a.clip(min=-35, max=35))
			snapshot = dart.neural.forwardPass(self.world)
		x_next = np.concatenate((self.robot.getPositions(), self.robot.getVelocities()))
		if compute_grads:
			forceVel = snapshot.getControlForceVelJacobian(self.world, perfLog=None)
			forcePos = np.zeros_like(forceVel)
			velVel = snapshot.getVelVelJacobian(self.world)
			velPos = snapshot.getVelPosJacobian(self.world)
			posVel = snapshot.getPosVelJacobian(self.world)
			posPos = snapshot.getPosPosJacobian(self.world)

			Fx = np.block([
			    [posPos[-self.dofs:,-self.dofs:], velPos[-self.dofs:,-self.dofs:]],
			    [posVel[-self.dofs:,-self.dofs:], velVel[-self.dofs:,-self.dofs:]]
			])

			idx = self.control_dofs-self.dofs
			Fu = np.block([
			    [forcePos[-self.dofs:,idx]],
			    [forceVel[-self.dofs:,idx]]
			])			
			return x_next, Fx, Fu
		else:
			return x_next

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


	def simulate_traj(self, X, U, render = False):
		cost = 0
		if render:
			self.gui = dart.NimbleGUI(self.world)
			self.gui.serve(8080)
			self.gui.displayState(torch.from_numpy(self.world.getState()))
			input('Press enter to begin rendering')
		for j in range(self.N-1):
			X[j+1,:] = self.dynamics(X[j,:], U[j,:])
			#print("Before x: ", X[j,:], "Action: ", U[j,:], "After x: ", X[j+1,:])
			if render:
				print("Action:", U[j,:])
				self.gui.displayState(torch.from_numpy(self.world.getState()))
				time.sleep(self.dt)
			cost += self.running_cost(X[j,:],U[j,:])*self.dt
		cost += self.terminal_cost(X[-1,:])

		if render: # repeat animation 5 times
			for _ in range(5):
				time.sleep(10*self.dt)
				for j in range(self.N-1):
					X[j+1,:] = self.dynamics(X[j,:], U[j,:])
					#print("Before x: ", X[j,:], "Action: ", U[j,:], "After x: ", X[j+1,:])
					self.gui.displayState(torch.from_numpy(self.world.getState()))
					time.sleep(self.dt)

		return cost


	