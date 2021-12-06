import numpy as np
from pdb import set_trace as bp
from DDP_opt import DDP_Traj_Optimizer
from MultiShot_opt import MultiShot_Traj_Optimizer
from envs.cart_pole import DartCartPoleEnv
from envs.snake_7link import DartSnake7LinkEnv
from envs.reacher2d import DartReacher2dEnv
from envs.half_cheetah import DartHalfCheetahEnv
from envs.inverted_double_pendulum import DartDoubleInvertedPendulumEnv
from envs.dog import DartDogEnv
from envs.hopper import DartHopperEnv
from envs.custom_cartpole import CustomCartPoleEnv
from envs.contact_cartpole import ContactCartPoleEnv
from envs.drone import CustomDroneEnv
import nimblephysics as dart
from argparse import ArgumentParser, ArgumentTypeError


def main(args):
	#Choose example and corresponding initial condition X0:

	#Env = CustomCartPoleEnv
	#X0 = [1., 15 / 180 * 3.1415, 0.0, 0.0]

	Env = ContactCartPoleEnv
	X0 = [1., 15/180 * 3.1415, 0.0, 0.0]

	#Env = DartCartPoleEnv
	#X0 = [0., 3.1415, 0., 0.] 

	#Env = DartSnake7LinkEnv
	#X0 = [0.]*18
	#X0[9] =0.1 

	#Env = DartReacher2dEnv
	#X0 = None

	#Env = DartDoubleInvertedPendulumEnv
	#X0 = [0., 3.14, 0.0, 0., 0., 0.] 

	# Env = DartHalfCheetahEnv
	# X0 = None

	#Env = DartDogEnv
	#X0 = None

	#Env = DartHopperEnv
	#X0 = None

	#Env = Catapult
	#X0 = None

	#Env = JumpWorm
	#X0 = None

	#Env = CustomDroneEnv
	#X0 = None
	
	FD = False #whether or not to use finite differencing
	U_guess = 'random'# 'random' #choose None or 'random' (use random for snake)
	T = 4.0 # planning horizon in seconds
	lr = 0.5 #learning rate, default value 1.0
	patience = 10

	maxIter = 100# maximum number of iterations
	threshold = None #0.0001#Optional, set to 'None' otherwise. Early stopping of optimization if cost doesn't improve more than this between iterations.

	Optim = MultiShot_Traj_Optimizer(Env=Env, T=T, X0=X0, FD=FD)
	# Optim = DDP_Traj_Optimizer(Env=Env,T=T,X0=X0,FD=FD,U_guess=U_guess,lr=lr,patience=patience, visualize_all= args.visualize_all, alter_strategy=args.alter_strategy)
	x,u,cost = Optim.optimize(maxIter = maxIter, thresh=threshold)
	print("Enter Here")
	#bp()
	#c = DDP.simulate_traj(x, u, render = True)
	if args.visualize_all:
		for i, (X, U) in enumerate(zip(Optim.x_buffer,Optim.u_buffer)):
			c =  Optim.simulate_traj(X, U, render = True, iter_num = i)
			print('Optimal trajectory cost: ', c)
	else:
		c = Optim.simulate_traj(x, u, render = True, iter_num = 0)
		print('Optimal trajectory cost: ', c)
	from matplotlib import pyplot as plt
	plt.figure()
	plt.plot(x)
	plt.title('States')
	plt.figure()
	plt.plot(u)
	plt.title('Controls')
	plt.figure()
	plt.plot(cost)
	plt.title('Cost')
	plt.xlabel('iteration')
	plt.show()

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--visualize_all", action = "store_true", default=False)
	parser.add_argument("--alter_strategy", action = "store_true", default = False)
	args = parser.parse_args()
	main(args)
