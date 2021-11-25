import torch
from pdb import set_trace as bp

import nimblephysics as dart
from typing import Tuple, Callable, List
import numpy as np
import math

def ComputeCostGrad(cost,x,u=None):
	# utility function to compute gradients and Hessians of provided cost functions
	# Coded per: https://discuss.pytorch.org/t/pytorch-most-efficient-jacobian-hessian-calculation/47417/2
	n_states = x.shape[0]
	Grad_x = torch.autograd.grad(cost, x, create_graph=True, retain_graph=True,allow_unused=True)[0]
	if Grad_x is None:
		grad_x = torch.zeros((n_states,))
	else:
		grad_x = Grad_x

	Hess_xx = torch.zeros((n_states,n_states))
	if Grad_x is not None:
		for i in range(n_states):
			Hess_xx[i,:] = torch.autograd.grad(Grad_x[i],x,create_graph=True, retain_graph=True,allow_unused=True)[0]

	if u is not None:
		n_controls = u.shape[0]
		Grad_u = torch.autograd.grad(cost, u, create_graph=True, retain_graph=True,allow_unused=True)[0]
		if Grad_u is None:
			grad_u = torch.zeros((n_controls,))
		else:
			grad_u = Grad_u
		Hess_uu = torch.zeros((n_controls,n_controls))
		Hess_xu = torch.zeros((n_controls,n_states))

		if Grad_u is not None:
			for i in range(n_controls):
				Hess_uu[i,:] = torch.autograd.grad(grad_u[i], u,create_graph=True, retain_graph=True,allow_unused=True)[0]

		if Grad_x is not None:
			for i in range(n_states):
				xu = torch.autograd.grad(Grad_x[i],u,create_graph=True, retain_graph=True,allow_unused=True)[0]
				if xu is None:
					Hess_xu[:,i] = 0.0
				else:
					Hess_xu[:,i] = xu

		Hess_ux = Hess_xu.T
		return cost.detach().numpy(), grad_x.detach().numpy(), Hess_xx.detach().numpy(), grad_u.detach().numpy(),  Hess_uu.detach().numpy(), Hess_ux.detach().numpy(),  Hess_xu.detach().numpy()
	else:
		return cost.detach().numpy(), grad_x.detach().numpy(), Hess_xx.detach().numpy()



class DartMapPosition(torch.autograd.Function):
    """
    This implements a single, differentiable timestep of DART as a PyTorch layer
    """
    @staticmethod
    def forward(ctx, world, mapping, pos):
        """
        We can't put type annotations on this declaration, because the supertype
        doesn't have any type annotations and otherwise mypy will complain, so here
        are the types:
        world: dart.simulation.World
        mapping: dart.neural.Mapping
        pos: torch.Tensor
        -> torch.Tensor
        """
        m: dart.neural.Mapping = mapping
        world.setPositions(pos.detach().numpy())
        mappedPos = mapping.getPositions(world)
        ctx.into = m.getRealPosToMappedPosJac(world)
        return torch.tensor(mappedPos)
    @staticmethod
    def backward(ctx, grad_pos):
        intoJac = torch.tensor(ctx.into, dtype=torch.float64)
        lossWrtPosition = torch.matmul(torch.transpose(intoJac, 0, 1), grad_pos)
        return (
            None,
            None,
            lossWrtPosition,
        )
def dart_map_pos(
        world: dart.simulation.World, map: dart.neural.Mapping, pos: torch.Tensor) -> torch.Tensor:
    """
    This maps the positions into the mapping passed in, storing necessary info in order to do a backwards pass.
    """
    return DartMapPosition.apply(world, map, pos)  # type: ignore