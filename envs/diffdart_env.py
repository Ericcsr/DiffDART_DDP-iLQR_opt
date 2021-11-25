import os
import nimblephysics as dart
from os import path
from pdb import set_trace as bp
import numpy as np


class DiffDartEnv():
    def __init__(self, model_paths=None, frame_skip=1, dt=0.002, FD=False): #observation_size, action_bounds, \
        if model_paths != None:
            if len(model_paths) < 1:
                raise Exception("At least one model file is needed.")

            if isinstance(model_paths, str):
                model_paths = [model_paths]

            full_paths = []
            for model_path in model_paths:
                if model_path.startswith("/"):
                    fullpath = model_path
                else:
                    fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
                if not path.exists(fullpath):
                    raise IOError("File %s does not exist"%fullpath)
                full_paths.append(fullpath)

            self.dart_world = dart.simulation.World()
            
            if full_paths[0][-5:] == '.urdf':
                loader = dart.utils.DartLoader()
                skeleton = loader.parseSkeleton(full_paths[0])
                self.dart_world.addSkeleton(skeleton)
            elif full_paths[0][-5:] == '.skel':
                self.dart_world = dart.utils.SkelParser.readWorld(full_paths[0])
            else:
                for fullpath in full_paths:
                    loader = dart.utils.DartLoader()
                    skeleton = loader.parseSkeleton(full_paths[0])
                    self.dart_world.addSkeleton(skeleton)

            self.dart_world.setUseFDOverride(FD)
            self.dart_world.setTimeStep(dt)
            num_skel = self.dart_world.getNumSkeletons()
            self.robot_skeleton = self.dart_world.getSkeleton(num_skel-1) # assume that the skeleton of interest is always the last one
        self.frame_skip= frame_skip
        self.timestep = dt
        self.track_skeleton_id = -1 # track the last skeleton's com by default
    
    @property
    def dt(self):
        return self.dart_world.getTimeStep() * self.frame_skip

#-----------------------------------------------------------------------------------------------------
#------ WARNING: THE FOLLOWING ARE NOT USED. ONLY RETAINED IN CASE WE WANT TO TRAIN AN RL POLICY -----
#-----------------------------------------------------------------------------------------------------
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.perturbation_duration = 0
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.robot_skeleton.getNumDofs(),) and qvel.shape == (self.robot_skeleton.getNumDofs(),)
        self.robot_skeleton.setPositions(qpos)
        self.robot_skeleton.setVelocities(qvel)

    def set_state_vector(self, state):
        self.robot_skeleton.setPositions(state[0:int(len(state)/2)])
        self.robot_skeleton.setVelocities(state[int(len(state)/2):])


    def do_simulation(self, tau, n_frames):
        if self.add_perturbation:
            if self.perturbation_duration == 0:
                self.perturb_force *= 0
                if np.random.random() < self.perturbation_parameters[0]:
                    axis_rand = np.random.randint(0, 2, 1)[0]
                    direction_rand = np.random.randint(0, 2, 1)[0] * 2 - 1
                    self.perturb_force[axis_rand] = direction_rand * self.perturbation_parameters[1]

            else:
                self.perturbation_duration -= 1

        for _ in range(n_frames):
            if self.add_perturbation:
                self.robot_skeleton.bodynodes[self.perturbation_parameters[2]].add_ext_force(self.perturb_force)

            self.robot_skeleton.setForces(tau)
            self.dart_world.step()

    def render(self, mode='human', close=False):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = -self.dart_world.skeletons[self.track_skeleton_id].com()[0]*1
        if close:
            if self.viewer is not None:
                self._get_viewer().close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            data = self._get_viewer().getFrame()
            return data
        elif mode == 'human':
            self._get_viewer().runSingleStep()

    def state_vector(self):
        return np.concatenate([
            self.robot_skeleton.getPositions(),
            self.robot_skeleton.getVelocities()
        ])