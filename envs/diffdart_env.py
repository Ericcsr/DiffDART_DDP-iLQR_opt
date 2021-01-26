import os
import diffdart as dart
from os import path
from pdb import set_trace as bp
import numpy as np


class DiffDartEnv():
    def __init__(self, model_paths, frame_skip, dt=0.002, FD=False): #observation_size, action_bounds, \
                 #dt=0.002,obs_type="parameter", action_type="continuous", visualize=True):
        
#        self.viewer = None

        if len(model_paths) < 1:
            raise StandardError("At least one model file is needed.")

        if isinstance(model_paths, str):
            model_paths = [model_paths]

        # convert everything to fullpath
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
        self.dart_world.setUseFDOverride(FD) #TODO: USING FD FOR GRADIENTS. REMOVE THIS ONCE GRADIENT ISSUES ARE FIXED!
#        self.np_random = np.random
        
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

        
        self.dart_world.setTimeStep(dt)
        num_skel = self.dart_world.getNumSkeletons()
        self.robot_skeleton = self.dart_world.getSkeleton(num_skel-1) # assume that the skeleton of interest is always the last one

#        self._obs_type = obs_type
        self.frame_skip= frame_skip
#        self.visualize = visualize  #Show the window or not
        #self.disableViewer = disableViewer

        # random perturbation
#        self.add_perturbation = False
#        self.perturbation_parameters = [0.05, 5, 2] # probability, magnitude, bodyid, duration
#        self.perturbation_duration = 40
#        self.perturb_force = np.array([0, 0, 0])

        #assert not done
#        self.obs_dim = observation_size
#        self.act_dim = len(action_bounds[0])

        # for discrete instances, action_space should be defined in the subclass
        #if action_type == "continuous":
        #    self.action_space = spaces.Box(action_bounds[1], action_bounds[0])

        self.track_skeleton_id = -1 # track the last skeleton's com by default

        # initialize the viewer, get the window size
        # initial here instead of in _render
        # in image learning
        #self.screen_width = screen_width
        #self.screen_height = screen_height
        #self._get_viewer()
        # Give different observation space for different kind of envs
        #if self._obs_type == 'parameter':
        #    high = np.inf*np.ones(self.obs_dim)
        #    low = -high
        #    self.observation_space = spaces.Box(low, high)
        #elif self._obs_type == 'image':
        #    # Change to grayscale image later
        #    self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height))
        #else:
        #    raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

        #self.seed()

        #self.viewer = None

        #self.metadata = {
        #    'render.modes': ['human', 'rgb_array'],
        #    'video.frames_per_second' : int(np.round(1.0 / self.dt))
        #}
        

        #self.gui = dart.DartGUI()
        #self.gui.serve(8080)
        #self.gui.stateMachine().renderWorld(self.dart_world)
        #bp()

    @property
    def dt(self):
        return self.dart_world.getTimeStep() * self.frame_skip














#-----------------------------------------------------------------------------------------------------
#------ WARNING: THE FOLLOWING ARE NOT USED. ONLY RETAINED IN CASE WE WANT TO TRAIN AN RL POLICY -----
#-----------------------------------------------------------------------------------------------------

    #def seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]

    # methods to override:
    # ----------------------------
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

    def getViewer(self, sim, title=None):
        # glutInit(sys.argv)
        win = StaticGLUTWindow(sim, title)
        win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0, zoom=0.1), 'gym_camera')
        win.scene.set_camera(win.scene.num_cameras()-1)

        # to add speed,
        if self._obs_type == 'image':
            win.run(self.screen_width, self.screen_height, _show_window=self.visualize)
        else:
            win.run(_show_window=self.visualize)
        return win

    def _get_viewer(self):
        if self.viewer is None and not self.disableViewer:
            self.viewer = self.getViewer(self.dart_world)
            self.viewer_setup()
        return self.viewer

    def state_vector(self):
        return np.concatenate([
            self.robot_skeleton.getPositions(),
            self.robot_skeleton.getVelocities()
        ])