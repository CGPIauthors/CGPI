import itertools
import os

import numpy as np
from gym.spaces.discrete import Discrete
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env

def _get_reacher_xml(custom_xml_type):
    if custom_xml_type is None:
        # Default XML file.
        return 'reacher.xml'

    assert custom_xml_type in [
        'timestep_0.02',
    ]
    xml_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        'assets_custom',
        f'reacher_{custom_xml_type}.xml'))
    assert os.path.exists(xml_path)
    return xml_path

class ReacherGymEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, use_discrete_actions, obs_space_spec, custom_xml_type=None, env_specific_random_seed=None):
        self.obs_space_spec = obs_space_spec

        utils.EzPickle.__init__(**locals())
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path=_get_reacher_xml(custom_xml_type),
            frame_skip=2)

        if env_specific_random_seed is not None:
            self.seed(env_specific_random_seed)

        # After mujoco_env.MujocoEnv.__init__()
        self.use_discrete_actions = use_discrete_actions

        if use_discrete_actions:
            self._orig_action_space = self.action_space
            assert isinstance(self._orig_action_space, spaces.Box)
            assert np.all(self._orig_action_space.low <= 0.0)
            assert np.all(self._orig_action_space.high >= 0.0)
            self.action_space = Discrete(3 ** np.prod(self._orig_action_space.shape))

    def step(self, a):
        if getattr(self, 'use_discrete_actions', False):
            a_cont = np.zeros(self._orig_action_space.shape, dtype=self._orig_action_space.dtype)
            for cont_action_idx in itertools.product( *[range(s) for s in self._orig_action_space.shape] ):
                curr_element = (a % 3)
                a = a // 3
                a_cont[cont_action_idx] = {
                    0: self._orig_action_space.low[cont_action_idx],
                    1: 0.0,
                    2: self._orig_action_space.high[cont_action_idx],
                }[curr_element]
            a = a_cont

        return self._step_impl(a)

    def _step_impl(self, a):
        #vec = self.get_body_com("fingertip")-self.get_body_com("target")
        #reward_dist = - np.linalg.norm(vec)
        #reward_ctrl = - np.square(a).sum()
        #reward = reward_dist + reward_ctrl
        reward = 0.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        #return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return ob, reward, done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        #while True:
        #    self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
        #    if np.linalg.norm(self.goal) < 0.2:
        #        break
        self.goal = np.zeros((2,))
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        if self.obs_space_spec == 'original_with_zero_target':
            theta = self.sim.data.qpos.flat[:2]
            obs = np.concatenate([
                np.cos(theta),                  # cos of two arms' angles
                np.sin(theta),                  # sin of two arms' angles
                self.sim.data.qpos.flat[2:],    # target x, y.
                self.sim.data.qvel.flat[:2],    # angular velocity of arms
                self.get_body_com("fingertip") - self.get_body_com("target")  # x, y, z (z is 0)
            ])
            assert len(obs) == 11
        elif self.obs_space_spec == 'original_without_target_info':
            theta = self.sim.data.qpos.flat[:2]
            obs = np.concatenate([
                np.cos(theta),                  # cos of two arms' angles
                np.sin(theta),                  # sin of two arms' angles
                self.sim.data.qvel.flat[:2],    # angular velocity of arms
                self.get_body_com("fingertip")  # x, y, z (z doesn't change)
            ])
            assert len(obs) == 9
        else:
            assert False
        return obs

