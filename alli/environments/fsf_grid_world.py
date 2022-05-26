# Very heavily modified from: https://github.com/chiamp/fast-reinforcement-learning/blob/main/classes.py

import cloudpickle
import hashlib
import numpy as np

import gym
from gym import error, spaces, utils
import torch
from all.core.state import State

from all.environments import Environment
from all.environments.duplicate_env import DuplicateEnvironment

## Source: https://github.com/pytorch/pytorch/blob/e4d522a3cf78d5a32964f774435247d683237a57/torch/testing/_internal/common_utils.py#L776-L788
#numpy_to_torch_dtype_dict = {
#    np.bool_      : torch.bool,
#    np.uint8      : torch.uint8,
#    np.int8       : torch.int8,
#    np.int16      : torch.int16,
#    np.int32      : torch.int32,
#    np.int64      : torch.int64,
#    np.float16    : torch.float16,
#    np.float32    : torch.float32,
#    np.float64    : torch.float64,
#    np.complex64  : torch.complex64,
#    np.complex128 : torch.complex128
#}

class WallPresets:
    @staticmethod
    def simple_rectangle(arena):
        arena[-1, :] = 1
        arena[:, -1] = 1

def _get_even_num_objects_per_class(no, nc):
    assert no >= nc
    return [(no // nc) + 1] * (no % nc) + [no // nc] * (nc - (no % nc))

class FSFGridWorld(Environment):
    actions_moves = [
        (-1, 0),  # UP
        (1, 0),   # DOWN
        (0, -1),  # LEFT
        (0, 1),   # RIGHT
    ]

    def __init__(self,
                 *,
                 task_vec_sampler,
                 max_path_length=50,
                 height=11,
                 width=11,
                 wall_func=WallPresets.simple_rectangle,
                 num_objects=10,
                 num_classes=2,
                 sample_objects_evenly=False,
                 egocentric=True,
                 device=torch.device('cpu'),
                 isolated_env_global_seed=None,
                 deterministic_transition=False,
                 deterministic_transition_salt: int=0,
                 debug_forced_test_task_vecs=None,
                 ):
        self.task_vec_sampler = task_vec_sampler
        self.max_path_length = max_path_length
        self.height = height
        self.width = width
        self.wall_func = wall_func
        self.num_objects = num_objects
        self.num_classes = num_classes
        self.sample_objects_evenly = sample_objects_evenly
        if self.sample_objects_evenly:
            self._num_objects_per_class = _get_even_num_objects_per_class(
                self.num_objects, self.num_classes)
            self._arange_classes = torch.arange(self.num_classes, device=device)

        self.egocentric = egocentric
        if not self.egocentric:
            # Need to add another channel for agent position.
            raise NotImplementedError
        self._device = device
        if isolated_env_global_seed is not None:
            self._global_rand_gen = torch.Generator().manual_seed(isolated_env_global_seed)
        else:
            self._global_rand_gen = None
        self.deterministic_transition = deterministic_transition
        self.deterministic_transition_salt = deterministic_transition_salt

        del debug_forced_test_task_vecs

        self._internal_state = None
        self._agent_pos = None
        self._task_vec = None
        self._step_count = 0
        self._prev_observation = None
        self._last_action = None

        self._phi = self.get_phi()

        self._observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.height * self.width * (self.num_classes+1),),
            #dtype=np.float32,
            dtype=np.int8,
        )
        self._action_space = spaces.Discrete(
            n=len(self.actions_moves),
        )

        self.reset()

    @property
    def name(self):
        return 'FSFGridWorld'

    @property
    def state_space(self):
        return self._observation_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def duplicate(self, n):
        dump = cloudpickle.dumps(self)
        return DuplicateEnvironment([cloudpickle.loads(dump) for _ in range(n)])

    @property
    def device(self):
        return self._device

    def render(self, **kwargs):
        raise NotImplementedError

    def close(self):
        #return self._env.close()
        pass

    def _get_internal_state_int_hash(self):
        return int(hashlib.sha1(
            self._internal_state.data.cpu().numpy().tobytes()).hexdigest(), 16) % 0xffff_ffff_ffff_ffff

    def _gen(self):
        if self.deterministic_transition:
            return torch.Generator().manual_seed(
                (self._get_internal_state_int_hash() + self.deterministic_transition_salt)
                % 0xffff_ffff_ffff_ffff)
        return self._global_rand_gen

    def _torch_randint(self, *args, **kwargs):
        return torch.randint(*args, **kwargs, generator=self._gen())

    def _phi_func(self, obses, actions, _):
        is_single_transition = (obses.ndim == 1)
        if is_single_transition:
            to_tensor = (
                lambda x: torch.tensor(x, device=self._device)
                if isinstance(obses, torch.Tensor)
                else np.asarray)
            obses = obses[None, :]
            #next_obses = next_obses[None, :]
            actions = to_tensor([actions])
        num_samples = obses.shape[0]
        obses = obses.reshape(num_samples, self.height, self.width, self.num_classes+1)

        phi_feats = (obses[
            [i for i in range(num_samples) for _ in range(4)],
            # UP, DOWN, LEFT, RIGHT
            [-1, 1, 0, 0] * num_samples,
            [0, 0, -1, 1] * num_samples
        ].reshape(num_samples, 4, self.num_classes+1)
        [list(range(num_samples)), actions]
        [:, :self.num_classes])

        if is_single_transition:
            phi_feats = phi_feats.squeeze(0)
        return phi_feats.float()

    def get_phi(self):
        if not self.egocentric:
            raise NotImplementedError
        return self._phi_func

    @classmethod
    def get_min_rewards(cls, task_vecs):
        is_single_task = (task_vecs.ndim == 1)
        if is_single_task:
            task_vecs = task_vecs[None, :]
        min_rewards = torch.minimum(task_vecs.min(dim=1)[0], torch.tensor(0.0, device=task_vecs.device))
        if is_single_task:
            min_rewards = min_rewards.squeeze(0)
        return min_rewards

    def get_observation(self):
        if self.egocentric:
            #obs = np.roll(
            #    self._internal_state,
            #    tuple(-p for p in self._agent_pos),
            #    axis=(0, 1))
            obs = torch.roll(
                self._internal_state,
                tuple(-p for p in self._agent_pos),
                dims=(0, 1))
            assert obs[0, 0, -1] == 1
            obs[0, 0, -1] = 0
        else:
            #obs = self._internal_state.copy()
            obs = torch.clone(self._internal_state)

        #return obs.reshape(self.observation_space.shape)
        return obs.view(self.observation_space.shape)

    def sample_open_position(self):
        #poses = np.stack(np.where(self._internal_state.max(axis=2) == 0), axis=0).T
        poses = torch.nonzero(self._internal_state.max(dim=2)[0] == 0)
        return tuple(poses[self._torch_randint(0, len(poses), ()).item()])

    def reset(self):
        self._task_vec = self.task_vec_sampler()
        self._step_count = 0
        self._prev_observation = None
        self._last_action = None
        
        #self._internal_state = np.zeros(
        #    (self.height, self.width, self.num_classes+1),
        #    dtype=self.observation_space.dtype,
        #)
        self._internal_state = torch.zeros(
            (self.height, self.width, self.num_classes+1),
            #dtype=numpy_to_torch_dtype_dict[self.observation_space.dtype],
            dtype=getattr(torch, self.observation_space.dtype.name),
            device=self._device,
        )

        self.wall_func(self._internal_state[:, :, -1])

        self._agent_pos = self.sample_open_position()
        self._internal_state[self._agent_pos][-1] = 1
        
        if self.sample_objects_evenly:
            for obj_cls, num_objs in enumerate(self._num_objects_per_class):
                for _ in range(num_objs):
                    self._internal_state[self.sample_open_position()][obj_cls] = 1
        else:
            for _ in range(self.num_objects):
                #self._internal_state[self.sample_open_position()][np.random.randint(0, self.num_classes)] = 1
                self._internal_state[self.sample_open_position()][self._torch_randint(0, self.num_classes, ()).item()] = 1
        
        return self.get_state()

    @property
    def state(self):
        return self.get_state()

    # For debugging.
    def _get_readable_observation(self, observation):
        observation = torch.clone(observation.view(
            self.height,
            self.width,
            (self.num_classes+1),))
        for i in range(self.num_classes):
            observation[:,:,i][ torch.where( observation[:,:,i]==1 ) ] = i+1
        observation[:,:,-1][ torch.where( observation[:,:,-1]==1 ) ] = -1
        return observation.sum(dim=-1)

    def get_state(self):
        observation = self.get_observation()
        reward = (
            #np.inner(self._phi(self._prev_observation,
            #                   self._last_action,
            #                   observation),
            #         self._task_vec)
            torch.inner(self._phi(self._prev_observation,
                               self._last_action,
                               observation),
                        self._task_vec).item()
            if self._prev_observation is not None
            else 0.0)
        reward_normalizer = self._task_vec.mean().item()
        if reward_normalizer == 0.0:
            reward_normalizer = 1e-6
        return State({
            'observation': observation,
            'done': self._step_count >= self.max_path_length,
            'step_count': self._step_count,
            'reward': reward,
            'reward_normalized': reward / reward_normalizer,
            'task_vec': self._task_vec,
            #'phi': self._phi,
        }, device=self._device)

    def step(self, action):
        self._prev_observation = self.get_observation()
        self._last_action = action
        self.apply_action(action)
        return self.get_state()

    def apply_action(self,action_index):
        self._step_count += 1

        new_agent_pos = tuple(
            ((p + a) % s)
            for p, a, s in zip(
                self._agent_pos,
                self.actions_moves[action_index],
                [self.height, self.width],
            )
        )
        if self._internal_state[new_agent_pos][-1] == 1:
            # Wall or not moving.
            return

        self._internal_state[self._agent_pos][-1] = 0
        num_objects_picked_up = self._internal_state[new_agent_pos][:-1].sum()
        if self.sample_objects_evenly:
            classes_picked_up = self._arange_classes[self._internal_state[new_agent_pos][:-1] > 0]
        self._internal_state[new_agent_pos][:-1] = 0
        self._internal_state[new_agent_pos][-1] = 1
        self._agent_pos = new_agent_pos

        if self.sample_objects_evenly:
            for obj_cls in classes_picked_up:
                self._internal_state[self.sample_open_position()][obj_cls.item()] = 1
        else:
            for _ in range(num_objects_picked_up):
                #self._internal_state[self.sample_open_position()][np.random.randint(0, self.num_classes)] = 1
                self._internal_state[self.sample_open_position()][self._torch_randint(0, self.num_classes, ()).item()] = 1

    def __str__(self):
        #aggregated_state = self._internal_state.copy()
        aggregated_state = torch.clone(self._internal_state)
        for i in range(self.num_classes): # convert objects in feature_i to a number equal to (i+1)
            #aggregated_state[:,:,i][ np.where( aggregated_state[:,:,i]==1 ) ] = i+1
            aggregated_state[:,:,i][ torch.where( aggregated_state[:,:,i]==1 ) ] = i+1
        #aggregated_state[:,:,-1][ np.where( aggregated_state[:,:,-1]==1 ) ] = -1 # convert agent location to a -1 value
        aggregated_state[:,:,-1][ torch.where( aggregated_state[:,:,-1]==1 ) ] = -1 # convert agent location to a -1 value
        #return str( aggregated_state.sum(axis=2) ) # aggregate all objects and the agent into a single
        return str( aggregated_state.sum(dim=2) ) # aggregate all objects and the agent into a single

    def __repr__(self):
        return str(self)

