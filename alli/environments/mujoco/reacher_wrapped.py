import sys
from typing import Optional

import torch

from all.environments.gym import GymEnvironment
from alli.environments.mujoco.reacher_gym import ReacherGymEnv

if '--debug_profile' in sys.argv:
    from rlkit.core.logging import profile
else:
    profile = lambda func: func

class ReacherWrappedEnv(GymEnvironment):
    def __init__(self,
                 *,
                 task_vec_sampler,
                 max_path_length,
                 phi_type,
                 source_xys,
                 obs_space_spec='original_with_zero_target',
                 custom_xml_type=None,
                 device=torch.device('cpu'),
                 env_specific_random_seed: Optional[int]=None,
                 debug_forced_test_task_vecs=None,
                 ):
        env = ReacherGymEnv(
            use_discrete_actions=True,
            obs_space_spec=obs_space_spec,
            custom_xml_type=custom_xml_type,
            env_specific_random_seed=env_specific_random_seed,
        )
        super().__init__(env, device, 'ReacherWrappedEnv')

        self.task_vec_sampler = task_vec_sampler
        self.max_path_length = max_path_length
        self.phi_type = phi_type
        assert phi_type in [
            'neg_dists_to_source_xys',
        ]
        self.source_xys = source_xys
        if phi_type == 'neg_dists_to_source_xys':
            assert isinstance(source_xys, torch.Tensor) and source_xys.ndim == 2
        self.obs_space_spec = obs_space_spec
        assert obs_space_spec in [
            'original_with_zero_target',
            'original_without_target_info',
        ]

        del debug_forced_test_task_vecs

        self._phi = self.get_phi()

    @classmethod
    def convert_target_xy_to_task_vec(cls, *args, **kwargs):
        return cls.convert_target_info_to_task_vec(*args, **kwargs)

    @classmethod
    def convert_target_info_to_task_vec(cls, target_info, phi_type):
        is_single = (target_info.ndim == 1)
        if is_single:
            target_info = target_info[None, :]

        assert target_info.ndim == 2

        if phi_type == 'neg_dists_to_source_xys':
            task_vec = target_info
        else:
            assert False


        if is_single:
            task_vec = task_vec.squeeze(0)
        return task_vec.float()

    @classmethod
    def get_task_vec_dim_general(cls, phi_type, source_xys):
        if phi_type == 'neg_dists_to_source_xys':
            return len(source_xys)

    def get_task_vec_dim(self):
        return ReacherWrappedEnv.get_task_vec_dim_general(self.phi_type, self.source_xys)

    def _phi_func(self, obses, actions, next_obses):
        del obses
        del actions
        assert isinstance(next_obses, torch.Tensor)

        is_single_transition = (next_obses.ndim == 1)
        if is_single_transition:
            next_obses = next_obses[None, :]

        if self.obs_space_spec in ['original_with_zero_target', 'original_without_target_info']:
            fingertip_xy = next_obses[:, -3:-1]
        else:
            assert False

        if self.phi_type == 'neg_dists_to_source_xys':
            phi_feats = (- torch.norm(
                fingertip_xy[:, None, :] - self.source_xys[None, :, :],
                p=2,
                dim=2,
            ))
        else:
            assert False

        assert phi_feats.size(1) == self.get_task_vec_dim()

        if is_single_transition:
            phi_feats = phi_feats.squeeze(0)
        return phi_feats.float()

    def get_phi(self):
        return self._phi_func

    @classmethod
    def get_min_rewards(cls, task_vecs, phi_type, method, *, source_xys):
        is_single_task = (task_vecs.ndim == 1)
        if is_single_task:
            task_vecs = task_vecs[None, :]

        assert source_xys is None or source_xys.ndim == 2

        # Maximum reachable distance (or radius).
        MRD = 0.21

        if method == 'geometric':
            if phi_type == 'neg_dists_to_source_xys':
                raise NotImplementedError
            else:
                assert False
        elif method == 'phi_range':
            if phi_type == 'neg_dists_to_source_xys':
                source_norms = torch.norm(source_xys, p=2, dim=1)
                max_possible_dists = torch.sqrt(
                    MRD ** 2 + 2 * MRD * source_norms + source_norms ** 2
                )
                min_rewards = (
                    task_vecs[:, :, None] * ( torch.cat([
                        (- max_possible_dists)[:, None],
                        torch.zeros_like(max_possible_dists)[:, None],
                    ], dim=1)[None, :, :] )
                ).min(dim=-1).values.sum(dim=1)
            else:
                assert False
        else:
            assert False

        if is_single_task:
            min_rewards = min_rewards.squeeze(0)
        return min_rewards

    @profile
    def _complete_state(self, state):
        state['observation'] = state['observation'].float()
        state['reward'] = torch.inner(
            self._phi(None, None, state['observation']),
            self._task_vec,
        ).item()
        state['step_count'] = self._step_count
        state['task_vec'] = self._task_vec

        if self._step_count >= self.max_path_length:
            state['done'] = True

    def reset(self):
        self._task_vec = self.task_vec_sampler()
        self._step_count = 0

        state = super().reset()
        self._complete_state(state)
        return state

    @profile
    def step(self, action):
        self._step_count += 1

        state = super().step(action)
        self._complete_state(state)
        return self._state

