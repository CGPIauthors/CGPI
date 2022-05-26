import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import argparse
import copy
import datetime
import glob
import os
import pprint
import re
import shutil
import sys
from typing import Optional
from typing_extensions import OrderedDict
from alli.approximation.approximation_ex import ApproximationEx
from alli.approximation.ensemble import EnsembleModule

from alli.core.tensor_util import unsqueeze_and_expand

g_file_path = __file__
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))

g_start_time = int(datetime.datetime.now().timestamp())

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import figure
#from matplotlib import cm

import copy
from gym import spaces
import numpy as np
import torch
from torch.optim import Adam
from all import nn
from all.core.state import State, StateArray
#from all.agents import DQN, DQNTestAgent
from alli.agents.usfa import USFA, USFATestAgent
from all.approximation import QNetwork
from alli.approximation.target.fixed_ex import FixedTargetEx
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.optim import LinearScheduler
from all.policies import GreedyPolicy
from all.presets.builder import PresetBuilder
from all.presets.preset import Preset
#from all.presets.classic_control.models import fc_relu_q
from test_scripts import exp_utils
#import exp_utils as exp_utils
from alli.approximation.checkpointer import PeriodicCheckpointerEx
from alli.approximation.usfa_network import USFAPsiModule, USFAPsiNetwork
from alli.environments.mujoco.reacher_wrapped import ReacherWrappedEnv
from alli.experiments.single_env_experiment_ex import SingleEnvExperimentEx
from rlkit.core.logging import logger_main, logger_sub

if '--debug_profile' in sys.argv:
    from rlkit.core.logging import profile
else:
    profile = lambda func: func

from test_scripts.train_fsf_grid_world_usfa import USFAPsiModelPresets


class ReacherTaskPresets:
    _device = 'cuda'
    _reacher_hyperparameters = None
    _usfa_con_hyperparameters = None


    class Base:
        @classmethod
        def get_task_vec_dim(cls):
            return ReacherWrappedEnv.get_task_vec_dim_general(
                phi_type=ReacherTaskPresets._reacher_hyperparameters['phi_type'],
                source_xys=cls.source_xys())

        @classmethod
        def usfa_con_task_vec_sampler(cls, shape):
            raise NotImplementedError

        @classmethod
        def _init_task_vecs(cls):
            if not hasattr(cls, '_train_task_vecs'):
                cls._train_task_vecs = ReacherWrappedEnv.convert_target_info_to_task_vec(
                    target_info=torch.as_tensor(cls._train_target_infos),
                    phi_type=ReacherTaskPresets._reacher_hyperparameters['phi_type'],
                ).tolist()
            if not hasattr(cls, '_test_task_vecs'):
                cls._test_task_vecs = ReacherWrappedEnv.convert_target_info_to_task_vec(
                    target_info=torch.as_tensor(cls._test_target_infos),
                    phi_type=ReacherTaskPresets._reacher_hyperparameters['phi_type'],
                ).tolist()

        @classmethod
        def train_task_vecs(cls):
            if not hasattr(cls, '_train_task_vecs'):
                cls._init_task_vecs()
            if not torch.is_tensor(cls._train_task_vecs):
                cls._train_task_vecs = torch.as_tensor(cls._train_task_vecs, device=ReacherTaskPresets._device)
                source_task_vecs_matrix_rank = torch.linalg.matrix_rank(cls._train_task_vecs).item()
                if ReacherTaskPresets._reacher_hyperparameters['phi_type'] == 'neg_dists_to_source_xys':
                    assert source_task_vecs_matrix_rank == cls.get_task_vec_dim()
                else:
                    assert False
            return cls._train_task_vecs

        @classmethod
        def test_task_vecs(cls):
            if not hasattr(cls, '_test_task_vecs'):
                cls._init_task_vecs()
            if not torch.is_tensor(cls._test_task_vecs):
                if ReacherTaskPresets._reacher_hyperparameters['debug_forced_test_task_vecs'] is not None:
                    cls._test_task_vecs = torch.as_tensor(
                        ReacherTaskPresets._reacher_hyperparameters['debug_forced_test_task_vecs'],
                        device=ReacherTaskPresets._device,
                    ).reshape(-1, cls.get_task_vec_dim())
                else:
                    cls._test_task_vecs = torch.as_tensor(cls._test_task_vecs, device=ReacherTaskPresets._device)
            return cls._test_task_vecs

        @classmethod
        def train_task_vec_sampler(cls):
            return cls.train_task_vecs()[
                torch.randint(0, len(cls.train_task_vecs()), ()).item()]

        _next_test_idx = 0
        @classmethod
        def test_task_vec_sampler(cls):
            #return cls.test_task_vecs()[np.random.randint(0, len(cls.test_task_vecs()))]
            vec = cls.test_task_vecs()[cls._next_test_idx]
            cls._next_test_idx = (cls._next_test_idx + 1) % len(cls.test_task_vecs())
            return vec

        @classmethod
        def source_xys(cls):
            return None



    class _TraitNegDists(Base):
        @classmethod
        def source_xys(cls):
            assert ReacherTaskPresets._reacher_hyperparameters['phi_type'] == 'neg_dists_to_source_xys'
            return torch.as_tensor(cls._source_xys)

    class NegDistsPositiveD4Type1(_TraitNegDists):
        _source_xys = [
            [0.1, 0.0],
            [0.0, 0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
        ]
        _test_task_vecs = [
            [1.0, 1.0, 1.0, 1.0],
        ]

        @classmethod
        def _init_task_vecs(cls):
            cls._train_task_vecs = np.eye(len(cls._source_xys)).tolist()
            super()._init_task_vecs()






class USFAPolicySamplerPresets:
    class CondGaussian0:
        @classmethod
        def policy_vec_sampler(cls, task_vec, shape):
            return unsqueeze_and_expand(
                task_vec,
                dim=0,
                num_repeat=np.prod(shape),
            ).view(*shape, *task_vec.shape)

    class CondGaussian1:
        @classmethod
        def policy_vec_sampler(cls, task_vec, shape):
            rand = torch.randn(*shape, *task_vec.shape, device=task_vec.device) * 0.1
            return task_vec + rand

    class CondGaussian2:
        @classmethod
        def policy_vec_sampler(cls, task_vec, shape):
            rand = torch.randn(*shape, *task_vec.shape, device=task_vec.device) * 0.25
            return task_vec + rand

    class CondGaussian3:
        @classmethod
        def policy_vec_sampler(cls, task_vec, shape):
            rand = torch.randn(*shape, *task_vec.shape, device=task_vec.device) * 0.5
            return task_vec + rand

def ensure_file(path):
    if os.path.exists(path):
        return path
    g = glob.glob(path)
    assert len(g) == 1
    return g[-1]

class USFAReacherPreset(Preset):
    """# {{{

    Args:
        env (all.environments.AtariEnvironment): The environment for which to construct the agent.
        name (str): A human-readable name for the preset.
        device (torch.device): The device on which to load the agent.

    Keyword Args:
        discount_factor (float, optional): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        minibatch_size (int): Number of experiences to sample in each training update.
        update_frequency (int): Number of timesteps per training update.
        target_update_frequency (int): Number of timesteps between updates the target network.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        replay_buffer_size (int): Maximum number of experiences to store in the replay buffer.
        initial_exploration (float): Initial probability of choosing a random action,
            decayed over course of training.
        final_exploration (float): Final probability of choosing a random action.
        final_exploration_step (int): The step at which exploration decay is finished
        test_exploration (float): The exploration rate of the test Agent
        model_constructor (function): The function used to construct the neural model.
    """# }}}

    def __init__(self,
                 env,
                 name,
                 device,
                 hyperparameters):
        super().__init__(name, device, hyperparameters)
        self.env = env
        #self.model = hyperparameters['model_constructor'](env).to(device)
        #self.model = USFAPsiModelPresets.fc_relu_psi1
        if hyperparameters['psi_ckpt_path'] is not None:
            ckpt_path = ensure_file(hyperparameters['psi_ckpt_path'])
            logger_main.log(f'Loading model from {ckpt_path}')
            if not hyperparameters['log_to_stdout']:
                print(f'Loading model from {ckpt_path}')

            loaded_model = torch.load(
                ckpt_path,
                map_location=hyperparameters['device'],
            )
            logger_main.log(f'Loaded model: {loaded_model}')
            if not hyperparameters['log_to_stdout']:
                print(f'Loaded model: {loaded_model}')

            if isinstance(loaded_model, EnsembleModule):
                self.model = [
                    psi_module.model
                    for psi_module in loaded_model.models
                ]
            elif isinstance(loaded_model, USFAPsiModule):
                self.model = [loaded_model.model]
            else:
                assert False
            del loaded_model
        else:
            self.model = [
                getattr(USFAPsiModelPresets, hyperparameters['psi_model'])(
                    env, env.get_task_vec_dim(), hyperparameters['psi_model_hidden']).to(device)
                for _ in range(hyperparameters['psi_num_ensembles'])
            ]
        self.n_actions = env.action_space.n
        self.discount_factor = self.hyperparameters['discount_factor']

    def agent(self, writer=DummyWriter(), train_steps=float('inf')):
        #optimizer = Adam(self.model.parameters(), lr=self.hyperparameters['lr'])
        optimizer = Adam(
            [p for m in self.model for p in m.parameters()],
            lr=self.hyperparameters['lr'])

        checkpointer = PeriodicCheckpointerEx(
            self.hyperparameters['save_frequency_in_updates'],
            os.path.join(writer.log_dir, 'psi_{:09d}.pt'))

        psi = USFAPsiNetwork(
            model=self.model,
            action_dim=self.n_actions,
            task_vec_dim=self.env.get_task_vec_dim(),
            normalize_policy_vecs=self.hyperparameters['normalize_policy_vecs'],
            ensemble_reduction_info=dict(
                psi_num_ensemble_groups=self.hyperparameters['psi_num_ensemble_groups'],
            ),
            optimizer=optimizer,
            target=FixedTargetEx(self.hyperparameters['target_update_frequency']),
            writer=writer,
            checkpointer=checkpointer,
        )

        psi_hash = psi.get_parameters_hash()
        logger_main.log(f'Initialized model {type(psi)} with hash {psi_hash}: {psi}')
        if not self.hyperparameters['log_to_stdout']:
            print(f'Initialized model {type(psi)} with hash {psi_hash}: {psi}')

        if self.hyperparameters['psi_num_ensemble_groups'] is not None:
            gpi_q = psi.construct_gpi_q_network(q_ensemble_reduction='specific_group_mean')
        else:
            gpi_q = psi.construct_gpi_q_network(q_ensemble_reduction='mean')

        gpi_policy = GreedyPolicy(
            gpi_q,
            self.n_actions,
            epsilon=self.hyperparameters['exploration'],
            #epsilon=LinearScheduler(
            #    self.hyperparameters['initial_exploration'],
            #    self.hyperparameters['final_exploration'],
            #    self.hyperparameters['replay_start_size'],
            #    self.hyperparameters['final_exploration_step'] - self.hyperparameters['replay_start_size'],
            #    name="exploration",
            #    writer=writer
            #)
        )

        replay_buffer = ExperienceReplayBuffer(
            self.hyperparameters['replay_buffer_size'],
            device=self.device
        )

        hps_usfa_con = self.hyperparameters['usfa_constraints']

        return USFA(
            psi=psi,
            gpi_policy=gpi_policy,
            replay_buffer=replay_buffer,
            phi=self.env.get_phi(),
            task_vec_dim=self.env.get_task_vec_dim(),
            action_dim=self.n_actions,
            train_task_vec_sampler=getattr(
                ReacherTaskPresets, self.hyperparameters['reacher']['task_preset']).train_task_vec_sampler,
            train_task_vec_sampling_frequency=self.hyperparameters['train_task_vec_sampling_frequency'],
            policy_vec_sampler=getattr(
                USFAPolicySamplerPresets, self.hyperparameters['policy_sampler']).policy_vec_sampler,
            num_policy_samples_per_task=self.hyperparameters['num_policy_samples_per_task'],
            #use_gpi_samples_for_training=True,
            discount_factor=self.hyperparameters['discount_factor'],
            max_path_length=self.hyperparameters['reacher']['max_path_length'],
            minibatch_size=self.hyperparameters['minibatch_size'],
            replay_start_size=self.hyperparameters['replay_start_size'],
            psi_num_ensembles=self.hyperparameters['psi_num_ensembles'],
            psi_num_ensemble_groups=self.hyperparameters['psi_num_ensemble_groups'],
            psi_ensemble_train_even_mask=self.hyperparameters['psi_ensemble_train_even_mask'],
            psi_ensemble_train_with_own_group_samples_only=self.hyperparameters['psi_ensemble_train_with_own_group_samples_only'],
            psi_ensemble_train_rand_mask_bernoulli_prob=self.hyperparameters['psi_ensemble_train_rand_mask_bernoulli_prob'],
            psi_ensemble_next_actions_strategy=self.hyperparameters['psi_ensemble_next_actions_strategy'],
            update_frequency=self.hyperparameters['update_frequency'],
            state_pool_size=self.hyperparameters['gpi_source_selection_state_pool_size'],
            usfa_con_task_vec_sampler=getattr(
                ReacherTaskPresets, self.hyperparameters['reacher']['task_preset']).usfa_con_task_vec_sampler,
            usfa_con_num_task_vecs=hps_usfa_con['usfa_con_num_task_vecs'],
            usfa_con_mode=hps_usfa_con['usfa_con_mode'],
            usfa_con_train_frequency=hps_usfa_con['usfa_con_train_frequency'],
            usfa_con_cache_and_reuse_last_bounds=hps_usfa_con['usfa_con_cache_and_reuse_last_bounds'],
            usfa_con_constraint_target=hps_usfa_con['usfa_con_constraint_target'],
            usfa_con_upper_bound_lp_solver=hps_usfa_con['usfa_con_upper_bound_lp_solver'],
            usfa_con_ensemble_reduction_for_source_values=hps_usfa_con['usfa_con_ensemble_reduction_for_source_values'],
            usfa_con_ensemble_reduction_info_for_source_values=dict(
                psi_num_ensemble_groups=self.hyperparameters['psi_num_ensemble_groups'],
                ucb_weight=hps_usfa_con['usfa_con_ensemble_reduction_ucb_weight'],
            ),
            usfa_con_ensemble_reduction_for_source_to_target_values=hps_usfa_con['usfa_con_ensemble_reduction_for_source_to_target_values'],
            usfa_con_ensemble_reduction_info_for_source_to_target_values=dict(
                psi_num_ensemble_groups=self.hyperparameters['psi_num_ensemble_groups'],
                lcb_weight=hps_usfa_con['usfa_con_ensemble_reduction_lcb_weight'],
            ),
            usfa_con_source_task_vecs=getattr(
                ReacherTaskPresets, self.hyperparameters['reacher']['task_preset']).train_task_vecs(),
            usfa_con_coeff=LinearScheduler(
                hps_usfa_con['usfa_con_coeff_initial'],
                hps_usfa_con['usfa_con_coeff_final'],
                hps_usfa_con['usfa_con_coeff_start'],
                hps_usfa_con['usfa_con_coeff_end'],
                name="usfa_con_coeff",
                writer=writer,
            ),
            usfa_con_coeff_start=hps_usfa_con['usfa_con_coeff_start'],
            usfa_con_get_min_rewards=self.env.get_min_rewards,
            usfa_con_min_value_adjustment=hps_usfa_con['usfa_con_min_value_adjustment'],
            usfa_con_non_negative_coeffs_and_rewards=hps_usfa_con['usfa_con_non_negative_coeffs_and_rewards'],
            usfa_con_allow_grad=hps_usfa_con['usfa_con_allow_grad'],
            debug_usfa_con_exclusive_training=hps_usfa_con['debug_usfa_con_exclusive_training'],
            writer=writer,
        )

    def test_agent(self, default_gpi_source_policy_vecs, include_target_task_vec_for_gpi):
        psi = USFAPsiNetwork(
            model=copy.deepcopy(self.model),
            action_dim=self.n_actions,
            task_vec_dim=self.env.get_task_vec_dim(),
            normalize_policy_vecs=self.hyperparameters['normalize_policy_vecs'],
            ensemble_reduction_info=dict(
                psi_num_ensemble_groups=self.hyperparameters['psi_num_ensemble_groups'],
            ),
        )
        gpi_q = psi.construct_gpi_q_network(q_ensemble_reduction='mean')
        gpi_policy = GreedyPolicy(gpi_q, self.n_actions, epsilon=self.hyperparameters['test_exploration'])
        return USFATestAgent(
            gpi_policy=gpi_policy,
            default_gpi_source_policy_vecs=default_gpi_source_policy_vecs,
            include_target_task_vec_for_gpi=include_target_task_vec_for_gpi,
        )

    @property
    def target_vectors(self):
        if not hasattr(self, '_target_vectors'):
            #lin = torch.linspace(-10, 10, 11, device=self.hyperparameters['device'])
            #lin = torch.linspace(-10, 10, 21, device=self.hyperparameters['device'])
            lin = torch.linspace(-10, 10, 15, device=self.hyperparameters['device'])
            self._target_vectors = torch.cartesian_prod(lin, lin) / 10.0
        return self._target_vectors

    def test_agents(self, determine_dummy=lambda x: False):
        light_test_agents_dict = OrderedDict()
        other_test_agents_dict = OrderedDict()

        _model_copy_wrap = [None]
        def _get_model():
            # TODO: Is copying just once okay?
            if _model_copy_wrap[0] is None:
                _model_copy_wrap[0] = [m.eval() for m in copy.deepcopy(self.model)]
            return _model_copy_wrap[0]
            #return copy.deepcopy(self.model)
        @profile
        def _construct_gpi_q_network(q_ensemble_reduction):
            return USFAPsiNetwork(
                model=_get_model(),
                action_dim=self.n_actions,
                task_vec_dim=self.env.get_task_vec_dim(),
                normalize_policy_vecs=self.hyperparameters['normalize_policy_vecs'],
                ensemble_reduction_info=dict(
                    psi_num_ensemble_groups=self.hyperparameters['psi_num_ensemble_groups'],
                ),
            ).construct_gpi_q_network(q_ensemble_reduction=q_ensemble_reduction)
        @profile
        def _construct_gpi_policy(q_ensemble_reduction):
            return GreedyPolicy(
                q=_construct_gpi_q_network(q_ensemble_reduction),
                num_actions=self.n_actions,
                epsilon=self.hyperparameters['test_exploration'],
            )

        source_task_vecs = getattr(
            ReacherTaskPresets, self.hyperparameters['reacher']['task_preset']).train_task_vecs()

        if determine_dummy('light_test_agents'):
            # {{{
            for k in [
                'GPITargetTaskVec',
                'GPISourceTaskVecs',
                'GPISourceAndTargetTaskVecs',
            ]:
                light_test_agents_dict[k] = None

            if self.hyperparameters['psi_num_ensembles'] > 1:
                for k in [
                    'GPITargetTaskVec_EnsemMin',
                    'GPISourceTaskVecs_EnsemMin',
                    'GPISourceAndTargetTaskVecs_EnsemMin',
                    'GPITargetTaskVec_EnsemMax',
                    'GPISourceTaskVecs_EnsemMax',
                    'GPISourceAndTargetTaskVecs_EnsemMax',
                ]:
                    light_test_agents_dict[k] = None

                if (self.hyperparameters['psi_num_ensemble_groups'] is not None and
                        self.hyperparameters['psi_num_ensemble_groups'] > 1):
                    for k in [
                        'GPITargetTaskVec_EnsemMaxOfMins',
                        'GPISourceTaskVecs_EnsemMaxOfMins',
                        'GPISourceAndTargetTaskVecs_EnsemMaxOfMins',
                    ]:
                        light_test_agents_dict[k] = None
            # }}}
        else:
            light_test_agents_dict['GPITargetTaskVec'] = USFATestAgent(
                gpi_policy=_construct_gpi_policy(q_ensemble_reduction='mean'),
                default_gpi_source_policy_vecs=torch.zeros(
                    (0, self.env.get_task_vec_dim()), device=self.device),
                include_target_task_vec_for_gpi=True,
            )
            light_test_agents_dict['GPISourceTaskVecs'] = USFATestAgent(
                gpi_policy=_construct_gpi_policy(q_ensemble_reduction='mean'),
                default_gpi_source_policy_vecs=source_task_vecs,
                include_target_task_vec_for_gpi=False,
            )
            light_test_agents_dict['GPISourceAndTargetTaskVecs'] = USFATestAgent(
                gpi_policy=_construct_gpi_policy(q_ensemble_reduction='mean'),
                default_gpi_source_policy_vecs=source_task_vecs,
                include_target_task_vec_for_gpi=True,
            )

            if self.hyperparameters['psi_num_ensembles'] > 1:
                light_test_agents_dict['GPITargetTaskVec_EnsemMin'] = USFATestAgent(
                    gpi_policy=_construct_gpi_policy(q_ensemble_reduction='min'),
                    default_gpi_source_policy_vecs=torch.zeros(
                        (0, self.env.get_task_vec_dim()), device=self.device),
                    include_target_task_vec_for_gpi=True,
                )
                light_test_agents_dict['GPISourceTaskVecs_EnsemMin'] = USFATestAgent(
                    gpi_policy=_construct_gpi_policy(q_ensemble_reduction='min'),
                    default_gpi_source_policy_vecs=source_task_vecs,
                    include_target_task_vec_for_gpi=False,
                )
                light_test_agents_dict['GPISourceAndTargetTaskVecs_EnsemMin'] = USFATestAgent(
                    gpi_policy=_construct_gpi_policy(q_ensemble_reduction='min'),
                    default_gpi_source_policy_vecs=source_task_vecs,
                    include_target_task_vec_for_gpi=True,
                )

                light_test_agents_dict['GPITargetTaskVec_EnsemMax'] = USFATestAgent(
                    gpi_policy=_construct_gpi_policy(q_ensemble_reduction='max'),
                    default_gpi_source_policy_vecs=torch.zeros(
                        (0, self.env.get_task_vec_dim()), device=self.device),
                    include_target_task_vec_for_gpi=True,
                )
                light_test_agents_dict['GPISourceTaskVecs_EnsemMax'] = USFATestAgent(
                    gpi_policy=_construct_gpi_policy(q_ensemble_reduction='max'),
                    default_gpi_source_policy_vecs=source_task_vecs,
                    include_target_task_vec_for_gpi=False,
                )
                light_test_agents_dict['GPISourceAndTargetTaskVecs_EnsemMax'] = USFATestAgent(
                    gpi_policy=_construct_gpi_policy(q_ensemble_reduction='max'),
                    default_gpi_source_policy_vecs=source_task_vecs,
                    include_target_task_vec_for_gpi=True,
                )

                if (self.hyperparameters['psi_num_ensemble_groups'] is not None):

                    light_test_agents_dict['GPITargetTaskVec_EnsemMaxOfMins'] = USFATestAgent(
                        gpi_policy=_construct_gpi_policy(q_ensemble_reduction='max_of_group_mins'),
                        default_gpi_source_policy_vecs=torch.zeros(
                            (0, self.env.get_task_vec_dim()), device=self.device),
                        include_target_task_vec_for_gpi=True,
                    )
                    light_test_agents_dict['GPISourceTaskVecs_EnsemMaxOfMins'] = USFATestAgent(
                        gpi_policy=_construct_gpi_policy(q_ensemble_reduction='max_of_group_mins'),
                        default_gpi_source_policy_vecs=source_task_vecs,
                        include_target_task_vec_for_gpi=False,
                    )
                    light_test_agents_dict['GPISourceAndTargetTaskVecs_EnsemMaxOfMins'] = USFATestAgent(
                        gpi_policy=_construct_gpi_policy(q_ensemble_reduction='max_of_group_mins'),
                        default_gpi_source_policy_vecs=source_task_vecs,
                        include_target_task_vec_for_gpi=True,
                    )

        return dict(
            light_test_agents=light_test_agents_dict,
            other_test_agents=other_test_agents_dict,
        )

    def on_test_ex(self, results, writer, frame, episode, agent):
        pass
















def get_reacher_task_preset_naming_value_dict():
    value_dict = {
        'NegDistsPositiveD4Type1': 'NDPD4T1',
    }
    assert len(value_dict) == len(set(value_dict.values()))
    return value_dict



def get_exp_name(hyperparameters, mode='expname'):
    assert mode in ['expname', 'glob', 'regex_strict']

    parser = get_argparser()

    exp_name = ''
    if mode == 'expname':
        exp_name += f'sd{hyperparameters.seed:03d}_'
        if 'SLURM_JOB_ID' in os.environ:
            exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
        if 'SLURM_PROCID' in os.environ:
            exp_name += f'{os.environ["SLURM_PROCID"]}.'
        exp_name_prefix = exp_name
        if 'SLURM_RESTART_COUNT' in os.environ:
            exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
        exp_name += f'{g_start_time}'
    elif mode == 'glob':
        try:
            exp_name += f'sd{hyperparameters.seed:03d}_'
        except (KeyError, AttributeError) as e:
            pass
        exp_name += '*'
        exp_name_prefix = ''

        count_since_last_add = 0

        re_brackets = re.compile(r'(\[|\])')
    elif mode == 'regex_strict':
        exp_name += '^'
        try:
            exp_name += f'sd{hyperparameters.seed:03d}_'
        except (KeyError, AttributeError) as e:
            exp_name += f'sd[0-9]+_'
        exp_name += '(s_)?[^_]*'
        exp_name_prefix = ''
    else:
        assert False

    exp_name_abbrs = set()
    exp_name_arguments = set()

    def list_to_str(arg_list):
        return str(arg_list).replace(",", "|").replace(" ", "").replace("'", "")

    #def add_name(abbr, argument, value_func=None, value_dict=None, max_length=None, log_only_if_changed=False):
    def add_name(abbr, argument, value_func=None, value_dict=None, max_length=None, log_only_if_changed=True):
        nonlocal exp_name
        nonlocal count_since_last_add

        if argument in [
            'reacher.task_preset',
            'reacher.max_path_length',
            'reacher.phi_type',
            'reacher.obs_space_spec',
            'reacher.custom_xml_type',
        ]:
            assert not log_only_if_changed

        if mode == 'glob':
            last_count = count_since_last_add
            if not log_only_if_changed:
                count_since_last_add += 1

        if abbr is not None:
            assert abbr not in exp_name_abbrs, f'{abbr} {argument}'
            exp_name_abbrs.add(abbr)
        else:
            abbr = ''
        exp_name_arguments.add(argument)

        if not isinstance(argument, (list, tuple)):
            argument = argument.split('.')

        value = hyperparameters
        if mode == 'expname':
            for a in argument:
                value = getattr(value, a)
        elif mode in ['glob', 'regex_strict']:
            matched = False
            for a in argument:
                try:
                    value = getattr(value, a)
                    matched = True
                except (AttributeError, KeyError) as e:
                    matched = False
            if not matched:
                if mode == 'regex_strict' and not log_only_if_changed:
                    exp_name += f'_{abbr}[^a-z_]+'
                return
            if mode in ['regex_strict'] and matched and value == '*':
                # Special case handling.
                exp_name += f'_{abbr}[^a-z_]+'
                return
        else:
            assert False

        if log_only_if_changed:
            effective_parser = parser
            for a in argument[:-1]:
                effective_parser = [
                    group
                    for group in effective_parser._action_groups
                    if group.title == a
                ][0]
            if effective_parser.get_default(argument[-1]) == value:
                return

        if value_func is not None:
            value = value_func(value)

        if value_dict is not None:
            assert len(set(value_dict.values())) == len(value_dict)

        if isinstance(value, list):
            if value_dict is not None:
                value = [value_dict.get(v) for v in value]
            value = list_to_str(value)
        elif value_dict is not None:
            value = value_dict.get(value)

        if value is None:
            value = 'X'

        if max_length is not None:
            #value = str(value)[:max_length]
            value = str(value)[:(max_length - int(max_length/2))] + str(value)[-int(max_length/2):]

        if isinstance(value, str):
            value = value.replace('/', '-')
            value = value.upper()

            if mode == 'glob':
                value = re_brackets.sub(r'[\1]', value)

        if mode == 'expname':
            exp_name += f'_{abbr}{value}'
        elif mode == 'glob':
            #if count_since_last_add > 1:
            if last_count >= 1:
                exp_name += '_*'
            exp_name += f'_{abbr}{value}'
        elif mode == 'regex_strict':
            exp_name += re.escape(f'_{abbr}{value}')
        else:
            assert False

        if mode == 'glob':
            count_since_last_add = 0 

    add_name('pc', 'psi_ckpt_path_id', log_only_if_changed=True)

    add_name('pm', 'psi_model', log_only_if_changed=False, max_length=3)
    add_name('pe', 'psi_num_ensembles', log_only_if_changed=True)
    add_name('eg', 'psi_num_ensemble_groups', log_only_if_changed=True)
    add_name('em', 'psi_ensemble_train_even_mask', log_only_if_changed=True)
    add_name('og', 'psi_ensemble_train_with_own_group_samples_only', log_only_if_changed=True)
    add_name('pp', 'psi_ensemble_train_rand_mask_bernoulli_prob', log_only_if_changed=True)
    add_name('pa', 'psi_ensemble_next_actions_strategy', log_only_if_changed=True, value_dict={
        'ensemble_each': 'E',
    })
    add_name('pmh', 'psi_model_hidden', log_only_if_changed=False)
    add_name('lr', 'lr', log_only_if_changed=False)
    add_name('tuf', 'target_update_frequency', log_only_if_changed=True)
    add_name('uf', 'update_frequency', log_only_if_changed=True)
    add_name('mb', 'minibatch_size', log_only_if_changed=False)
    add_name('rb', 'replay_buffer_size', log_only_if_changed=False)
    add_name('rs', 'replay_start_size', log_only_if_changed=False)

    add_name('d', 'discount_factor', log_only_if_changed=False)
    add_name('e', 'exploration', log_only_if_changed=False)

    add_name('nte', 'num_test_episodes', log_only_if_changed=False)

    add_name('np', 'num_policy_samples_per_task', log_only_if_changed=True)
    add_name('tsf', 'train_task_vec_sampling_frequency', log_only_if_changed=True)
    add_name('ps', 'policy_sampler', log_only_if_changed=True, value_dict={
        'CondGaussian0': 'CG0',
        'CondGaussian1': 'CG1',
        'CondGaussian2': 'CG2',
        'CondGaussian3': 'CG3',
    })
    add_name('npv', 'normalize_policy_vecs', log_only_if_changed=False)

    add_name('gss', 'gpi_source_selection_state_pool_size', log_only_if_changed=False)

    #add_name('te', 'test_exploration', log_only_if_changed=False)
    assert parser.get_default('test_exploration') == 0.0

    add_name('tp', 'reacher.task_preset', log_only_if_changed=False,
        value_dict=get_reacher_task_preset_naming_value_dict())
    add_name('mpl', 'reacher.max_path_length', log_only_if_changed=False)
    add_name('ph', 'reacher.phi_type', log_only_if_changed=False, value_dict={
        'neg_dists_to_source_xys': 'ND',
    })
    add_name('os', 'reacher.obs_space_spec', log_only_if_changed=False, value_dict={
        'original_with_zero_target': 'OZ',
        'original_without_target_info': 'OW',
    })
    add_name('xm', 'reacher.custom_xml_type', log_only_if_changed=False, value_dict={
        None: 'X',
        'timestep_0.02': 'T02',
    })

    add_name('dft', 'reacher.debug_forced_test_task_vecs', log_only_if_changed=True)

    add_name('un', 'usfa_constraints.usfa_con_num_task_vecs', log_only_if_changed=False)
    add_name('um', 'usfa_constraints.usfa_con_mode', log_only_if_changed=False, value_dict={
        'both': 'B',
        'lower': 'L',
        'upper': 'U',
    })

    add_name('utf', 'usfa_constraints.usfa_con_train_frequency', log_only_if_changed=False)
    add_name('uc', 'usfa_constraints.usfa_con_cache_and_reuse_last_bounds', log_only_if_changed=False)
    add_name('uct', 'usfa_constraints.usfa_con_constraint_target', log_only_if_changed=False, value_dict={
        'ensemble_mean': 'M',
        'ensemble_each': 'E',
    })

    add_name('uus', 'usfa_constraints.usfa_con_upper_bound_lp_solver', log_only_if_changed=False, value_dict={
        'cvxpylayers': 'CL',
        'special_case_pos_one_hots': 'PO',
        'special_case_pos_and_neg_one_hots_with_nonneg_coeffs': 'PN',
    })
    add_name('urs', 'usfa_constraints.usfa_con_ensemble_reduction_for_source_values', log_only_if_changed=False, value_dict={
        'mean': 'A',
        'max': 'X',
        'ucb': 'U',
    })
    add_name('uu', 'usfa_constraints.usfa_con_ensemble_reduction_ucb_weight', log_only_if_changed=True)
    add_name('urt', 'usfa_constraints.usfa_con_ensemble_reduction_for_source_to_target_values', log_only_if_changed=False, value_dict={
        'mean': 'A',
        'min': 'N',
        'lcb': 'L',
    })
    add_name('ul', 'usfa_constraints.usfa_con_ensemble_reduction_lcb_weight', log_only_if_changed=True)
    add_name('uci', 'usfa_constraints.usfa_con_coeff_initial', log_only_if_changed=False)
    add_name('ucf', 'usfa_constraints.usfa_con_coeff_final', log_only_if_changed=False)
    add_name('ucs', 'usfa_constraints.usfa_con_coeff_start', log_only_if_changed=False)
    add_name('uce', 'usfa_constraints.usfa_con_coeff_end', log_only_if_changed=False)
    add_name('uca', 'usfa_constraints.usfa_con_min_value_adjustment', log_only_if_changed=True, value_dict={
        None: 'X',
        'per_source_task': 'PT',
        'per_sample': 'PS',
    })
    add_name('ucn', 'usfa_constraints.usfa_con_non_negative_coeffs_and_rewards', log_only_if_changed=False)
    add_name('ug', 'usfa_constraints.usfa_con_allow_grad', log_only_if_changed=False)

    add_name('dux', 'usfa_constraints.debug_usfa_con_exclusive_training', log_only_if_changed=True)

    if True:
        # Check missing changed arguments
        change_allowed_args = {
            'verbose', 'log_to_stdout',
            'seed', 'run_group',
            'frames', 'test_frequency_in_episodes', 'test_light_frequency_in_episodes', 'save_frequency_in_updates',
            'test_exploration', 'debug_no_initial_test',
            'psi_ckpt_path',
        }
        for key in vars(hyperparameters):
            if key in change_allowed_args:
                continue
            if key not in exp_name_arguments and parser.get_default(key) != getattr(hyperparameters, key):
                raise Exception(f'{key} changed but not in exp_name')

    if mode == 'glob':
        if count_since_last_add >= 1:
            exp_name += '_*'

    if mode == 'expname':
        assert len(exp_name) <= os.pathconf('/', 'PC_NAME_MAX')

    if mode == 'regex_strict':
        exp_name += '$'

    return exp_name, exp_name_prefix



def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--log_to_stdout', default=False, action='store_true')
    parser.add_argument('--debug_profile', default=False, action='store_true')
    parser.add_argument('--debug_no_initial_test', type=int, default=0, choices=[0, 1])

    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run_group', type=str, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frames', type=int, default=4000000)
    parser.add_argument('--num_test_episodes', type=int, default=20)
    parser.add_argument('--test_frequency_in_episodes', type=int, default=100)
    parser.add_argument('--test_light_frequency_in_episodes', type=int, default=1250)
    parser.add_argument('--save_frequency_in_updates', type=int, default=50000)

    parser.add_argument('--psi_ckpt_path', type=str, default=None)
    parser.add_argument('--psi_ckpt_path_id', type=str, default=None)

    parser.add_argument('--psi_model', type=str, default='fc_relu_psi1',
                        choices=exp_utils.get_public_attrs(USFAPsiModelPresets))
    parser.add_argument('--psi_num_ensembles', type=int, default=1)
    parser.add_argument('--psi_num_ensemble_groups', type=int, default=None)
    parser.add_argument('--psi_ensemble_train_even_mask', type=int, default=0, choices=[0, 1])
    parser.add_argument('--psi_ensemble_train_with_own_group_samples_only', type=int, default=0, choices=[0, 1])
    parser.add_argument('--psi_ensemble_train_rand_mask_bernoulli_prob', type=float, default=None)
    parser.add_argument('--psi_ensemble_next_actions_strategy', type=str, default='ensemble_each',
                        choices=['ensemble_each'])
    parser.add_argument('--psi_model_hidden', type=int, nargs='*', default=[128, 256])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--target_update_frequency', type=int, default=500)
    parser.add_argument('--update_frequency', type=int, default=1)
    parser.add_argument('--minibatch_size', type=int, default=32)
    parser.add_argument('--replay_buffer_size', type=int, default=2048)
    parser.add_argument('--replay_start_size', type=int, default=2048)

    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--exploration', type=float, default=0.1)

    parser.add_argument('--num_policy_samples_per_task', type=int, default=5)
    parser.add_argument('--train_task_vec_sampling_frequency', type=int, default=1)
    parser.add_argument('--policy_sampler', type=str, default='CondGaussian1',
                        choices=exp_utils.get_public_attrs(USFAPolicySamplerPresets))
    parser.add_argument('--normalize_policy_vecs', type=int, default=0,
                        choices=[0, 1])

    parser.add_argument('--gpi_source_selection_state_pool_size', type=int, default=None)

    parser.add_argument('--test_exploration', type=float, default=0.0)

    reacher = parser.add_argument_group('reacher')
    reacher.add_argument('--task_preset', type=str, default='NegDistsPositiveD4Type1',
                         choices=exp_utils.get_public_attrs(ReacherTaskPresets))
    reacher.add_argument('--max_path_length', type=int, default=500)
    reacher.add_argument('--phi_type', type=str, default='neg_dists_to_source_xys',
                         choices=[
                             'neg_dists_to_source_xys',
                         ])
    reacher.add_argument('--obs_space_spec', type=str, default='original_with_zero_target',
                         choices=[
                             'original_with_zero_target',
                             'original_without_target_info',
                         ])
    reacher.add_argument('--custom_xml_type', type=str, default=None,
                         choices=[
                             None,
                             'timestep_0.02',
                         ])
    reacher.add_argument('--debug_forced_test_task_vecs', type=float, nargs='*', default=None)

    usfa_constraints = parser.add_argument_group('usfa_constraints')
    usfa_constraints.add_argument('--usfa_con_num_task_vecs', type=int, default=0)
    usfa_constraints.add_argument('--usfa_con_mode', type=str, default='both',
                                  choices=['both', 'lower', 'upper'])
    usfa_constraints.add_argument('--usfa_con_train_frequency', type=int, default=1)
    usfa_constraints.add_argument('--usfa_con_cache_and_reuse_last_bounds', type=int, default=None,
                                  choices=[1])
                                  #choices=[0, 1])
    usfa_constraints.add_argument('--usfa_con_constraint_target', type=str, default='ensemble_mean',
                                  choices=['ensemble_mean'])
    usfa_constraints.add_argument('--usfa_con_upper_bound_lp_solver', type=str, default='cvxpylayers',
                                  choices=['cvxpylayers', 'special_case_pos_one_hots', 'special_case_pos_and_neg_one_hots_with_nonneg_coeffs'])
    usfa_constraints.add_argument('--usfa_con_ensemble_reduction_for_source_values', type=str, default='mean',
                                  choices=['mean'])
    usfa_constraints.add_argument('--usfa_con_ensemble_reduction_ucb_weight', type=float, default=None)
    usfa_constraints.add_argument('--usfa_con_ensemble_reduction_for_source_to_target_values', type=str, default='mean',
                                  choices=['mean'])
    usfa_constraints.add_argument('--usfa_con_ensemble_reduction_lcb_weight', type=float, default=None)
    usfa_constraints.add_argument('--usfa_con_coeff_initial', type=float, default=0.0)
    usfa_constraints.add_argument('--usfa_con_coeff_final', type=float, default=0.0)
    usfa_constraints.add_argument('--usfa_con_coeff_start', type=int, default=0)
    usfa_constraints.add_argument('--usfa_con_coeff_end', type=int, default=0)
    usfa_constraints.add_argument('--usfa_con_min_value_adjustment', type=str, default=None,
                                  choices=[None, 'per_source_task', 'per_sample'])
    usfa_constraints.add_argument('--usfa_con_non_negative_coeffs_and_rewards', type=int, default=0, choices=[0, 1])
    usfa_constraints.add_argument('--usfa_con_allow_grad', type=int, default=0, choices=[0, 1])
    usfa_constraints.add_argument('--debug_usfa_con_exclusive_training', type=int, default=0, choices=[0, 1])

    return parser

def get_hyperparameters(defaults, args=None):
    parser = get_argparser()

    args = parser.parse_args(args)

    hyperparameters = exp_utils.AttributeDict(copy.copy(defaults))

    for group in parser._action_groups:
        group_dict = exp_utils.AttributeDict({ a.dest: getattr(args, a.dest, None) for a in group._group_actions })
        if group.title == 'positional arguments':
            assert len(group_dict) == 0
        elif group.title == 'optional arguments':
            del group_dict['help']
            hyperparameters.update(group_dict)
        else:
            hyperparameters[group.title] = group_dict

    return hyperparameters

def _on_dirs_created(dirs_info):
    if 'group_dir' in dirs_info:
        print(os.path.join(dirs_info['group_dir'], os.path.basename(dirs_info['log_dir'])))
    else:
        print(dirs_info['log_dir'])

    try:
        fd = os.open(
            os.path.join(dirs_info['log_dir'], os.path.basename(g_file_path)),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as dest:
            with open(g_file_path) as src:
                shutil.copyfileobj(src, dest)
    except Exception as e:
        pass

    if 'group_dir' in dirs_info:
        try:
            fd = os.open(
                os.path.join(dirs_info['group_dir'], os.path.basename(g_file_path)),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as dest:
                with open(g_file_path) as src:
                    shutil.copyfileobj(src, dest)
        except Exception as e:
            pass

def main():
    hyperparameters = get_hyperparameters({})

    logger_main.set_no_stdout(not hyperparameters['log_to_stdout'])
    logger_sub.set_no_stdout(not hyperparameters['log_to_stdout'])

    exp_utils.set_seed(hyperparameters['seed'])
    ReacherTaskPresets._device = hyperparameters['device']
    ReacherTaskPresets._reacher_hyperparameters = hyperparameters['reacher']
    ReacherTaskPresets._usfa_con_hyperparameters = hyperparameters['usfa_constraints']

    reacher_hps = exp_utils.AttributeDict(copy.copy(dict(hyperparameters['reacher'])))
    reacher_hps['task_vec_sampler'] = getattr(
        ReacherTaskPresets, reacher_hps['task_preset']).test_task_vec_sampler
    reacher_hps['source_xys'] = getattr(
        ReacherTaskPresets, reacher_hps['task_preset']).source_xys()

    for v in getattr(ReacherTaskPresets, reacher_hps['task_preset']).train_task_vecs():
        assert v.size(0) == getattr(
            ReacherTaskPresets, reacher_hps['task_preset']).get_task_vec_dim()
    for v in getattr(ReacherTaskPresets, reacher_hps['task_preset']).test_task_vecs():
        assert v.size(0) == getattr(
            ReacherTaskPresets, reacher_hps['task_preset']).get_task_vec_dim()

    del reacher_hps['task_preset']

    env = ReacherWrappedEnv(
        **reacher_hps,
        device=hyperparameters['device'],
    )

    preset_builder = PresetBuilder(
        'usfa',
        { 'hyperparameters': hyperparameters },
        USFAReacherPreset,
        device=hyperparameters['device'],
        env=env)
    preset = preset_builder.build()
    experiment = SingleEnvExperimentEx(
        preset,
        env,
        train_steps=hyperparameters['frames'],
        num_test_episodes=hyperparameters['num_test_episodes'],
        test_frequency_in_episodes=hyperparameters['test_frequency_in_episodes'],
        test_light_frequency_in_episodes=hyperparameters['test_light_frequency_in_episodes'],
        save_frequency_in_episodes=None,
        logdir='exp',
        dir_name=get_exp_name(hyperparameters)[0],
        run_group=hyperparameters['run_group'],
        quiet=not hyperparameters['verbose'],
        render=False,
        write_loss=True,
        writer="tensorboard",
        write_to_tensorboard_events=False,
        on_dirs_created=_on_dirs_created,
    )

    logger_main.log(pprint.PrettyPrinter(indent=2).pformat(hyperparameters))
    if not hyperparameters['log_to_stdout']:
        print(pprint.PrettyPrinter(indent=2).pformat(hyperparameters))

    experiment.train(
        frames=hyperparameters['frames'],
        debug_no_initial_test=hyperparameters['debug_no_initial_test'],
    )
    #experiment.save()
    #experiment.test(episodes=test_episodes)
    experiment.close()



if __name__ == "__main__":
    main()

