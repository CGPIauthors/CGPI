import copy
import socket
import sys
from timeit import default_timer as timer
import numpy as np
import torch
from torch.nn import functional as F
from all.agents._agent import Agent
from all.core.state import State, StateArray
from all.logging import DummyWriter
from all.memory import ExperienceReplayBuffer
from all.nn import weighted_mse_loss
from alli.approximation.optimal_value_bounder import compute_optimal_value_bounds
from alli.approximation.usfa_network import USFAGPIQNetwork, USFAPsiNetwork
from alli.core import state_util
from alli.core.tensor_util import unsqueeze_and_expand
from rlkit.core.logging import logger_main, logger_sub

if '--debug_profile' in sys.argv:
    from rlkit.core.logging import profile
else:
    profile = lambda func: func

g_curr_hostname = socket.gethostname()

class USFA(Agent):
    def __init__(self,
                 psi: USFAPsiNetwork,
                 #policy,
                 gpi_policy,
                 replay_buffer: ExperienceReplayBuffer,
                 phi,
                 task_vec_dim,
                 action_dim,
                 train_task_vec_sampler,
                 train_task_vec_sampling_frequency,
                 policy_vec_sampler,
                 num_policy_samples_per_task,
                 #use_gpi_samples_for_training=True,
                 discount_factor=0.99,
                 max_path_length=50,
                 loss=weighted_mse_loss,
                 minibatch_size=32,
                 replay_start_size=5000,
                 psi_num_ensembles=1,
                 psi_num_ensemble_groups=None,
                 psi_ensemble_train_even_mask=False,
                 psi_ensemble_train_with_own_group_samples_only=False,
                 psi_ensemble_train_rand_mask_bernoulli_prob=None,
                 psi_ensemble_next_actions_strategy='ensemble_each',
                 update_frequency=1,
                 state_pool_size=None,
                 usfa_con_task_vec_sampler=None,
                 usfa_con_num_task_vecs=None,
                 usfa_con_mode='both',
                 usfa_con_train_frequency=1,
                 usfa_con_cache_and_reuse_last_bounds=1,
                 usfa_con_constraint_target='ensemble_mean',
                 usfa_con_upper_bound_lp_solver='cvxpylayers',
                 usfa_con_ensemble_reduction_for_source_values=None,
                 usfa_con_ensemble_reduction_info_for_source_values=dict(),
                 usfa_con_ensemble_reduction_for_source_to_target_values=None,
                 usfa_con_ensemble_reduction_info_for_source_to_target_values=dict(),
                 #usfa_con_min_alpha=None,
                 #usfa_con_max_alpha=None,
                 usfa_con_source_task_vecs=None,
                 usfa_con_coeff=0.0,
                 usfa_con_coeff_start=0,
                 usfa_con_get_min_rewards=None,
                 usfa_con_min_value_adjustment=None,
                 usfa_con_non_negative_coeffs_and_rewards=False,
                 usfa_con_allow_grad=False,
                 debug_usfa_con_exclusive_training=0,
                 debug_usfa_con_ignore_lower_bound=0,
                 debug_usfa_con_ignore_upper_bound=0,
                 writer=DummyWriter(),
                 ):
        # objects
        self.psi = psi
        #self.device = self.psi.model.device
        self.device = self.psi.device
        self.q = self.psi.construct_q_network()
        #self.policy = policy
        self.gpi_policy = gpi_policy
        self.replay_buffer = replay_buffer
        self.phi = phi
        self.task_vec_dim = task_vec_dim
        self.action_dim = action_dim
        self.train_task_vec_sampler = train_task_vec_sampler
        self.train_task_vec_sampling_frequency = train_task_vec_sampling_frequency
        self.policy_vec_sampler = policy_vec_sampler
        self.num_policy_samples_per_task = num_policy_samples_per_task
        #self.use_gpi_samples_for_training = use_gpi_samples_for_training

        self.loss = loss
        # hyperparameters
        self.replay_start_size = replay_start_size
        self.psi_num_ensembles = psi_num_ensembles
        self.psi_num_ensemble_groups = psi_num_ensemble_groups
        if self.psi_num_ensemble_groups is not None:
            assert psi_num_ensembles % psi_num_ensemble_groups == 0
            assert (psi_ensemble_train_even_mask + psi_ensemble_train_with_own_group_samples_only) == 1, (
                'With `psi_num_ensemble_groups`, '
                'the same group should use the same training samples and '
                'different groups should use different samples.')
        self.psi_ensemble_train_even_mask = psi_ensemble_train_even_mask
        if self.psi_ensemble_train_even_mask:
            if self.psi_num_ensemble_groups is not None:
                assert (minibatch_size % psi_num_ensemble_groups == 0)
            else:
                assert (minibatch_size % psi_num_ensembles == 0)
        self.psi_ensemble_train_with_own_group_samples_only = psi_ensemble_train_with_own_group_samples_only
        if self.psi_ensemble_train_with_own_group_samples_only:
            assert psi_num_ensemble_groups is not None
        self.psi_ensemble_train_rand_mask_bernoulli_prob = psi_ensemble_train_rand_mask_bernoulli_prob
        self.psi_ensemble_next_actions_strategy = psi_ensemble_next_actions_strategy
        self.update_frequency = update_frequency
        self.state_pool_size = state_pool_size
        self.state_pool = []
        self.state_count = 0
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.max_path_length = max_path_length

        self.usfa_con_task_vec_sampler = usfa_con_task_vec_sampler
        self.usfa_con_num_task_vecs = usfa_con_num_task_vecs
        self.usfa_con_mode = usfa_con_mode
        self.usfa_con_train_frequency = usfa_con_train_frequency
        self.usfa_con_cache_and_reuse_last_bounds = usfa_con_cache_and_reuse_last_bounds
        if usfa_con_cache_and_reuse_last_bounds:
            assert not usfa_con_allow_grad
            self._usfa_con_cached_data = None
        self.usfa_con_constraint_target = usfa_con_constraint_target
        assert self.usfa_con_constraint_target in ['ensemble_mean', 'ensemble_each']
        self.usfa_con_upper_bound_lp_solver = usfa_con_upper_bound_lp_solver
        assert self.usfa_con_upper_bound_lp_solver in [
            'cvxpylayers',
            'special_case_pos_one_hots',
            'special_case_pos_and_neg_one_hots_with_nonneg_coeffs',
        ]
        self.usfa_con_ensemble_reduction_for_source_values = usfa_con_ensemble_reduction_for_source_values
        assert self.usfa_con_ensemble_reduction_for_source_values in ['mean', 'max', 'ucb']
        self.usfa_con_ensemble_reduction_info_for_source_values = usfa_con_ensemble_reduction_info_for_source_values
        self.usfa_con_ensemble_reduction_for_source_to_target_values = usfa_con_ensemble_reduction_for_source_to_target_values
        assert self.usfa_con_ensemble_reduction_for_source_to_target_values in ['mean', 'min', 'lcb']
        self.usfa_con_ensemble_reduction_info_for_source_to_target_values = usfa_con_ensemble_reduction_info_for_source_to_target_values
        #self.usfa_con_min_alpha = usfa_con_min_alpha
        #self.usfa_con_max_alpha = usfa_con_max_alpha
        self.usfa_con_source_task_vecs = usfa_con_source_task_vecs
        self.usfa_con_coeff = usfa_con_coeff
        self.usfa_con_coeff_start = usfa_con_coeff_start
        self.usfa_con_get_min_rewards = usfa_con_get_min_rewards
        self.usfa_con_min_value_adjustment = usfa_con_min_value_adjustment
        self.usfa_con_non_negative_coeffs_and_rewards = usfa_con_non_negative_coeffs_and_rewards
        self.usfa_con_allow_grad = usfa_con_allow_grad
        self.debug_usfa_con_exclusive_training = debug_usfa_con_exclusive_training
        self.debug_usfa_con_ignore_lower_bound = debug_usfa_con_ignore_lower_bound
        self.debug_usfa_con_ignore_upper_bound = debug_usfa_con_ignore_upper_bound

        self.writer = writer

        # private
        self._state = None
        self._action = None
        self._frames_seen = 0

        self._last_loss_usfa_psi_only = 0.0
        self._last_loss_usfa_con = 0.0
        self._last_loss_usfa_con_scaled = 0.0
        self._last_loss_usfa_full = 0.0

        self._last_time_optimal_value_bounding = 0.0
        self._n_last_vals_time_optimal_value_bounding = []

        self._last_lower_bound_violation_ratio = 0.0
        self._last_upper_bound_violation_ratio = 0.0
        self._last_incompatible_lower_upper_bound_ratio = 0.0

        self._last_summary_ensemble_q_std = (0.0, 0.0, 0.0, 0.0)
        self._last_summary_ensemble_q_diff_max_min = (0.0, 0.0, 0.0, 0.0)

        self._last_summary_usfa_con_lower = (0.0, 0.0, 0.0, 0.0)
        self._last_summary_usfa_con_upper = (0.0, 0.0, 0.0, 0.0)
        self._last_summary_usfa_con_upper_m_lower = (0.0, 0.0, 0.0, 0.0)
        self._last_summary_usfa_con_q_m_lower = (0.0, 0.0, 0.0, 0.0)
        self._last_summary_usfa_con_upper_m_q = (0.0, 0.0, 0.0, 0.0)

    def _add_state_to_state_pool(self, state):
        if not self.state_pool_size:
            return
        state = copy.copy(state)
        self.state_count += 1
        if len(self.state_pool) < self.state_pool_size:
            self.state_pool.append(state)
        else:
            idx = np.random.randint(0, self.state_count)
            if idx < self.state_pool_size:
                self.state_pool[idx] = state

    @profile
    def act(self, state):
        assert len(state.shape) == 0
        self._add_state_to_state_pool(state)

        def _select_new_ensemble_for_gpi_policy_for_sampling():
            self._curr_selected_group_idx = torch.randint(0, self.psi_num_ensemble_groups, (1,)).item()
            assert isinstance(self.gpi_policy.q, USFAGPIQNetwork)
            self.gpi_policy.q.model.q_ensemble_reduction_info.update(dict(
                selected_group_idx=self._curr_selected_group_idx,
            ))

        if self.psi_num_ensemble_groups is not None and self._state is None:
            _select_new_ensemble_for_gpi_policy_for_sampling()

        if self._state is not None:
            phi_feat = self.phi(self._state['observation'], self._action, state['observation'])
            state['reward'] = torch.inner(phi_feat, self._state['task_vec']).item()
            reward_normalizer = self._state['task_vec'].mean().item()
            if reward_normalizer == 0.0:
                reward_normalizer = 1e-6
            state['reward_normalized'] = state['reward'] / reward_normalizer

            s = copy.copy(state)
            s['task_vec'] = self._state['task_vec']
            #s['policy_vec'] = self._state['policy_vec']
            s['gpi_source_policy_vecs'] = self._state['gpi_source_policy_vecs']

            if self.psi_ensemble_train_with_own_group_samples_only:
                assert self.psi_num_ensemble_groups is not None
                train_mask = torch.repeat_interleave(
                    F.one_hot(
                        torch.as_tensor(self._curr_selected_group_idx, device=self.device),
                        self.psi_num_ensemble_groups,
                    ).to(torch.float),
                    self.psi_num_ensembles // self.psi_num_ensemble_groups,
                    dim=0,
                )
                self._state['ensemble_train_mask'] = train_mask
                s['ensemble_train_mask'] = train_mask
            elif (not self.psi_ensemble_train_even_mask) and self.psi_ensemble_train_rand_mask_bernoulli_prob is not None:
                rand_mask = torch.bernoulli(
                    (torch.ones(self.psi_num_ensembles, device=self.device) *
                        self.psi_ensemble_train_rand_mask_bernoulli_prob),
                )
                self._state['ensemble_train_mask'] = rand_mask
                s['ensemble_train_mask'] = rand_mask

            self.replay_buffer.store(self._state, self._action, s)

        if state['step_count'] % self.train_task_vec_sampling_frequency == 0:
            self.curr_task_vec = self.train_task_vec_sampler()
        self.curr_task_vec = self.train_task_vec_sampler()
        state['task_vec'] = self.curr_task_vec
        #state['policy_vec'] = self.policy_vec_sampler(self.curr_task_vec)
        state['gpi_source_policy_vecs'] = self.policy_vec_sampler(
            self.curr_task_vec, (self.num_policy_samples_per_task,))
        self._train()

        self._state = state
        if self.psi_num_ensemble_groups is not None and self._state.done:
            _select_new_ensemble_for_gpi_policy_for_sampling()
        self._action = self.gpi_policy.no_grad(state)
        return self._action

    def eval(self, state):
        return self.gpi_policy.eval(state)

    def _prepare_minibatch_for_training(self, minibatch):
        (states, actions, rewards, next_states, weights) = minibatch
        if not isinstance(states, StateArray):
            states = State.array([states])
        if not isinstance(next_states, StateArray):
            next_states = State.array([next_states])

        if self.psi_ensemble_train_with_own_group_samples_only:
            assert self.psi_num_ensemble_groups is not None
            pass
        elif self.psi_ensemble_train_even_mask:
            if self.psi_num_ensemble_groups is not None:
                even_mask = torch.repeat_interleave(
                    torch.repeat_interleave(
                        torch.eye(self.psi_num_ensemble_groups, device=self.device),
                        self.minibatch_size // self.psi_num_ensemble_groups,
                        dim=0,
                    ),
                    self.psi_num_ensembles // self.psi_num_ensemble_groups,
                    dim=1,
                )
            else:
                even_mask = torch.repeat_interleave(
                    torch.eye(self.psi_num_ensembles, device=self.device),
                    self.minibatch_size // self.psi_num_ensembles,
                    dim=0,
                )
            states['ensemble_train_mask'] = even_mask
            next_states['ensemble_train_mask'] = even_mask

        states_policy_vecs = states['gpi_source_policy_vecs'].view(
            *states.shape[:-1],
            states.shape[-1] * self.num_policy_samples_per_task,
            *states['gpi_source_policy_vecs'].size()[len(states.shape)+1:],
        )
        states = state_util.repeat_states(states, self.num_policy_samples_per_task)
        states['policy_vec'] = states_policy_vecs

        next_states_policy_vecs = next_states['gpi_source_policy_vecs'].view(
            *next_states.shape[:-1],
            next_states.shape[-1] * self.num_policy_samples_per_task,
            *next_states['gpi_source_policy_vecs'].size()[len(next_states.shape)+1:],
        )
        next_states = state_util.repeat_states(next_states, self.num_policy_samples_per_task)
        next_states['policy_vec'] = next_states_policy_vecs

        # Use the sampled z_i vectors as both input to psi() and the vector to take the inner product with,
        # because the argmax'd actions need to be used for the (psi-)Bellman optimality update.
        next_states['task_vec'] = next_states_policy_vecs

        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device)
        actions = torch.repeat_interleave(actions, self.num_policy_samples_per_task, dim=0)
        if isinstance(rewards, list):
            rewards = torch.tensor(rewards, device=self.device)
        rewards = torch.repeat_interleave(rewards, self.num_policy_samples_per_task, dim=0)
        if isinstance(weights, list):
            weights = torch.tensor(weights, device=self.device)
        weights = torch.repeat_interleave(weights, self.num_policy_samples_per_task, dim=0)

        return (states, actions, rewards, next_states, weights)

    def _compute_q_values(self, states, task_vecs, ensemble_reduction):
        num_states = np.prod(states.shape)
        observations = unsqueeze_and_expand(
            states['observation'].view(
                num_states,
                *states['observation'].size()[len(states.shape):]),
            dim=0,
            num_repeat=task_vecs.size(0))
        observations = observations.reshape(
            task_vecs.size(0) * num_states,
            *observations.size()[2:])
        policy_vecs = unsqueeze_and_expand(
            task_vecs,
            dim=1,
            num_repeat=num_states)
        policy_vecs = policy_vecs.reshape(
            task_vecs.size(0) * num_states,
            *policy_vecs.size()[2:])

        q_values = self.q(
            StateArray(
                {
                    'observation': observations,
                    'policy_vec': policy_vecs,
                    'task_vec': policy_vecs,
                },
                shape=(task_vecs.size(0) * num_states,),
                device=self.device,
            ),
            ensemble_reduction=ensemble_reduction,
        )
        if ensemble_reduction == 'none':
            q_values = q_values.view(q_values.size(0), task_vecs.size(0), num_states, q_values.size(-1))
        else:
            q_values = q_values.view(task_vecs.size(0), num_states, q_values.size(-1))
        return q_values

    def _get_summary(self, v):
        if v is not None:
            v_np = v.detach().cpu().numpy()
            return (np.mean(v_np), np.std(v_np), np.max(v_np), np.min(v_np))
        return (0.0, 0.0, 0.0, 0.0)

    @profile
    def _get_usfa_constraints_loss(self, states):
        if not self.usfa_con_num_task_vecs:
            return 0.0

        if self._frames_seen < self.usfa_con_coeff_start:
            return 0.0

        right_on_train_frequency = (self._frames_seen % self.usfa_con_train_frequency == 0)

        if (not right_on_train_frequency) and (not self.usfa_con_cache_and_reuse_last_bounds):
            assert False, 'Set usfa_con_cache_and_reuse_last_bounds to 1 for the consistency of the loss'
            return 0.0

        if (not right_on_train_frequency) and self._usfa_con_cached_data is None:
            logger_main.log(f'Skipping USFA constraint loss as there is no cached data')
            return 0.0

        if right_on_train_frequency:
            target_task_vecs = self.usfa_con_task_vec_sampler((self.usfa_con_num_task_vecs,))

            debug_sanity_check = False

            start_time = timer()
            bounds = compute_optimal_value_bounds(
                psi=self.psi,
                source_task_vecs=self.usfa_con_source_task_vecs,
                states=states,
                target_task_vecs=target_task_vecs,
                source_task_min_rewards=self.usfa_con_get_min_rewards(self.usfa_con_source_task_vecs),
                #min_alpha=self.usfa_con_min_alpha,
                #max_alpha=self.usfa_con_max_alpha,
                discount_factor=self.discount_factor,
                max_path_length=self.max_path_length,
                device=self.device,
                solver=self.usfa_con_upper_bound_lp_solver,
                ensemble_reduction_for_source_values=self.usfa_con_ensemble_reduction_for_source_values,
                ensemble_reduction_info_for_source_values=self.usfa_con_ensemble_reduction_info_for_source_values,
                ensemble_reduction_for_source_to_target_values=self.usfa_con_ensemble_reduction_for_source_to_target_values,
                ensemble_reduction_info_for_source_to_target_values=self.usfa_con_ensemble_reduction_info_for_source_to_target_values,
                use_v_values=False,
                min_value_adjustment=self.usfa_con_min_value_adjustment,
                non_negative_coeffs_and_rewards=self.usfa_con_non_negative_coeffs_and_rewards,
                #detach=True,
                detach=(not self.usfa_con_allow_grad),
                compute_targets={
                    'both': ['optimal_lower_bounds', 'optimal_upper_bounds'],
                    'lower': ['optimal_lower_bounds'],
                    'upper': ['optimal_upper_bounds'],
                }[self.usfa_con_mode],

                debug_sanity_check=debug_sanity_check,
            )
            self._last_time_optimal_value_bounding = (timer() - start_time)

            if debug_sanity_check:
                # {{{
                bounds2 = compute_optimal_value_bounds(
                    psi=self.psi,
                    source_task_vecs=self.usfa_con_source_task_vecs,
                    states=states,
                    target_task_vecs=target_task_vecs,
                    source_task_min_rewards=self.usfa_con_get_min_rewards(self.usfa_con_source_task_vecs),
                    #min_alpha=self.usfa_con_min_alpha,
                    #max_alpha=self.usfa_con_max_alpha,
                    discount_factor=self.discount_factor,
                    max_path_length=self.max_path_length,
                    device=self.device,
                    solver='cvxpylayers',
                    ensemble_reduction_for_source_values=self.usfa_con_ensemble_reduction_for_source_values,
                    ensemble_reduction_info_for_source_values=self.usfa_con_ensemble_reduction_info_for_source_values,
                    ensemble_reduction_for_source_to_target_values=self.usfa_con_ensemble_reduction_for_source_to_target_values,
                    ensemble_reduction_info_for_source_to_target_values=self.usfa_con_ensemble_reduction_info_for_source_to_target_values,
                    use_v_values=False,
                    min_value_adjustment=self.usfa_con_min_value_adjustment,
                    non_negative_coeffs_and_rewards=self.usfa_con_non_negative_coeffs_and_rewards,
                    #detach=True,
                    detach=(not self.usfa_con_allow_grad),
                    compute_targets={
                        'both': ['optimal_lower_bounds', 'optimal_upper_bounds'],
                        'lower': ['optimal_lower_bounds'],
                        'upper': ['optimal_upper_bounds'],
                    }[self.usfa_con_mode],

                    debug_sanity_check=debug_sanity_check,
                )

                print('-----------------------------')
                print(bounds['optimal_upper_bounds'])
                print(bounds2['optimal_upper_bounds'])
                print(bounds['optimal_upper_bounds'] - bounds2['optimal_upper_bounds'])
                print((bounds['optimal_upper_bounds'] - bounds2['optimal_upper_bounds']).max())
                print((bounds['optimal_upper_bounds'] - bounds2['optimal_upper_bounds']).min())
                if self.usfa_con_upper_bound_lp_solver in ['special_case_pos_one_hots']:
                    assert torch.allclose(bounds['optimal_upper_bounds'], bounds2['optimal_upper_bounds'], rtol=1e-3, atol=1e-3)
                print('=============================')
                # }}}

            self._n_last_vals_time_optimal_value_bounding.append(self._last_time_optimal_value_bounding)
            if len(self._n_last_vals_time_optimal_value_bounding) >= 20:
                logger_main.log(f'[{g_curr_hostname}] avg. time bounding: {np.mean(self._n_last_vals_time_optimal_value_bounding)}')
                self._n_last_vals_time_optimal_value_bounding = []

            self._usfa_con_cached_data = dict(
                target_task_vecs=target_task_vecs,
                bounds=bounds,
            )
        else:
            assert self.usfa_con_cache_and_reuse_last_bounds
            target_task_vecs = self._usfa_con_cached_data['target_task_vecs']
            bounds = self._usfa_con_cached_data['bounds']

        lower = bounds['optimal_lower_bounds']
        upper = bounds['optimal_upper_bounds']
        q_values = self._compute_q_values(states, target_task_vecs, ensemble_reduction=({
            'ensemble_mean': 'mean',
            'ensemble_each': 'none',
        }[self.usfa_con_constraint_target]))

        if lower is not None:
            assert lower.size()[-3:] == q_values.size()[-3:]
        if upper is not None:
            assert upper.size()[-3:] == q_values.size()[-3:]

        if self.usfa_con_constraint_target == 'ensemble_each':
            if lower is not None:
                lower = lower[None]
            if upper is not None:
                upper = upper[None]

        if True:
            self._last_summary_usfa_con_lower = self._get_summary(lower)
            self._last_summary_usfa_con_upper = self._get_summary(upper)
            self._last_summary_usfa_con_upper_m_lower = (
                self._get_summary(upper - lower)
                if lower is not None and upper is not None
                else (0.0, 0.0, 0.0, 0.0)
            )
            self._last_summary_usfa_con_q_m_lower = (
                self._get_summary(q_values - lower)
                if lower is not None
                else (0.0, 0.0, 0.0, 0.0)
            )
            self._last_summary_usfa_con_upper_m_q = (
                self._get_summary(upper - q_values)
                if upper is not None
                else (0.0, 0.0, 0.0, 0.0)
            )

            if lower is not None:
                self._last_lower_bound_violation_ratio = (
                    (lower > q_values).sum().item()
                    / float(q_values.numel())
                )
            if upper is not None:
                self._last_upper_bound_violation_ratio = (
                    (upper < q_values).sum().item()
                    / float(q_values.numel())
                )
            if lower is not None and upper is not None:
                self._last_incompatible_lower_upper_bound_ratio = (
                    (lower > upper).sum().item()
                    / float(lower.numel())
                )


        return torch.mean(sum([
            F.relu(lower - q_values) if lower is not None else 0.0,
            F.relu(q_values - upper) if upper is not None else 0.0,
        ]))

    @profile
    def _train(self):
        # Because self.usfa_con_coeff can be an instance of LinearScheduler.
        curr_usfa_con_coeff = self.usfa_con_coeff
        called_backward_pass = False

        if self._should_train():
            # sample transitions from buffer
            minibatch = self.replay_buffer.sample(self.minibatch_size)

            if self.debug_usfa_con_exclusive_training and self._frames_seen >= self.usfa_con_coeff_start:
                loss = 0.0
                self._last_loss_usfa_psi_only = 0.0
            else:
                (states, actions, rewards, next_states, weights) = self._prepare_minibatch_for_training(minibatch)

                assert len(states.shape) == 1
                assert len(next_states.shape) == 1

                # self.replay_buffer is always ExperienceReplayBuffer.
                if (self.psi_ensemble_train_with_own_group_samples_only or 
                        self.psi_ensemble_train_even_mask or
                        self.psi_ensemble_train_rand_mask_bernoulli_prob is not None):

                    weights = states['ensemble_train_mask'].T
                    assert weights.size() == (self.psi_num_ensembles, *states.shape)
                else:
                    weights = weights[None]
                weights = weights[:, :, None]

                # forward pass
                # `states` includes `policy_vec`
                psi_values = self.psi(states, actions, ensemble_reduction='none')

                # compute targets
                strat = self.psi_ensemble_next_actions_strategy
                if strat == 'ensemble_each':
                    if True:
                        next_actions = torch.argmax(
                            self.q.no_grad(next_states, ensemble_reduction='none'),
                            dim=2)

                    assert next_actions.size() == (self.psi_num_ensembles, *next_states.shape)
                    next_psi_values = self.psi.target(
                        next_states,
                        ensemble_model_wise_kwargs=dict(
                            actions=next_actions,
                        ),
                        ensemble_reduction='none')
                else:
                    assert False, f'psi_ensemble_next_actions_strategy: {self.psi_ensemble_next_actions_strategy}'

                phi_feats = self.phi(states['observation'], actions, next_states['observation'])
                target_psi_values = phi_feats + self.discount_factor * next_psi_values

                # compute loss
                loss = self.loss(psi_values, target_psi_values, weights)
                self._last_loss_usfa_psi_only = loss.detach().item()

                with torch.no_grad():
                    num_models, num_samples, _ = psi_values.size()
                    assert num_models == self.psi_num_ensembles
                    q_values = torch.bmm(
                        psi_values.view(num_models * num_samples, 1, self.task_vec_dim),
                        unsqueeze_and_expand(
                            states['task_vec'],
                            dim=0,
                            num_repeat=num_models,
                        ).reshape(num_models * num_samples, self.task_vec_dim, 1),
                    ).view(num_models, num_samples)

                    self._last_summary_ensemble_q_std = self._get_summary(
                        q_values.std(dim=0))
                    self._last_summary_ensemble_q_diff_max_min = self._get_summary(
                        q_values.max(dim=0)[0] - q_values.min(dim=0)[0])

            usfa_con_loss = self._get_usfa_constraints_loss(minibatch[0])
            self._last_loss_usfa_con = (
                usfa_con_loss.detach().item()
                if torch.is_tensor(usfa_con_loss)
                else usfa_con_loss
            )
            usfa_con_loss = curr_usfa_con_coeff * usfa_con_loss
            self._last_loss_usfa_con_scaled = (
                usfa_con_loss.detach().item()
                if torch.is_tensor(usfa_con_loss)
                else usfa_con_loss
            )

            loss = loss + usfa_con_loss
            if torch.is_tensor(loss):
                self._last_loss_usfa_full = loss.detach().item()
                # backward pass
                self.psi.reinforce(loss, update_grad_norm=(self._frames_seen == 1 or self._frames_seen % 10 == 0))
                called_backward_pass = True
            else:
                self._last_loss_usfa_full = loss

            ### update replay buffer priorities
            ##td_errors = targets - values
            ##self.replay_buffer.update_priorities(td_errors.abs())

        self.writer.add_loss('usfa_psi_only', self._last_loss_usfa_psi_only,
                             dump_targets=['episode', 'frame'], write_to_tensorboard_events=True)
        self.writer.add_loss('usfa_con', self._last_loss_usfa_con,
                             dump_targets=['episode', 'frame'], write_to_tensorboard_events=bool(self.usfa_con_num_task_vecs))
        self.writer.add_loss('usfa_con_scaled', self._last_loss_usfa_con_scaled,
                             dump_targets=['episode', 'frame'], write_to_tensorboard_events=bool(self.usfa_con_num_task_vecs))
        self.writer.add_loss('usfa_full', self._last_loss_usfa_full,
                             dump_targets=['episode', 'frame'], write_to_tensorboard_events=bool(self.usfa_con_num_task_vecs))

        self.writer.add_scalar('time_optimal_value_bounding', self._last_time_optimal_value_bounding,
                               dump_targets=['episode', 'frame'])

        self.writer.add_scalar('lower_bound_violation_ratio', self._last_lower_bound_violation_ratio,
                               dump_targets=['episode', 'frame'])
        self.writer.add_scalar('upper_bound_violation_ratio', self._last_upper_bound_violation_ratio,
                               dump_targets=['episode', 'frame'])
        self.writer.add_scalar('incompatible_lower_upper_bound_ratio', self._last_incompatible_lower_upper_bound_ratio,
                               dump_targets=['episode', 'frame'])

        self.writer.add_summary('ensemble_q_std', *self._last_summary_ensemble_q_std,
                                dump_targets=['episode', 'frame'])
        self.writer.add_summary('ensemble_q_diff_max_min', *self._last_summary_ensemble_q_diff_max_min,
                                dump_targets=['episode', 'frame'])

        self.writer.add_summary('usfa_con_lower', *self._last_summary_usfa_con_lower,
                                dump_targets=['episode', 'frame'])
        self.writer.add_summary('usfa_con_upper', *self._last_summary_usfa_con_upper,
                                dump_targets=['episode', 'frame'])
        self.writer.add_summary('usfa_con_upper_m_lower', *self._last_summary_usfa_con_upper_m_lower,
                                dump_targets=['episode', 'frame'])
        self.writer.add_summary('usfa_con_q_m_lower', *self._last_summary_usfa_con_q_m_lower,
                                dump_targets=['episode', 'frame'])
        self.writer.add_summary('usfa_con_upper_m_q', *self._last_summary_usfa_con_upper_m_q,
                                dump_targets=['episode', 'frame'])

        #self.writer.add_string('model_params/psi', self.psi.get_parameters_hash(),
        #                       dump_targets=['episode', 'frame'])

        if not called_backward_pass:
            self.psi.step_dummy()

    def _should_train(self):
        self._frames_seen += 1
        return self._frames_seen > self.replay_start_size and self._frames_seen % self.update_frequency == 0


class USFATestAgent(Agent):
    def __init__(self,
                 gpi_policy,
                 default_gpi_source_policy_vecs,
                 include_target_task_vec_for_gpi):
        self.gpi_policy = gpi_policy
        self.default_gpi_source_policy_vecs = default_gpi_source_policy_vecs
        self.include_target_task_vec_for_gpi = include_target_task_vec_for_gpi

    def act(self, state):
        assert len(state.shape) == 0
        gpi_source_policy_vecs = self.default_gpi_source_policy_vecs
        if self.include_target_task_vec_for_gpi:
            gpi_source_policy_vecs = torch.cat([
                gpi_source_policy_vecs,
                state['task_vec'].unsqueeze(0),
            ], dim=0)
        state['gpi_source_policy_vecs'] = gpi_source_policy_vecs
        return self.gpi_policy.eval(state)

