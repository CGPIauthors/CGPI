import copy
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import diffcp

from all.approximation.approximation import Approximation
from all.core.state import State, StateArray
from alli.approximation.ensemble import reduce_ensemble
from alli.approximation.usfa_network import USFAPsiNetwork
from alli.core.misc_util import NoopContext
from alli.core.tensor_util import unsqueeze_and_expand
from rlkit.core.logging import logger_main, logger_sub

if '--debug_profile' in sys.argv:
    from rlkit.core.logging import profile
else:
    profile = lambda func: func

_g_cvxpylayer_cache = dict()

def get_cvxpylaer(var_dim, num_ineq_constraints, num_eq_constraints):
    params = (var_dim, num_ineq_constraints, num_eq_constraints)
    if params not in _g_cvxpylayer_cache:
        cp_z = cp.Variable(var_dim)

        cp_p = cp.Parameter(var_dim)

        cp_G = cp.Parameter((num_ineq_constraints, var_dim))
        cp_h = cp.Parameter(num_ineq_constraints)

        cp_A = cp.Parameter((num_eq_constraints, var_dim))
        cp_b = cp.Parameter(num_eq_constraints)

        objective = cp.Minimize(cp.sum(cp.multiply(cp_p, cp_z)))
        constraints = [
            cp_G @ cp_z <= cp_h,
            cp_A @ cp_z == cp_b,
        ]

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(
            problem,
            parameters=[
                cp_p,
                cp_G,
                cp_h,
                cp_A,
                cp_b,
            ],
            variables=[cp_z],
        )
        _g_cvxpylayer_cache[params] = cvxpylayer
    return _g_cvxpylayer_cache[params]


def compute_optimal_value_bounds(**kwargs):
    with (torch.no_grad if kwargs.get('detach', True) else NoopContext)():
        return _compute_optimal_value_bounds_impl(**kwargs)

@profile
def _compute_optimal_value_bounds_impl(
        *,
        psi: USFAPsiNetwork,
        source_task_vecs: torch.Tensor,
        states: State,
        target_task_vecs: torch.Tensor,
        source_task_min_rewards,
        discount_factor,
        max_path_length,
        device,
        solver,
        ensemble_reduction_for_source_values,
        ensemble_reduction_info_for_source_values,
        ensemble_reduction_for_source_to_target_values,
        ensemble_reduction_info_for_source_to_target_values,
        use_v_values=False,
        min_value_adjustment=None,
        non_negative_coeffs_and_rewards=False,
        detach=True,
        compute_targets=['optimal_lower_bounds', 'optimal_upper_bounds'],

        workaround_source_task_vecs_are_one_hots=False,
        
        debug_sol=False,
        debug_return_locals=False,
        debug_with_bounded_sol_space=False,
        debug_min_alpha=-1000.0,
        debug_max_alpha=1000.0,
        debug_sanity_check=False,
    ):

    ret = dict()

    assert solver in [
        'cvxpylayers',
        'special_case_pos_one_hots',
        'special_case_pos_and_neg_one_hots_with_nonneg_coeffs',
    ]
    assert min_value_adjustment in [None, 'per_source_task', 'per_sample']
    assert ensemble_reduction_for_source_values in ['mean', 'max', 'ucb']
    assert ensemble_reduction_for_source_to_target_values in ['mean', 'min', 'lcb']

    assert source_task_vecs.dim() == 2
    assert target_task_vecs.dim() == 2

    if workaround_source_task_vecs_are_one_hots:
        assert False, 'workaround_source_task_vecs_are_one_hots is deprecated. Use solver instead.'

    source_task_min_values = (
        source_task_min_rewards *
        (1.0 - discount_factor ** max_path_length) / (1.0 - discount_factor))
    assert (source_task_vecs.size(0),) == source_task_min_values.size()

    num_states = int(np.prod(states.shape))
    observations = unsqueeze_and_expand(
        states['observation'].view(
            num_states,
            *states['observation'].size()[len(states.shape):]),
        dim=1,
        num_repeat=source_task_vecs.size(0))
    observations = observations.reshape(
        num_states * source_task_vecs.size(0),
        *observations.size()[2:])
    source_vecs = unsqueeze_and_expand(
        source_task_vecs,
        dim=0,
        num_repeat=num_states)
    source_vecs = source_vecs.reshape(
        num_states * source_task_vecs.size(0),
        *source_vecs.size()[2:])

    source_psi_values = psi(
        StateArray(
            {
                'observation': observations,
                'policy_vec': source_vecs,
            },
            shape=(num_states * source_task_vecs.size(0),),
            device=device,
        ),
        ensemble_reduction='none',
    )

    num_ensemble_models, _d1, action_dim, task_vec_dim = source_psi_values.size()
    assert _d1 == num_states * source_task_vecs.size(0)
    del _d1
    assert task_vec_dim == target_task_vecs.size(-1)
    num_all_combs = target_task_vecs.size(0) * num_states * source_task_vecs.size(0) * action_dim

    source_values = torch.bmm(
        source_psi_values.view(
            num_ensemble_models * num_states * source_task_vecs.size(0) * action_dim, 1, task_vec_dim,
        ),
        unsqueeze_and_expand(
            unsqueeze_and_expand(source_vecs, dim=0, num_repeat=num_ensemble_models),
            dim=2,
            num_repeat=action_dim,
        ).reshape(
            num_ensemble_models * num_states * source_task_vecs.size(0) * action_dim, task_vec_dim, 1,
        ),
    )
    source_values = reduce_ensemble(
        source_values.view(
            num_ensemble_models, num_states * source_task_vecs.size(0) * action_dim, 1, 1),
        reduction=ensemble_reduction_for_source_values,
        reduction_info=ensemble_reduction_info_for_source_values,
        dim=0,
    )
    assert (num_states * source_task_vecs.size(0) * action_dim, 1, 1) == source_values.size()
    if min_value_adjustment == 'per_source_task':
        if use_v_values:
            mins_from_source_values = source_values.view(
                num_states, source_task_vecs.size(0), action_dim).max(dim=2)[0].min(dim=0)[0]
        else:
            mins_from_source_values = source_values.view(
                num_states, source_task_vecs.size(0), action_dim).min(dim=2)[0].min(dim=0)[0]
        source_task_min_values = torch.minimum(
            source_task_min_values,
            mins_from_source_values)
    source_values = unsqueeze_and_expand(
        source_values,
        dim=0,
        num_repeat=target_task_vecs.size(0),
    ).reshape(num_all_combs)

    source_to_target_values = torch.bmm(
        unsqueeze_and_expand(
            source_psi_values,
            dim=1,
            num_repeat=target_task_vecs.size(0),
        ).reshape(
            num_ensemble_models * num_all_combs, 1, task_vec_dim,
        ),
        unsqueeze_and_expand(
            unsqueeze_and_expand(target_task_vecs, dim=0, num_repeat=num_ensemble_models),
            dim=2,
            num_repeat=num_states * source_task_vecs.size(0) * action_dim,
        ).reshape(
            num_ensemble_models * num_all_combs, task_vec_dim, 1,
        ),
    )
    source_to_target_values = reduce_ensemble(
        source_to_target_values.view(
            num_ensemble_models, num_all_combs, 1, 1),
        reduction=ensemble_reduction_for_source_to_target_values,
        reduction_info=ensemble_reduction_info_for_source_to_target_values,
        dim=0,
    )
    assert (num_all_combs, 1, 1) == source_to_target_values.size()

    if use_v_values:
        desired_output_shape = (target_task_vecs.size(0), num_states)
        problem_batch_size = target_task_vecs.size(0) * num_states
        # State-values.
        source_values = source_values.view(
            problem_batch_size, source_task_vecs.size(0), action_dim).max(dim=2)[0]
        source_to_target_values = source_to_target_values.view(
            problem_batch_size, source_task_vecs.size(0), action_dim).max(dim=2)[0]
    else:
        desired_output_shape = (target_task_vecs.size(0), num_states, action_dim)
        problem_batch_size = target_task_vecs.size(0) * num_states * action_dim
        source_values = source_values.view(
            target_task_vecs.size(0) * num_states, source_task_vecs.size(0), action_dim)
        source_values = source_values.transpose(1, 2).reshape(
            problem_batch_size, source_task_vecs.size(0))
        source_to_target_values = source_to_target_values.view(
            target_task_vecs.size(0) * num_states, source_task_vecs.size(0), action_dim)
        source_to_target_values = source_to_target_values.transpose(1, 2).reshape(
            problem_batch_size, source_task_vecs.size(0))
    assert (problem_batch_size, source_task_vecs.size(0)) == source_to_target_values.size()
    assert (problem_batch_size, source_task_vecs.size(0)) == source_values.size()

    if min_value_adjustment == 'per_sample':
        source_task_min_values_ex = torch.minimum(source_task_min_values[None, :], source_values)
    else:
        assert min_value_adjustment in [None, 'per_source_task']
        source_task_min_values_ex = unsqueeze_and_expand(
            source_task_min_values,
            dim=0,
            num_repeat=problem_batch_size,
        )
    assert (problem_batch_size, source_task_vecs.size(0)) == source_task_min_values_ex.size()

    if 'optimal_lower_bounds' in compute_targets:
        optimal_lower_bounds = source_to_target_values.max(dim=1)[0].view(*desired_output_shape)
        if detach:
            optimal_lower_bounds = optimal_lower_bounds.detach()
    else:
        optimal_lower_bounds = None

    if 'optimal_upper_bounds' in compute_targets:
        if solver in ['special_case_pos_one_hots', 'special_case_pos_and_neg_one_hots_with_nonneg_coeffs']:
            if solver == 'special_case_pos_one_hots':
                # {{{
                assert source_task_vecs.size(0) == task_vec_dim
                if debug_sanity_check:
                    assert torch.allclose(
                        source_task_vecs.sum(dim=1),
                        torch.ones_like(source_task_vecs[:, 0]))
                    assert torch.allclose(
                        source_task_vecs.sum(dim=0),
                        torch.ones_like(source_task_vecs[0, :]))

                corresponding_dims = unsqueeze_and_expand(
                    source_task_vecs.argmax(dim=-1),
                    dim=0,
                    num_repeat=target_task_vecs.size(0),
                )
                assert corresponding_dims.size() == target_task_vecs.size()

                sol_coeffs = unsqueeze_and_expand(
                    torch.gather(target_task_vecs, 1, corresponding_dims),
                    dim=1,
                    num_repeat=(num_states if use_v_values else (num_states * action_dim)),
                ).reshape(problem_batch_size, source_task_vecs.size(0))
                # }}}
            elif solver == 'special_case_pos_and_neg_one_hots_with_nonneg_coeffs':
                # {{{
                assert source_task_vecs.size(0) == source_task_vecs.size(1) * 2
                if debug_sanity_check:
                    assert torch.allclose(
                        source_task_vecs.sum(dim=1).abs(),
                        torch.ones_like(source_task_vecs[:, 0]))
                    assert torch.allclose(
                        source_task_vecs.sum(dim=0),
                        torch.zeros_like(source_task_vecs[0, :]))

                source_task_vecs_ex = unsqueeze_and_expand(
                    source_task_vecs,
                    dim=0,
                    num_repeat=target_task_vecs.size(0),
                )

                corresponding_dims_pos = unsqueeze_and_expand(
                    source_task_vecs.argmax(dim=-1),
                    dim=0,
                    num_repeat=target_task_vecs.size(0),
                )
                assert corresponding_dims_pos.size() == (target_task_vecs.size(0), source_task_vecs.size(0))
                sol_coeffs_orig_pos = torch.gather(
                    target_task_vecs, 1, corresponding_dims_pos)
                source_task_vecs_at_argmax = source_task_vecs_ex.max(dim=-1).values
                filter_pos = torch.logical_and(
                    sol_coeffs_orig_pos >= 0,
                    source_task_vecs_at_argmax > 0.5,  # Reasonably safe value.
                ).to(torch.float)

                corresponding_dims_neg = unsqueeze_and_expand(
                    source_task_vecs.argmin(dim=-1),
                    dim=0,
                    num_repeat=target_task_vecs.size(0),
                )
                assert corresponding_dims_neg.size() == (target_task_vecs.size(0), source_task_vecs.size(0))
                sol_coeffs_orig_neg = torch.gather(
                    target_task_vecs, 1, corresponding_dims_neg)
                source_task_vecs_at_argmin = source_task_vecs_ex.min(dim=-1).values
                filter_neg = torch.logical_and(
                    sol_coeffs_orig_neg < 0,
                    source_task_vecs_at_argmin < (- 0.5),  # Reasonably safe value.
                ).to(torch.float)

                sol_coeffs = unsqueeze_and_expand(
                    (filter_pos * sol_coeffs_orig_pos - filter_neg * sol_coeffs_orig_neg),
                    dim=1,
                    num_repeat=(num_states if use_v_values else (num_states * action_dim)),
                ).reshape(problem_batch_size, source_task_vecs.size(0))
                # }}}
            else:
                assert False

            if debug_sanity_check:
                # {{{
                A_half = unsqueeze_and_expand(
                    source_task_vecs.T,
                    dim=0,
                    num_repeat=problem_batch_size,
                )
                reconstructed_target_task_vecs = torch.bmm(
                    A_half, sol_coeffs[:, :, None],
                ).squeeze(2)

                logger_main.log(f'Orig target task vecs:\n{target_task_vecs}, reconstructed ones:\n{reconstructed_target_task_vecs}')

                target_task_vecs_ex = unsqueeze_and_expand(
                    target_task_vecs,
                    dim=1,
                    num_repeat=(num_states if use_v_values else (num_states * action_dim)),
                ).reshape(problem_batch_size, task_vec_dim)
                assert torch.allclose(target_task_vecs_ex, reconstructed_target_task_vecs)
                # }}}

            optimal_upper_bounds = torch.maximum(
                sol_coeffs * source_values,
                sol_coeffs * source_task_min_values_ex,
            ).sum(dim=1).view(*desired_output_shape)
            if detach:
                optimal_upper_bounds = optimal_upper_bounds.detach()

        elif solver in ['cvxpylayers']:
            # {{{

            if non_negative_coeffs_and_rewards:
                raise NotImplementedError

            qp_p = unsqueeze_and_expand(
                torch.cat([
                    torch.ones(source_task_vecs.size(0), device=device),
                    torch.zeros(source_task_vecs.size(0), device=device),
                ], dim=0),
                dim=0,
                num_repeat=problem_batch_size,
            )
            if detach:
                qp_p = qp_p.detach()

            qp_kwargs = dict(
                ## Q
                #Q=unsqueeze_and_expand(
                #    torch.zeros(2 * source_task_vecs.size(0), 2 * source_task_vecs.size(0), device=device),
                #    dim=0,
                #    num_repeat=problem_batch_size,
                #),
                # p
                p=qp_p,
                # G
                G=torch.cat([
                    torch.cat([
                        unsqueeze_and_expand(
                            torch.diag(-torch.ones(source_task_vecs.size(0), device=device)),
                            dim=0,
                            num_repeat=problem_batch_size,
                        ),
                        torch.diag_embed(source_values),
                    ], dim=2),
                    torch.cat([
                        unsqueeze_and_expand(
                            torch.diag(-torch.ones(source_task_vecs.size(0), device=device)),
                            dim=0,
                            num_repeat=problem_batch_size,
                        ),
                        #unsqueeze_and_expand(
                        #    torch.diag(source_task_min_values),
                        #    dim=0,
                        #    num_repeat=problem_batch_size,
                        #),
                        torch.diag_embed(source_task_min_values_ex),
                    ], dim=2),
                ] + ([] if not debug_with_bounded_sol_space else [
                    torch.cat([
                        unsqueeze_and_expand(
                            torch.zeros(source_task_vecs.size(0), source_task_vecs.size(0), device=device),
                            dim=0,
                            num_repeat=problem_batch_size,
                        ),
                        unsqueeze_and_expand(
                            torch.diag(-torch.ones(source_task_vecs.size(0), device=device)),
                            dim=0,
                            num_repeat=problem_batch_size,
                        ),
                    ], dim=2),
                    torch.cat([
                        unsqueeze_and_expand(
                            torch.zeros(source_task_vecs.size(0), source_task_vecs.size(0), device=device),
                            dim=0,
                            num_repeat=problem_batch_size,
                        ),
                        unsqueeze_and_expand(
                            torch.diag(torch.ones(source_task_vecs.size(0), device=device)),
                            dim=0,
                            num_repeat=problem_batch_size,
                        ),
                    ], dim=2),
                ]), dim=1),
                # h
                h=unsqueeze_and_expand(
                    torch.cat([
                        torch.zeros(2 * source_task_vecs.size(0), device=device),
                    ] + ([] if not debug_with_bounded_sol_space else [
                        -debug_min_alpha * torch.ones(source_task_vecs.size(0), device=device),
                        debug_max_alpha * torch.ones(source_task_vecs.size(0), device=device),
                    ]), dim=0),
                    dim=0,
                    num_repeat=problem_batch_size,
                ),
                # A
                A=unsqueeze_and_expand(
                    torch.cat([
                        torch.zeros(task_vec_dim, source_task_vecs.size(0), device=device),
                        source_task_vecs.T,
                    ], dim=1),
                    dim=0,
                    num_repeat=problem_batch_size,
                ),
                # b
                b=unsqueeze_and_expand(
                    target_task_vecs,
                    dim=1,
                    num_repeat=(num_states if use_v_values else (num_states * action_dim)),
                ).reshape(problem_batch_size, task_vec_dim),
            )
            #if detach:
            #    for k in range(len(qp_args)):
            #        qp_args[k] = qp_args[k].detach()
            if detach:
                for k in qp_kwargs.keys():
                    qp_kwargs[k] = qp_kwargs[k].detach()

            if solver == 'cvxpylayers':
                # {{{
                cvxpylayer = get_cvxpylaer(
                    2 * source_task_vecs.size(0),
                    (2 if not debug_with_bounded_sol_space else 4) * source_task_vecs.size(0),
                    task_vec_dim,
                )

                try:
                    qp_sol, = cvxpylayer(
                        qp_kwargs['p'],
                        qp_kwargs['G'],
                        qp_kwargs['h'],
                        qp_kwargs['A'],
                        qp_kwargs['b'],
                        solver_args={
                            'n_jobs_forward': 1,
                            'n_jobs_backward': 1,
                        },
                    )

                except diffcp.cone_program.SolverError as e:
                    qp_sol = None
                # }}}
            else:
                assert False


            if qp_sol is not None:
                assert (problem_batch_size, 2 * source_task_vecs.size(0)) == qp_sol.size()

                optimal_upper_bounds = torch.bmm(
                    qp_p.view(problem_batch_size, 1, 2 * source_task_vecs.size(0)),
                    qp_sol.view(problem_batch_size, 2 * source_task_vecs.size(0), 1),
                ).view(*desired_output_shape)
                if detach:
                    optimal_upper_bounds = optimal_upper_bounds.detach()
            else:
                optimal_upper_bounds = None
            # }}}
        else:
            assert False
    else:
        optimal_upper_bounds = None

    ret.update(dict(
        optimal_lower_bounds=optimal_lower_bounds,
        optimal_upper_bounds=optimal_upper_bounds,
    ))

    if debug_return_locals:
        if 'optimal_upper_bounds' in compute_targets:
            if solver in ['cvxpylayers']:
                alpha = qp_sol[:, source_task_vecs.size(0):]
            else:
                assert False
        ret['locals'] = locals()

    return ret

