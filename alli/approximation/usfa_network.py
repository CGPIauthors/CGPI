import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from alli.approximation.approximation_ex import ApproximationEx
from all.core.state import StateArray
from all.nn import RLNetwork
from alli.approximation.ensemble import EnsembleModule, reduce_ensemble


class USFAPsiNetwork(ApproximationEx):
    def __init__(
            self,
            model,
            action_dim,
            task_vec_dim,
            normalize_policy_vecs,
            ensemble_reduction_info,
            optimizer=None,
            name='psi',
            **kwargs,
    ):
        self.action_dim = action_dim
        self.task_vec_dim = task_vec_dim
        self.normalize_policy_vecs = normalize_policy_vecs
        self.ensemble_reduction_info = ensemble_reduction_info

        if isinstance(model, EnsembleModule):
            for m in model.models:
                assert isinstance(m, USFAPsiModule)
        else:
            if isinstance(model, USFAPsiModule):
                model = EnsembleModule(models=[model])
            else:
                if (not isinstance(model, nn.ModuleList)) and (not isinstance(model, list)):
                    model = [model]

                model = EnsembleModule(
                    models=[
                        USFAPsiModule(
                            model=m,
                            action_dim=action_dim,
                            task_vec_dim=task_vec_dim,
                            normalize_policy_vecs=normalize_policy_vecs,
                        )
                        for m in model
                    ]
                )

        self.device = next(model.parameters()).device

        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

    def __call__(self, *inputs, ensemble_reduction):
        return self.model(
            *inputs,
            ensemble_reduction=ensemble_reduction,
            ensemble_reduction_info=self.ensemble_reduction_info,
        )

    def get_inner_models(self):
        return [psi_module.model for psi_module in self.model.models]

    def construct_q_network(self, **kwargs):
        return USFAQNetwork(**dict(dict(
            model=self.get_inner_models(),
            #action_dim=self.model.action_dim,
            #task_vec_dim=self.model.task_vec_dim,
            #normalize_policy_vecs=self.model.normalize_policy_vecs,
            action_dim=self.action_dim,
            task_vec_dim=self.task_vec_dim,
            normalize_policy_vecs=self.normalize_policy_vecs,
            ensemble_reduction_info=self.ensemble_reduction_info,
        ), **kwargs))

    def construct_gpi_q_network(self, **kwargs):
        return USFAGPIQNetwork(**dict(dict(
            model=self.get_inner_models(),
            #action_dim=self.model.action_dim,
            #task_vec_dim=self.model.task_vec_dim,
            #normalize_policy_vecs=self.model.normalize_policy_vecs,
            action_dim=self.action_dim,
            task_vec_dim=self.task_vec_dim,
            normalize_policy_vecs=self.normalize_policy_vecs,
            q_ensemble_reduction_info=self.ensemble_reduction_info,
        ), **kwargs))


class USFAPsiModule(nn.Module):
    def __init__(self,
                 model,
                 action_dim,
                 task_vec_dim,
                 normalize_policy_vecs,
        ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.action_dim = action_dim
        self.task_vec_dim = task_vec_dim
        self.normalize_policy_vecs = normalize_policy_vecs

    def forward(self, states, actions=None):
        states_as_input = [states.as_input(key) for key in ['observation', 'policy_vec']]
        if self.normalize_policy_vecs:
            states_as_input[-1] = F.normalize(states_as_input[-1], p=2.0, dim=-1)

        psi_values = states.apply_mask(states.as_output(
            self.model(torch.cat(states_as_input, dim=1))
        ))
        psi_values = psi_values.view(
                *states.shape, self.action_dim, self.task_vec_dim)
        assert psi_values.dim() <= 3
        if actions is None:
            return psi_values
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device)
        return psi_values.gather(1, actions.view(-1, 1, 1).expand(-1, -1, self.task_vec_dim)).squeeze(1)

class USFAQNetwork(ApproximationEx):
    def __init__(
            self,
            model,
            action_dim,
            task_vec_dim,
            normalize_policy_vecs,
            ensemble_reduction_info,
            optimizer=None,
            name='psi',
            **kwargs,
    ):

        self.ensemble_reduction_info = ensemble_reduction_info

        if (not isinstance(model, nn.ModuleList)) and (not isinstance(model, list)):
            model = [model]

        if isinstance(model, EnsembleModule):
            for m in model.models:
                assert isinstance(m, USFAQModule)
        else:
            if isinstance(model, USFAQModule):
                model = EnsembleModule(models=[model])
            else:
                model = EnsembleModule(
                    models=[
                        USFAQModule(
                            model=m,
                            action_dim=action_dim,
                            task_vec_dim=task_vec_dim,
                            normalize_policy_vecs=normalize_policy_vecs,
                        )
                        for m in model
                    ]
                )

        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

    def __call__(self, *inputs, ensemble_reduction):
        return self.model(
            *inputs,
            ensemble_reduction=ensemble_reduction,
            ensemble_reduction_info=self.ensemble_reduction_info,
        )

    def step(self):
        assert False, 'Updating with Q-values is not the way USFAs are trained'

    def reinforce(self):
        assert False, 'Updating with Q-values is not the way USFAs are trained'

class USFAQModule(nn.Module):
    def __init__(self, model, action_dim, task_vec_dim, normalize_policy_vecs):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.action_dim = action_dim
        self.task_vec_dim = task_vec_dim
        self.normalize_policy_vecs = normalize_policy_vecs

    def forward(self, states, actions=None):
        states_as_input = [states.as_input(key) for key in ['observation', 'policy_vec']]
        if self.normalize_policy_vecs:
            states_as_input[-1] = F.normalize(states_as_input[-1], p=2.0, dim=-1)

        psi_values = states.apply_mask(states.as_output(
            self.model(torch.cat(states_as_input, dim=1))
        ))
        task_vecs = states.as_input('task_vec')
        if actions is None:
            psi_values = psi_values.view(
                np.prod(states.shape, dtype=int) * self.action_dim, 1, self.task_vec_dim)
            task_vecs = task_vecs[:, None, :].expand(-1, self.action_dim, -1).reshape(
                np.prod(states.shape, dtype=int) * self.action_dim, self.task_vec_dim, 1)
            return (torch.bmm(psi_values, task_vecs)
                .squeeze(2).squeeze(1).view(*states.shape, self.action_dim))

        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device)
        psi_values = psi_values.view(
            *states.shape, self.action_dim, self.task_vec_dim)
        psi_values = psi_values.gather(1, actions.view(-1, 1)).squeeze(1)
        if psi_values.dim() == 1:
            psi_values = psi_values[None, :]
        assert psi_values.dim() == 2
        return (torch.bmm(psi_values[:, None, :], task_vecs[:, :, None])
            .squeeze(2).squeeze(1).view(*states.shape))

class USFAGPIQNetwork(ApproximationEx):
    def __init__(
            self,
            model,
            action_dim,
            task_vec_dim,
            normalize_policy_vecs,
            q_ensemble_reduction_info,
            optimizer=None,
            name='psi',
            q_ensemble_reduction=None,
            q_processor_pre_reduction=None,
            q_processor_post_reduction=None,
            q_processor_post_max=None,
            **kwargs
    ):
        if isinstance(model, USFAGPIQModule):
            assert isinstance(model.model, EnsembleModule)
        else:
            if isinstance(model, EnsembleModule):
                pass
            else:
                if (not isinstance(model, nn.ModuleList)) and (not isinstance(model, list)):
                    model = [model]
                model = EnsembleModule(models=model)

            model = USFAGPIQModule(
                model=model,
                action_dim=action_dim,
                task_vec_dim=task_vec_dim,
                normalize_policy_vecs=normalize_policy_vecs,
                q_ensemble_reduction=q_ensemble_reduction,
                q_ensemble_reduction_info=q_ensemble_reduction_info,
                q_processor_pre_reduction=q_processor_pre_reduction,
                q_processor_post_reduction=q_processor_post_reduction,
                q_processor_post_max=q_processor_post_max,
            )

        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

    #def __call__(self, *inputs, q_ensemble_reduction, q_ensemble_reduction_info):
    #    return self.model(
    #        *inputs,
    #        q_ensemble_reduction=q_ensemble_reduction,
    #        q_ensemble_reduction_info=q_ensemble_reduction_info,
    #    )

    def step(self):
        assert False, 'GPI is for inference'

    def reinforce(self):
        assert False, 'GPI is for inference'

class USFAGPIQModule(nn.Module):
    def __init__(self,
                 model: EnsembleModule,
                 action_dim,
                 task_vec_dim,
                 normalize_policy_vecs,
                 q_ensemble_reduction=None,
                 q_ensemble_reduction_info=dict(),
                 q_processor_pre_reduction=None,
                 q_processor_post_reduction=None,
                 q_processor_post_max=None,
        ):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.action_dim = action_dim
        self.task_vec_dim = task_vec_dim
        self.normalize_policy_vecs = normalize_policy_vecs
        self.q_ensemble_reduction = q_ensemble_reduction
        self.q_ensemble_reduction_info = q_ensemble_reduction_info
        self.q_processor_pre_reduction = q_processor_pre_reduction
        self.q_processor_post_reduction = q_processor_post_reduction
        self.q_processor_post_max = q_processor_post_max

    def forward(
            self,
            states,
            actions=None,
            *,
            q_ensemble_reduction=None,
            q_ensemble_reduction_info=dict(),
            q_processor_pre_reduction=None,
            q_processor_post_reduction=None,
            q_processor_post_max=None,
        ):
        assert (self.q_ensemble_reduction is None) != (q_ensemble_reduction is None)
        q_ensemble_reduction = q_ensemble_reduction or self.q_ensemble_reduction

        q_ensemble_reduction_info = dict(
            self.q_ensemble_reduction_info,
            **q_ensemble_reduction_info,
        )

        assert not (self.q_processor_pre_reduction is not None and q_processor_pre_reduction is not None)
        q_processor_pre_reduction = q_processor_pre_reduction or self.q_processor_pre_reduction

        assert not (self.q_processor_post_reduction is not None and q_processor_post_reduction is not None)
        q_processor_post_reduction = q_processor_post_reduction or self.q_processor_post_reduction

        assert not (self.q_processor_post_max is not None and q_processor_post_max is not None)
        q_processor_post_max = q_processor_post_max or self.q_processor_post_max

        # GPI is for inference
        with torch.no_grad():
            num_ensemble_models = self.model.num_models()

            gpi_source_policy_vecs = states['gpi_source_policy_vecs']
            if self.normalize_policy_vecs:
                gpi_source_policy_vecs = F.normalize(gpi_source_policy_vecs, p=2.0, dim=-1)
            num_gpi_source_policies_per_state = gpi_source_policy_vecs.size(len(states.shape))
            num_inputs = np.prod(states.shape, dtype=int) * num_gpi_source_policies_per_state

            def _expand_for_gpi(t):
                t = t.unsqueeze(len(states.shape)).expand(
                    *(-1 for _ in range(len(states.shape))),
                    num_gpi_source_policies_per_state,
                    *t.shape[len(states.shape):])
                return t.view(num_inputs, -1)
            masks = _expand_for_gpi(torch.as_tensor(states['mask'], device=self.device))
            obses = _expand_for_gpi(states['observation'])
            task_vecs = _expand_for_gpi(states['task_vec'])
            gpi_source_policy_vecs = gpi_source_policy_vecs.view(
                num_inputs, -1)
            all_psi_values = self.model(
                masks * torch.cat([obses, gpi_source_policy_vecs], dim=1),
                ensemble_reduction='none')
            assert all_psi_values.size() == (
                num_ensemble_models,
                num_inputs,
                self.action_dim * self.task_vec_dim)

            if actions is None:
                all_psi_values = all_psi_values.view(
                    num_ensemble_models * num_inputs * self.action_dim,
                    1,
                    self.task_vec_dim)
                task_vecs = task_vecs[None, :, None, :].expand(num_ensemble_models, -1, self.action_dim, -1).reshape(
                    num_ensemble_models * num_inputs * self.action_dim,
                    self.task_vec_dim,
                    1)
                all_q_values = (torch.bmm(all_psi_values, task_vecs)
                    .squeeze(2).squeeze(1).view(
                        num_ensemble_models, *states.shape, num_gpi_source_policies_per_state, self.action_dim))
                if q_processor_pre_reduction is not None:
                    all_q_values = q_processor_pre_reduction(
                        all_q_values,
                        states=states,
                        actions=actions,
                    )
                all_q_values = reduce_ensemble(
                    all_q_values,
                    reduction=q_ensemble_reduction,
                    reduction_info=q_ensemble_reduction_info,
                    dim=0,
                )
                if q_processor_post_reduction is not None:
                    all_q_values = q_processor_post_reduction(
                        all_q_values,
                        states=states,
                        actions=actions,
                    )
                all_q_values = all_q_values.max(len(states.shape))[0]
                if q_processor_post_max is not None:
                    all_q_values = q_processor_post_max(
                        all_q_values,
                        states=states,
                        actions=actions,
                    )
                return all_q_values

            if isinstance(actions, list):
                actions = torch.tensor(actions, device=self.device)
            all_psi_values = all_psi_values.view(
                num_ensemble_models * num_inputs,
                self.action_dim,
                self.task_vec_dim)
            all_psi_values = all_psi_values.gather(1, actions.view(-1, 1))
            task_vecs = task_vecs[None, :, :].expand(num_ensemble_models, -1, -1).view(
                num_ensemble_models * num_inputs,
                self.task_vec_dim,
                1)
            all_q_values = (torch.bmm(all_psi_values, task_vecs)
                .squeeze(2).squeeze(1).view(num_ensemble_models, *states.shape, num_gpi_source_policies_per_state))
            if q_processor_pre_reduction is not None:
                all_q_values = q_processor_pre_reduction(
                    all_q_values,
                    states=states,
                    actions=actions,
                )
            all_q_values = reduce_ensemble(
                all_q_values,
                reduction=q_ensemble_reduction,
                reduction_info=q_ensemble_reduction_info,
                dim=0,
            )
            if q_processor_post_reduction is not None:
                all_q_values = q_processor_post_reduction(
                    all_q_values,
                    states=states,
                    actions=actions,
                )
            all_q_values = all_q_values.max(len(states.shape))[0]
            if q_processor_post_max is not None:
                all_q_values = q_processor_post_max(
                    all_q_values,
                    states=states,
                    actions=actions,
                )
            return all_q_values

