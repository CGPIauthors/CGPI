import numpy as np
import torch
from torch import nn

_g_standard_normal = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

def reduce_ensemble(values, *, reduction, reduction_info=dict(), dim=0):
    psi_num_ensembles = values.size()[dim]
    psi_num_ensemble_groups = reduction_info.get('psi_num_ensemble_groups', None)

    group_reduction = reduction_info.get('group_reduction', None)
    if group_reduction is not None:
        # {{{
        assert psi_num_ensemble_groups is not None
        group_reduction_info = dict(
            {
                k: v
                for k, v in reduction_info.items()
                if k not in [
                    'psi_num_ensemble_groups',
                    'group_reduction',
                    'group_reduction_info',
                ]
            },
            **reduction_info.get('group_reduction_info', dict())
        )

        values = reduce_ensemble(
            values.view(
                *values.size()[:dim],
                psi_num_ensemble_groups,
                psi_num_ensembles // psi_num_ensemble_groups,
                *values.size()[dim+1:],
            ),
            reduction=group_reduction,
            reduction_info=group_reduction_info,
            dim=dim+1,
        )
        assert values.size()[dim] == psi_num_ensemble_groups
        psi_num_ensembles = values.size()[dim]
        psi_num_ensemble_groups = None
        # }}}

    if reduction == 'none':
        res = values
    elif reduction == 'mean':
        res = values.mean(dim=dim)
    elif reduction == 'min':
        res = values.min(dim=dim)[0]
    elif reduction == 'max':
        res = values.max(dim=dim)[0]
    elif reduction == 'lcb':
        assert psi_num_ensembles >= 2
        mean = values.mean(dim=dim)
        std = values.std(dim=dim)
        assert reduction_info['lcb_weight'] >= 0
        res = mean - reduction_info['lcb_weight'] * std
    elif reduction == 'ucb':
        assert psi_num_ensembles >= 2
        mean = values.mean(dim=dim)
        std = values.std(dim=dim)
        assert reduction_info['ucb_weight'] >= 0
        res = mean + reduction_info['ucb_weight'] * std
    elif reduction == 'max_of_group_mins':
        assert group_reduction is None
        res = reduce_ensemble(
            values,
            reduction='max',
            reduction_info=dict(
                reduction_info,
                group_reduction='min',
            ),
        )
    elif reduction == 'specific_group_mean':
        assert group_reduction is None
        assert psi_num_ensemble_groups is not None
        assert psi_num_ensembles % psi_num_ensemble_groups == 0
        selected_group_idx = reduction_info['selected_group_idx']
        res = values.view(
            *values.size()[:dim],
            psi_num_ensemble_groups,
            psi_num_ensembles // psi_num_ensemble_groups,
            *values.size()[dim+1:],
        )[(*(slice(None) for _ in range(dim)), selected_group_idx)].mean(dim=dim)
    elif reduction == 'gaussian_expected_min':
        assert psi_num_ensembles >= 2
        assert psi_num_ensemble_groups in [None, 1, psi_num_ensembles]

        mean = values.mean(dim=dim)
        std = values.std(dim=dim)

        res = mean - std * _g_standard_normal.icdf(torch.as_tensor(
            (psi_num_ensembles - np.pi / 8.0) /
            (psi_num_ensembles - np.pi / 4.0 + 1.0)
        ))
    elif reduction == 'gaussian_expected_max':
        assert psi_num_ensembles >= 2
        assert psi_num_ensemble_groups in [None, 1, psi_num_ensembles]

        mean = values.mean(dim=dim)
        std = values.std(dim=dim)

        res = mean + std * _g_standard_normal.icdf(torch.as_tensor(
            (psi_num_ensembles - np.pi / 8.0) /
            (psi_num_ensembles - np.pi / 4.0 + 1.0)
        ))
    else:
        assert False
    return res


class EnsembleModule(nn.Module):
    def __init__(self, models):
        super().__init__()
        if not isinstance(models, nn.ModuleList):
            models = nn.ModuleList(models)
        self.models = models

    def num_models(self):
        return len(self.models)

    def forward(self,
                *args,
                ensemble_reduction,
                ensemble_reduction_info=dict(),
                ensemble_model_wise_kwargs=dict(),
                **kwargs):
        assert ensemble_reduction in ['none', 'mean', 'min', 'max', 'max_of_group_mins']
        for v in ensemble_model_wise_kwargs.values():
            assert len(v) == self.num_models()
        ensemble_model_wise_kwargs = [
            {
                k: v[idx]
                for k, v in ensemble_model_wise_kwargs.items()
            }
            for idx in range(self.num_models())
        ]
        values = torch.stack([
            m(*args, **mwkwargs, **kwargs)
            for m, mwkwargs in zip(self.models, ensemble_model_wise_kwargs)
        ], dim=0)

        return reduce_ensemble(
            values,
            reduction=ensemble_reduction,
            reduction_info=ensemble_reduction_info,
            dim=0,
        )

