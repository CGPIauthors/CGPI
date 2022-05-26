# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python [conda env:psfs]
#     language: python
#     name: conda-env-psfs-py
# ---

# %% [markdown] tags=[]
# # Misc. setup

# %% tags=[]
import os
ipynb_dir = globals()['_dh'][0]
main_dir = os.path.join(ipynb_dir, '..')

# %% tags=[]
os.chdir(main_dir)

# %%
import torch
torch.set_num_threads(1)

# %% tags=[]
import seaborn as sns
cm = sns.light_palette("green", as_cmap=True)

# %% tags=[]
import matplotlib
import matplotlib.pyplot as plt
#plt.rcParams['figure.figsize'] = (60, 60)
#plt.rcParams['font.size'] = 80
plt.rcParams['figure.dpi'] = 40
#matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'

import os
import glob
import numpy as np

# %% tags=[]
# %matplotlib inline
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from PIL import Image
from IPython.core.pylabtools import figsize

mpl.rcParams['pdf.fonttype'] = 42     # use true-type
mpl.rcParams['ps.fonttype'] = 42      # use true-type
mpl.rcParams['font.size'] = 12

# %config InlineBackend.figure_format = 'retina'

# %% tags=[]
import pandas as pd
import numpy as np

# %% tags=[]
from test_scripts.train_reacher_usfa import *

# %% [markdown] tags=[]
# # Hyperparameters and inputs

# %% tags=[]
hyperparameters = get_hyperparameters({}, args=[
    '--device', 'cpu',
    '--run_group', 'jupyter',
    '--seed', '1',
    '--test_exploration', '0.0',
    '--num_test_episodes', '10',
])

load_caches_only = False

use_deterministic_transition_with_salts = True

#common_sort_key = 'avg. return'
common_sort_key = 'avg. undiscounted_return'

verbose_print = False

# %% tags=[]
import itertools
import torch
import numpy as np

target_task_vecs_spec = 'PN2T1'
target_task_xys = sorted([
    list(t)
    for t in itertools.product(*[[-1.0, 1.0] for _ in range(4)])
])

target_task_xys_np = np.asarray(target_task_xys)

target_task_vecs = ReacherWrappedEnv.convert_target_info_to_task_vec(
    torch.as_tensor(target_task_xys, dtype=torch.float64),
    phi_type='neg_dists_to_source_xys',
).tolist()

print(str(target_task_vecs).replace('\n', ' '))

# %%
all_target_task_vecs_filters_np = dict()
all_target_task_vecs_filters_np['all'] = np.ones_like(np.asarray(target_task_vecs)[:, 0], dtype=bool)
all_target_task_vecs_filters_np['equal_or_more_negatives'] = (np.asarray(target_task_vecs).sum(axis=-1) <= 1e-5)

target_task_vecs_filter_np = all_target_task_vecs_filters_np['all']

# %% tags=[]
import os
import re

psi_ckpts_infos_test = [
    dict(
        name='Reacher',
        paths=[
            r'/path/to/your/expdir1/psi_001000000.pt',
            r'/path/to/your/expdir2/psi_001000000.pt',
            r'/path/to/your/expdir3/psi_001000000.pt',
            r'/path/to/your/expdir4/psi_001000000.pt',
            r'/path/to/your/expdir5/psi_001000000.pt',
            r'/path/to/your/expdir6/psi_001000000.pt',
            r'/path/to/your/expdir7/psi_001000000.pt',
            r'/path/to/your/expdir8/psi_001000000.pt',
        ],
    ),
]

# %% [markdown] tags=[]
# # Defining functions and objects

# %%
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

task_preset = reacher_hps['task_preset']
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




# %%
for info in psi_ckpts_infos_test:
    info['source_task_vecs'] = getattr(ReacherTaskPresets, task_preset).train_task_vecs()
    info['source_xys'] = getattr(ReacherTaskPresets, task_preset).source_xys()


# %%
import copy
def construct_env_for_specific_task_vec(task_vec, env_kwargs=dict()):
    def _sampler():
        return task_vec
    hps = exp_utils.AttributeDict(copy.copy(dict(reacher_hps)))
    hps['task_vec_sampler'] = _sampler
    assert 'source_xys' in env_kwargs
    return ReacherWrappedEnv(
        **dict(
            hps,
            **env_kwargs,
        ),
        device=hyperparameters['device'],
    )
    


# %%
import glob
import os

def _sort_key(path):
    return tuple(reversed(os.path.split(path)))
    
def ensure_file(path):
    if os.path.exists(path):
        if verbose_print:
            print(path)
        return path
    g = sorted(glob.glob(path), key=_sort_key)
    #assert len(g) == 1
    if verbose_print:
        print(g[-1])
    return g[-1]

def load_usfa_net(path):
    psi_module = torch.load(
        ensure_file(path),
        map_location=hyperparameters['device'])
    return USFAPsiNetwork(
        model=psi_module,
        action_dim=preset.n_actions,
        task_vec_dim=preset.env.get_task_vec_dim(),
        normalize_policy_vecs=hyperparameters['normalize_policy_vecs'],
        ensemble_reduction_info=dict(
        ),
    )


# %% tags=[]
def construct_test_agent_from_psi(psi,
                                  default_gpi_source_policy_vecs=torch.zeros((0, preset.env.get_task_vec_dim()), device=hyperparameters['device']),
                                  include_target_task_vec_for_gpi=True,
                                  epsilon=hyperparameters['test_exploration'],
                                  gpi_q_network_kwargs=dict(q_ensemble_reduction='mean')):
    gpi_q = psi.construct_gpi_q_network(**gpi_q_network_kwargs)
    gpi_policy = GreedyPolicy(gpi_q, preset.n_actions, epsilon=epsilon)
    return USFATestAgent(
        gpi_policy=gpi_policy,
        default_gpi_source_policy_vecs=default_gpi_source_policy_vecs,
        include_target_task_vec_for_gpi=include_target_task_vec_for_gpi,
    )

def construct_test_agent_from_ckpt(path, *args, **kwargs):
    psi = load_usfa_net(path)
    return construct_test_agent_from_psi(psi, *args, **kwargs)



# %%
import time
from all.core.state import State, StateArray
from alli.core.state_util import compute_returns

def _collect_states_from_single_rollout(env, test_agent):
    #time_start = time.time()

    states = []
    # initialize the episode
    states.append(env.reset())
    action = test_agent.act(states[-1])

    # loop until the episode is finished
    while not states[-1].done:
        states.append(env.step(action))
        action = test_agent.act(states[-1])

    #time_end = time.time()
    #print(f'fps: {len(states) / (time_end - time_start)}, len: {len(states)}')
        
    return State.array(states)
    
def collect_states_with_generator(envs_generator, test_agent, post_rollout_callbacks=[]):
    all_states = []
    for env in envs_generator:
        all_states.append(_collect_states_from_single_rollout(env=env, test_agent=test_agent))
        for cb in post_rollout_callbacks:
            cb(env, test_agent, all_states[-1])
    all_states = State.array(all_states)
    all_states['return'] = compute_returns(
        all_states,
        hyperparameters['discount_factor'],
        dim=1)
    all_states['undiscounted_return'] = compute_returns(
        all_states,
        1.0,
        dim=1)
    return all_states


# %%
import numpy as np
from alli.core.state_util import apply_slices

def collect_states_on_target_tasks(
    psi_info,
    path,
    gpi_type,
    num_test_episodes,
    *,
    use_deterministic_transition_with_salts: bool,
    get_test_agent=None,
    target_task_vecs=target_task_vecs,
    post_rollout_callbacks=[],
):
    assert gpi_type in ['target', 'source', 'source + target']
    psi = load_usfa_net(path)
    source_task_vecs = psi_info['source_task_vecs']
    
    def _get_test_agent(t):
        if gpi_type == 'target':
            default_gpi_source_policy_vecs = torch.as_tensor([t], device=hyperparameters['device'])
        elif gpi_type == 'source':
            default_gpi_source_policy_vecs = source_task_vecs
        elif gpi_type == 'source + target':
            default_gpi_source_policy_vecs = torch.cat([
                source_task_vecs,
                torch.as_tensor([t], device=hyperparameters['device']),
            ], dim=0)
        else:
            assert False
        return construct_test_agent_from_psi(
            psi,
            default_gpi_source_policy_vecs=default_gpi_source_policy_vecs,
            include_target_task_vec_for_gpi=False,
        )
    
    if get_test_agent is None:
        get_test_agent = _get_test_agent

    def _envs_generator(t):
        for idx in range(num_test_episodes):
            if use_deterministic_transition_with_salts:
                yield construct_env_for_specific_task_vec(
                    torch.as_tensor([t], device=hyperparameters['device']),
                    dict(
                        env_specific_random_seed=idx,
                        source_xys=psi_info['source_xys'],
                    ),
                )
            else:
                yield construct_env_for_specific_task_vec(
                    torch.as_tensor([t], device=hyperparameters['device']),
                    dict(
                        source_xys=psi_info['source_xys'],
                    ),
                )
    
    return [
        collect_states_with_generator(
            envs_generator=_envs_generator(t),
            test_agent=get_test_agent(t),
            post_rollout_callbacks=post_rollout_callbacks,
        )
        for t in target_task_vecs
    ]


# %% [markdown]
# # Caching

# %%
from collections import defaultdict
import hashlib
import os
import json
import pickle
import cloudpickle

def pickle_dumps(obj):
    try:
        return pickle.dumps(obj)
    except (pickle.PicklingError, AttributeError) as e:
        return cloudpickle.dumps(obj)

def get_cache_ex_json_file_path(psi_info, path, name, setting_name, hyp, dir_name_suffix=''):
    cache_dir_name = f'ipynbs/cache.eval_perf_stats_ex_json_reacher{dir_name_suffix}'
    try:
        os.makedirs(cache_dir_name)
    except:
        pass

    if callable(path):
        path = path()

    path_identifier = os.path.basename(os.path.dirname(path)) + os.path.sep + os.path.basename(path)

    filename = (
        name + '__' +
        hashlib.md5(str.encode(path_identifier)).hexdigest() + '__' +
        setting_name + '__' +
        hashlib.md5(pickle_dumps(hyp)).hexdigest() +
        '.json'
    )

    if verbose_print:
        print(path_identifier, '->', hashlib.md5(str.encode(path_identifier)).hexdigest())
        print(hyp, '->', hashlib.md5(pickle_dumps(hyp)).hexdigest())

    assert len(filename) <= os.pathconf('/', 'PC_NAME_MAX'), filename
    
    if verbose_print:
        print(filename)

    return os.path.join(
        cache_dir_name,
        filename)

def read_cache_ex_json(psi_info, path, name, setting_name, hyp, **kwargs):
    cache_path = get_cache_ex_json_file_path(psi_info, path, name, setting_name, hyp, **kwargs)
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_cache_ex_json(psi_info, path, name, setting_name, hyp, data, **kwargs):
    cache_path = get_cache_ex_json_file_path(psi_info, path, name, setting_name, hyp, **kwargs)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# %% [markdown]
# # Perf stat-related definitions

# %%
def get_perf_stats_with_gpi_options(
        psi_info_test,
        psi,
        t,
        *,
        num_test_episodes,
        use_deterministic_transition_with_salts: bool,
        source_task_vecs=None,
        include_target_task_vec_for_gpi=False,
        gpi_q_network_kwargs=dict(q_ensemble_reduction='mean'),
        post_rollout_callbacks=[],
    ):

    if source_task_vecs is None:
        source_task_vecs = psi_info_test['source_task_vecs']
    target_task_vecs_test = torch.as_tensor([t], device=hyperparameters['device'])
    
    gpi_source_policy_vecs = source_task_vecs
    test_agent = construct_test_agent_from_psi(
        psi,
        default_gpi_source_policy_vecs=gpi_source_policy_vecs,
        include_target_task_vec_for_gpi=include_target_task_vec_for_gpi,
        gpi_q_network_kwargs=gpi_q_network_kwargs,
    )

    def _envs_generator(t):
        for idx in range(num_test_episodes):
            if use_deterministic_transition_with_salts:
                yield construct_env_for_specific_task_vec(
                    torch.as_tensor([t], device=hyperparameters['device']),
                    dict(
                        env_specific_random_seed=idx,
                        source_xys=psi_info_test['source_xys'],
                    ),
                )
            else:
                yield construct_env_for_specific_task_vec(
                    torch.as_tensor([t], device=hyperparameters['device']),
                    dict(
                        source_xys=psi_info_test['source_xys'],
                    ),
                )
    
    initial_state = apply_slices(
        collect_states_with_generator(
            envs_generator=_envs_generator(t),
            test_agent=test_agent,
            post_rollout_callbacks=post_rollout_callbacks,
        ),
        (slice(None), 0),
    )
    
    return {
        'return': initial_state['return'],
        'undiscounted_return': initial_state['undiscounted_return'],
    }



# %%
from collections import defaultdict
from IPython.display import Markdown, display
from typing import Union, Callable

import copy
import pickle
import hashlib
import re

re_name_ensem_group = re.compile(r'ensemGroup-(?:[0-9]+)-([0-9]+)')
re_path_ensem_group = re.compile(r'_eg([0-9]+)_')

def _identity_hyp_processor(hyp, psi_info_test, path, psi, t, curr_data, post_rollout_callback_registrator):
    return hyp

def gather_perf_stats_for_gpi_setting(
        *,
        settings,
        hyp_processor=_identity_hyp_processor,
        get_gpi_type,
        source_task_vecs: Union[None, torch.Tensor, str, Callable[..., torch.Tensor]],
        include_target_task_as_source,
        requires_grouped_ensembles,
        psi_ckpts_infos_test=psi_ckpts_infos_test,
        psi_info_filter=lambda x: True,
        num_test_episodes=hyperparameters["num_test_episodes"],
    ):

    if len(settings) == 0 or len(psi_ckpts_infos_test) == 0:
        # Early quitting.
        return
    
    keys_blacklist = [
        'name',
        'path',
        'gpi_type',
        'deterministic_transition',
        'num_test_episodes',
        'use_deterministic_transition_with_salts',
        'hyperparameters',
        'target_task_vecs_spec',
        'target_task_xys',
        'target_task_vecs',
    ]

    def _get_all_gpi_types():
        for hyp_orig in settings:
            yield get_gpi_type(*hyp_orig)

    for psi_info_test in psi_ckpts_infos_test:
        if not psi_info_filter(psi_info_test):
            continue

        if requires_grouped_ensembles:
            re_name_match = re_name_ensem_group.search(psi_info_test['name'])
            if re_name_match is None:
                continue
            psi_num_ensemble_groups = int(re_name_match.group(1))
            
        if 'perf_stats' not in psi_info_test:
            psi_info_test['perf_stats'] = dict()
        
        for path_idx, path in enumerate(psi_info_test['paths']):
            if requires_grouped_ensembles:
                assert psi_num_ensemble_groups == int(re_path_ensem_group.search(path).group(1))

            if not callable(path):
                psi = load_usfa_net(path)
            curr_psi_stats = dict()

            for hyp_orig in settings:
                hyp_orig = copy.deepcopy(hyp_orig)
                if requires_grouped_ensembles:
                    for hi in hyp_orig:
                        if 'q_ensemble_reduction_info' in hi:
                            hi['q_ensemble_reduction_info']['psi_num_ensemble_groups'] = psi_num_ensemble_groups

                gpi_type = get_gpi_type(*hyp_orig)

                setting_name = f'{target_task_vecs_spec}_{gpi_type}_{int(use_deterministic_transition_with_salts)}_{num_test_episodes}'

                COMPACT_CACHE = '.compact-v2'

                curr_data = read_cache_ex_json(
                    psi_info=psi_info_test,
                    path=path,
                    name=psi_info_test['name'],
                    setting_name=setting_name,
                    hyp=hyp_orig,
                    dir_name_suffix=COMPACT_CACHE,
                )

                if curr_data is None:
                    # {{{
                    curr_data = read_cache_ex_json(
                        psi_info=psi_info_test,
                        path=path,
                        name=psi_info_test['name'],
                        setting_name=setting_name,
                        hyp=hyp_orig,
                    )
                        
                    if curr_data is None:
                        # {{{
                        if load_caches_only:
                            continue
                        print(psi_info_test['name'], gpi_type)
                        curr_data = dict( returns=[], undiscounted_returns=[], )
                        for t_idx, t in enumerate(target_task_vecs):

                            path_final = (path(t) if callable(path) else path)

                            if gpi_type in ['target', 'source', 'source + target']:
                                # {{{
                                initial_state = apply_slices(
                                    collect_states_on_target_tasks(
                                        psi_info_test,
                                        path_final,
                                        gpi_type,
                                        num_test_episodes,
                                        use_deterministic_transition_with_salts=use_deterministic_transition_with_salts,
                                        target_task_vecs=[t],
                                    )[0],
                                    (slice(None), 0),
                                )
                                stat = {
                                    'return': initial_state['return'],
                                    'undiscounted_return': initial_state['undiscounted_return'],
                                }
                                # }}}
                            else:
                                # {{{
                                post_rollout_callbacks = []
                                def _post_rollout_callback_registrator(cb):
                                    post_rollout_callbacks.append(cb)

                                hyp = hyp_processor(
                                    copy.deepcopy(hyp_orig),
                                    psi_info_test=psi_info_test,
                                    path=path_final,
                                    psi=psi,
                                    t=t,
                                    curr_data=curr_data,
                                    post_rollout_callback_registrator=_post_rollout_callback_registrator,
                                )

                                target_task_vecs_test = torch.as_tensor([t], device=hyperparameters['device'])

                                source_task_vecs_test = source_task_vecs
                                if not isinstance(source_task_vecs_test, torch.Tensor):
                                    if callable(source_task_vecs_test):
                                        source_task_vecs_test = source_task_vecs_test(psi_info_test, path_final, t, hyp)
                                    elif source_task_vecs_test == 'own_source':
                                        source_task_vecs_test = psi_info_test['source_task_vecs']

                                if include_target_task_as_source:
                                    if source_task_vecs_test is None:
                                        source_task_vecs_test = target_task_vecs_test
                                    else:
                                        source_task_vecs_test = torch.cat([
                                            source_task_vecs_test,
                                            target_task_vecs_test,
                                        ], dim=0)

                                stat = get_perf_stats_with_gpi_options(
                                    psi_info_test,
                                    psi,
                                    t,
                                    num_test_episodes=num_test_episodes,
                                    use_deterministic_transition_with_salts=use_deterministic_transition_with_salts,
                                    source_task_vecs=source_task_vecs_test,
                                    include_target_task_vec_for_gpi=False,
                                    gpi_q_network_kwargs=hyp[0],
                                    post_rollout_callbacks=post_rollout_callbacks,
                                )
                                # }}}

                            curr_data['returns'].append(stat['return'].tolist())
                            curr_data['undiscounted_returns'].append(stat['undiscounted_return'].tolist())
                            
                        data_to_save = copy.deepcopy(curr_data)
                        data_to_save['name'] = psi_info_test['name']
                        data_to_save['path'] = path_final
                        data_to_save['gpi_type'] = gpi_type
                        data_to_save['num_test_episodes'] = num_test_episodes
                        data_to_save['use_deterministic_transition_with_salts'] = use_deterministic_transition_with_salts
                        data_to_save['hyperparameters'] = hyperparameters
                        data_to_save['target_task_vecs_spec'] = target_task_vecs_spec
                        data_to_save['target_task_xys'] = target_task_xys
                        data_to_save['target_task_vecs'] = target_task_vecs
                        save_cache_ex_json(
                            psi_info=psi_info_test,
                            path=path,
                            name=psi_info_test['name'],
                            setting_name=setting_name,
                            hyp=hyp_orig,
                            data=data_to_save,
                        )
                        # }}}

                    compact_data_to_save = copy.deepcopy(curr_data)
                    compact_data_to_save['name'] = psi_info_test['name']
                    compact_data_to_save['path'] = path_final
                    compact_data_to_save['gpi_type'] = gpi_type
                    compact_data_to_save['num_test_episodes'] = num_test_episodes
                    compact_data_to_save['use_deterministic_transition_with_salts'] = use_deterministic_transition_with_salts
                    compact_data_to_save['hyperparameters'] = hyperparameters
                    compact_data_to_save['target_task_vecs_spec'] = target_task_vecs_spec
                    compact_data_to_save['target_task_xys'] = target_task_xys
                    compact_data_to_save['target_task_vecs'] = target_task_vecs
                    for k in [
                        'returns',
                        'undiscounted_returns',
                        'lower_bound_violation_ratio_post_reduction',
                        'lower_bound_violation_ratio_post_max',
                        'upper_bound_violation_ratio_post_reduction',
                        'upper_bound_violation_ratio_post_max',
                        'gpi_action_changed_ratio',
                    ]:
                        if k in compact_data_to_save:
                            compact_data_to_save[k] = [
                                [
                                    np.mean(e, keepdims=True).tolist()
                                    for e in d
                                ]
                                for d in compact_data_to_save[k]
                            ]
                            curr_data[k] = compact_data_to_save[k]
                    save_cache_ex_json(
                        psi_info=psi_info_test,
                        path=path,
                        name=psi_info_test['name'],
                        setting_name=setting_name,
                        hyp=hyp_orig,
                        data=compact_data_to_save,
                        dir_name_suffix=COMPACT_CACHE,
                    )
                    # }}}

                for key in curr_data.keys():
                    if key in keys_blacklist:
                        continue
                    assert len(curr_data[key]) == len(target_task_vecs), key

                curr_psi_stats[gpi_type] = {
                    key: [
                        [np.mean(v) for v in e]
                        for e in curr_data[key]
                    ]
                    for key in curr_data.keys()
                    if key not in keys_blacklist
                }

                for key in curr_data.keys():
                    if key not in keys_blacklist:
                        assert len(curr_psi_stats[gpi_type][key]) == len(target_task_vecs)
                    
            for gt, d in curr_psi_stats.items():
                if gt not in psi_info_test['perf_stats']:
                    psi_info_test['perf_stats'][gt] = {
                        key: []
                        for key in curr_psi_stats[gt].keys()
                    }
                if any((not isinstance(v, list))
                        for v in psi_info_test['perf_stats'][gt].values()):
                    continue
                    
                for key in curr_psi_stats[gt].keys():
                    psi_info_test['perf_stats'][gt][key].append(curr_psi_stats[gt][key])
                
                    
                
        for gpi_type in _get_all_gpi_types():
            if gpi_type not in psi_info_test['perf_stats']:
                continue
            for key in psi_info_test['perf_stats'][gpi_type].keys():
                if isinstance(psi_info_test['perf_stats'][gpi_type][key], list):
                    psi_info_test['perf_stats'][gpi_type][key] = np.asarray(psi_info_test['perf_stats'][gpi_type][key]).transpose(1, 0, 2)



# %%
from collections import defaultdict
from IPython.display import Markdown, display

import pickle
import hashlib


def ensure_common_perf_stats_on_target_tasks(psi_info, num_test_episodes):
    settings = [
        ('target',),
        ('source',),
        ('source + target',),
    ]

    def _get_gpi_type(gpi_type):
        return gpi_type

    gather_perf_stats_for_gpi_setting(
        settings=settings,
        get_gpi_type=_get_gpi_type,
        source_task_vecs=None,
        include_target_task_as_source=None,
        requires_grouped_ensembles=False,
        psi_ckpts_infos_test=[psi_info],
        psi_info_filter=lambda x: True,
        num_test_episodes=num_test_episodes,
    )



# %% [markdown]
# # Comparison of empirical returns

# %%
def ensure_perf_stats2(info):
    ensure_common_perf_stats_on_target_tasks(
        info, hyperparameters['num_test_episodes'])



# %% tags=[]
for info in psi_ckpts_infos_test:
    ensure_perf_stats2(info)

# %%
from all.core.state import State, StateArray
from alli.core.tensor_util import unsqueeze_and_expand

# %% [markdown]
# ## Constrained GPI

# %% tags=[]
from collections import defaultdict
import copy
import pickle
import hashlib
import re
import time

from IPython.display import Markdown, display
import numpy as np
import pprint

from alli.approximation.optimal_value_bounder import compute_optimal_value_bounds
from alli.core.tensor_util import unsqueeze_and_expand


# %% tags=[]
def _cgpi_hyp_processor(hyp, psi_info_test, path, psi, t, curr_data, post_rollout_callback_registrator):
    gpi_q_network_kwargs = hyp[0]
    extra_info = hyp[1]

    stat_keys = [
        'lower_bound_violation_ratio_post_reduction',
        'lower_bound_violation_ratio_post_max',
        'upper_bound_violation_ratio_post_reduction',
        'upper_bound_violation_ratio_post_max',
        'gpi_action_changed_ratio',
    ]
    for k in stat_keys:
        if k not in curr_data:
            curr_data[k] = []
        curr_data[k].append([])

    stacked_data = dict()

    def _post_rollout_callback(env, test_agent, states):
        #print(len(states), len(stacked_data[list(stacked_data.keys())[0]]))
        assert len(states) == len(stacked_data[list(stacked_data.keys())[0]])

        for k in stacked_data.keys():
            curr_data[k][-1].append(stacked_data[k])
            stacked_data[k] = []

    post_rollout_callback_registrator(_post_rollout_callback)

    def _q_processor_post_reduction(
        all_q_values,
        states,
        actions,
    ):
        # {{{
        assert actions is None

        if 'BQ1D4' in psi_info_test['name'] or 'BPDX' in psi_info_test['name'] or 'BPNDX' in psi_info_test['name']:
            workaround_source_task_vecs_are_one_hots = False
        else:
            workaround_source_task_vecs_are_one_hots = False

        source_task_vecs = psi_info_test['source_task_vecs']
        target_task_vecs = torch.as_tensor([t], device=hyperparameters['device'])

        compute_targets = []

        if extra_info['reduction_for_bounds'] in ['lm_um', 'lm']:
            compute_targets.append('optimal_lower_bounds')
            ensemble_reduction_for_source_to_target_values = 'mean'
            ensemble_reduction_info_for_source_to_target_values = dict()
        else:
            # DUMMY
            ensemble_reduction_for_source_to_target_values = 'mean'
            ensemble_reduction_info_for_source_to_target_values = dict()

        if extra_info['reduction_for_bounds'] in ['lm_um', 'um']:
            compute_targets.append('optimal_upper_bounds')
            ensemble_reduction_for_source_values = 'mean'
            ensemble_reduction_info_for_source_values = dict()
        else:
            # DUMMY
            ensemble_reduction_for_source_values = 'mean'
            ensemble_reduction_info_for_source_values = dict()

        solver = hyperparameters['usfa_constraints']['usfa_con_upper_bound_lp_solver']

        bounds = compute_optimal_value_bounds(
            psi=psi,
            source_task_vecs=source_task_vecs,
            states=states,
            target_task_vecs=target_task_vecs,
            source_task_min_rewards=ReacherWrappedEnv.get_min_rewards(
                source_task_vecs,
                phi_type='neg_dists_to_source_xys',
                method='phi_range',
                source_xys=psi_info_test['source_xys'],
            ),
            discount_factor=hyperparameters['discount_factor'],
            max_path_length=hyperparameters['reacher']['max_path_length'],
            device=hyperparameters['device'],
            solver=solver,
            ensemble_reduction_for_source_values=ensemble_reduction_for_source_values,
            ensemble_reduction_info_for_source_values=ensemble_reduction_info_for_source_values,
            ensemble_reduction_for_source_to_target_values=ensemble_reduction_for_source_to_target_values,
            ensemble_reduction_info_for_source_to_target_values=ensemble_reduction_info_for_source_to_target_values,
            use_v_values=False,
            min_value_adjustment=None,
            detach=True,
            compute_targets=compute_targets,
            workaround_source_task_vecs_are_one_hots=workaround_source_task_vecs_are_one_hots,
        )

        num_states = np.prod(states.shape)

        num_gpi_source_policies_per_state = all_q_values.size(-2)
        all_q_values_max = all_q_values.max(len(states.shape)).values

        this_data = dict()

        updated_all_q_values = all_q_values

        if 'optimal_lower_bounds' in compute_targets:
            # {{{
            assert bounds['optimal_lower_bounds'].size() == (target_task_vecs.size(0), num_states, preset.n_actions)

            optimal_lower_bounds_orig = bounds['optimal_lower_bounds'].squeeze(0)
            optimal_lower_bounds = unsqueeze_and_expand(
                optimal_lower_bounds_orig,
                dim=1,
                num_repeat=num_gpi_source_policies_per_state,
            ).view(*states.shape, num_gpi_source_policies_per_state, preset.n_actions)
            assert optimal_lower_bounds.size() == all_q_values.size()

            this_data['lower_bound_violation_ratio_post_reduction'] = (
                (optimal_lower_bounds > all_q_values).sum().item()
                / float(all_q_values.numel())
            )
            this_data['lower_bound_violation_ratio_post_max'] = (
                (optimal_lower_bounds_orig > all_q_values_max).sum().item()
                / float(all_q_values_max.numel())
            )

            updated_all_q_values = torch.maximum(updated_all_q_values, optimal_lower_bounds)
            # }}}

        if 'optimal_upper_bounds' in compute_targets:
            # {{{
            assert bounds['optimal_upper_bounds'].size() == (target_task_vecs.size(0), num_states, preset.n_actions)

            optimal_upper_bounds_orig = bounds['optimal_upper_bounds'].squeeze(0)
            optimal_upper_bounds = unsqueeze_and_expand(
                optimal_upper_bounds_orig,
                dim=1,
                num_repeat=num_gpi_source_policies_per_state,
            ).view(*states.shape, num_gpi_source_policies_per_state, preset.n_actions)
            assert optimal_upper_bounds.size() == all_q_values.size()

            this_data['upper_bound_violation_ratio_post_reduction'] = (
                (optimal_upper_bounds < all_q_values).sum().item()
                / float(all_q_values.numel())
            )
            this_data['upper_bound_violation_ratio_post_max'] = (
                (optimal_upper_bounds_orig < all_q_values_max).sum().item()
                / float(all_q_values_max.numel())
            )

            updated_all_q_values = torch.minimum(updated_all_q_values, optimal_upper_bounds)
            # }}}

        gpi_actions = all_q_values_max.argmax(-1)
        this_data['gpi_action_changed_ratio'] = (
            (updated_all_q_values.max(len(states.shape)).values.argmax(-1) != gpi_actions).sum().item()
            / float(gpi_actions.numel())
        )

        for k, v in this_data.items():
            if k not in stacked_data:
                stacked_data[k] = []
            stacked_data[k].append(v)

        return updated_all_q_values

        # }}}

    gpi_q_network_kwargs['q_processor_post_reduction'] = _q_processor_post_reduction

    return (gpi_q_network_kwargs, extra_info)

# %% tags=[]
def _cgpi_get_gpi_type(gpi_q_network_kwargs, extra_info, gpi_input):
    con_info = ''
    con_info += 'mP '

    if 'targetrand' in gpi_input:
        gpi_input += f' ({extra_info["num_rands"]} {extra_info["rand_std"]})'

    if extra_info["reduction_for_bounds"] in ['lm_um', 'lm', 'um']:
        con_info += f'{extra_info["reduction_for_bounds"]}'
    else:
        assert False
    return f'con ({con_info}) {gpi_input} {gpi_q_network_kwargs["q_ensemble_reduction"]}'


# %%
settings = [
    (
        dict(q_ensemble_reduction='mean',),
        dict(reduction_for_bounds='lm_um',),
    ),
]

import functools
_get_gpi_type = functools.partial(_cgpi_get_gpi_type, gpi_input='target')

source_task_vecs_for_perf = None

gather_perf_stats_for_gpi_setting(
    settings=settings,
    hyp_processor=_cgpi_hyp_processor,
    get_gpi_type=_get_gpi_type,
    source_task_vecs=source_task_vecs_for_perf,
    include_target_task_as_source=True,
    requires_grouped_ensembles=False,
    psi_ckpts_infos_test=psi_ckpts_infos_test,
    psi_info_filter=lambda x: ('ensem' not in x['name']),
)

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Rliable

# %% [markdown]
# ### Preparation

# %%
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from collections import OrderedDict
from IPython.display import Markdown, display

# %%
def get_algo_name(name, gpi_type):
    return f'{name} ({gpi_type})'

# %%
def gather_min_max_scores(
    *,
    perf_quantity_info,
    plot_infos,
    target_task_vecs_filter_np,
):
    min_scores = None
    max_scores = None

    pqi = perf_quantity_info
    pq = pqi['quantity']

    for pi in plot_infos:
        name = pi['name']
        info_candidates = [
            i for i in psi_ckpts_infos_test
            if i['name'] == name
        ]
        if len(info_candidates) == 0:
            continue
        info = info_candidates[-1]
        stats = info['perf_stats']

        gpi_types = pi['gpi_types']
        for gpi_type in gpi_types:
            # Each psi_info_test['perf_stats'][gpi_type][key] has a shape of:
            # (num_target_tasks x num_ckpts x num_runs)
            curr_stats = stats[gpi_type][pq]
            curr_stats = curr_stats[target_task_vecs_filter_np]
            num_target_tasks, num_ckpts, num_runs = curr_stats.shape

            # Now, num_ckpts becomes the number of runs for Rliable.
            curr_scores = curr_stats.transpose(1, 0, 2).reshape(
                num_ckpts, num_target_tasks * num_runs,
            )
            curr_min_scores = curr_scores.min(axis=0)
            curr_max_scores = curr_scores.max(axis=0)

            if min_scores is None:
                min_scores = curr_min_scores
            else:
                min_scores = np.minimum(min_scores, curr_min_scores)

            if max_scores is None:
                max_scores = curr_max_scores
            else:
                max_scores = np.maximum(max_scores, curr_max_scores)

    return dict(
        min_scores=min_scores,
        max_scores=max_scores,
    )

# %%
def get_normalized_score_dict(
    *,
    perf_quantity_info,
    plot_infos,
    target_task_vecs_filter_np,
    epsilon=1e-3,
):
    pqi = perf_quantity_info
    pq = pqi['quantity']

    min_max_scores_info = gather_min_max_scores(
        perf_quantity_info=pqi,
        plot_infos=plot_infos,
        target_task_vecs_filter_np=target_task_vecs_filter_np,
    )

    score_dict = OrderedDict()

    for pi in plot_infos:
        name = pi['name']
        info_candidates = [
            i for i in psi_ckpts_infos_test
            if i['name'] == name
        ]
        if len(info_candidates) == 0:
            continue
        info = info_candidates[-1]
        stats = info['perf_stats']

        gpi_types = pi['gpi_types']
        for gpi_type in gpi_types:
            algo = get_algo_name(name, gpi_type)

            # Each psi_info_test['perf_stats'][gpi_type][key] has a shape of:
            # (num_target_tasks x num_ckpts x num_runs)
            curr_stats = stats[gpi_type][pq]
            curr_stats = curr_stats[target_task_vecs_filter_np]
            num_target_tasks, num_ckpts, num_runs = curr_stats.shape

            # Now, num_ckpts becomes the number of runs for Rliable.
            curr_scores = curr_stats.transpose(1, 0, 2).reshape(
                num_ckpts, num_target_tasks * num_runs,
            )

            score_dict[algo] = (curr_scores - min_max_scores_info['min_scores']) / (min_max_scores_info['max_scores'] - min_max_scores_info['min_scores'] + epsilon)
            print(pq, algo, score_dict[algo].shape)

    return score_dict


# %% [markdown]
# ### Rliable plots

# %%
all_possible_algorithms = [
    r"\textbf{USFAs, CGPI (ours) w/ target}",
    r"USFAs, GPI w/ source + target",
    r"USFAs, GPI w/ target",
    r"USFAs, GPI w/ source",
]
color_palette = sns.color_palette('colorblind', n_colors=len(all_possible_algorithms))
predefined_colors = dict(zip(all_possible_algorithms, color_palette))


# %%
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12

from matplotlib import rc

# activate latex text rendering
rc('text', usetex=True)

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# %%
def _algo_name_converter(name):
    if name.endswith(' (source)'):
        new_name = r"USFAs, GPI w/ source"
    if name.endswith(' (target)'):
        new_name = r"USFAs, GPI w/ target"
    if name.endswith(' (source + target)'):
        new_name = r"USFAs, GPI w/ source + target"
    if name.endswith(' (con (mP lm_um) target mean)'):
        new_name = r"\textbf{USFAs, CGPI (ours) w/ target}"
    if name.endswith(' (con (mP lm_um) source mean)'):
        new_name = r"\textbf{USFAs, CGPI (ours) w/ source}"
    if name.endswith(' (con (mP lm_um) source + target mean)'):
        new_name = r"\textbf{USFAs, CGPI (ours) w/ source + target}"
    return new_name

# %%
_plot_infos = [
    dict(
        name='Reacher',
        gpi_types=[
            'target',
            'source',
            'source + target',
            'con (mP lm_um) target mean',
        ],
    ),
]

_perf_quantity_infos = [
    dict(
        quantity='undiscounted_returns', 
    ),
]

for task_filter in ['all', 'equal_or_more_negatives']:
    score_dict = get_normalized_score_dict(
        perf_quantity_info=_perf_quantity_infos[0],
        plot_infos=_plot_infos,
        target_task_vecs_filter_np=all_target_task_vecs_filters_np[task_filter],
        epsilon=1e-3,
    )

    score_dict = OrderedDict([
        (_algo_name_converter(k), v)
        for k, v in score_dict.items()
    ])

    algorithms = [
        r"USFAs, GPI w/ source",
        r"USFAs, GPI w/ target",
        r"USFAs, GPI w/ source + target",
        r"\textbf{USFAs, CGPI (ours) w/ target}",
    ]
    algorithms = list(reversed(algorithms))

    score_dict = OrderedDict([
        (k, score_dict[k])
        for k in algorithms
    ])

    # Load ALE scores as a dictionary mapping algorithms to their human normalized
    # score matrices, each of which is of size `(num_runs x num_games)`.
    aggregate_func = lambda x: np.array([
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_optimality_gap(x),
    ])

    rep = 500
    rep = 50000


    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=rep)
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores, aggregate_score_cis,
        metric_names=[
            'Median',
            'IQM',
            'Mean',
            'Optimality Gap',
        ],
        colors=predefined_colors,
        algorithms=algorithms, xlabel='Normalized Score', xlabel_y_coordinate=-0.3)

    display(fig)

    fig.clf()



# %%
_plot_infos = [
    dict(
        name='Reacher',
        gpi_types=[
            'target',
            'source',
            'source + target',
            'con (mP lm_um) target mean',
        ],
    ),
]

_perf_quantity_infos = [
    dict(
        quantity='undiscounted_returns', 
    ),
]

for task_filter in ['all', 'equal_or_more_negatives']:
    score_dict = get_normalized_score_dict(
        perf_quantity_info=_perf_quantity_infos[0],
        plot_infos=_plot_infos,
        target_task_vecs_filter_np=all_target_task_vecs_filters_np[task_filter],
        epsilon=1e-3,
    )

    score_dict = OrderedDict([
        (_algo_name_converter(k), v)
        for k, v in score_dict.items()
    ])
    print(list(score_dict.keys()))

    algorithms = [
        r"USFAs, GPI w/ source",
        r"USFAs, GPI w/ target",
        r"USFAs, GPI w/ source + target",
        r"\textbf{USFAs, CGPI (ours) w/ target}",
    ]

    score_dict = OrderedDict([
        (k, score_dict[k])
        for k in algorithms
    ])

    # Load ALE scores as a dictionary mapping algorithms to their human normalized
    # score matrices, each of which is of size `(num_runs x num_games)`.

    # Human normalized score thresholds
    thresholds = np.linspace(0.0, 1.0, 5)
    thresholds = np.linspace(0.0, 1.0, 100)
    thresholds = np.linspace(0.0, 1.0, 20)
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        score_dict, thresholds)
    # Plot score distributions
    fig, ax = plt.subplots(ncols=1, figsize=(4.9, 3.5))
    plot_utils.plot_performance_profiles(
        score_distributions, thresholds,
        performance_profile_cis=score_distributions_cis,
        colors=predefined_colors,
        xlabel=r'Normalized Score $(\tau)$',
        ax=ax,
        legend=False,
    )

    display(fig)

    fig.clf()















