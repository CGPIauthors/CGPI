from typing import List
import numpy as np
import scipy.signal

import torch
from all.core.state import State, StateArray
import warnings

def repeat_states(state: StateArray, num_repeat: int):
    #if not isinstance(state, StateArray):
    #    state = State.array([state])

    x = {}
    device = state.device
    shape = (state.shape[0] * num_repeat, *state.shape[1:])
    for key in state.keys():
        try:
            x[key] = torch.repeat_interleave(state[key], num_repeat, dim=0)
        except KeyError:
            warnings.warn('KeyError while repeating state component key "{}", omitting.'.format(key))
        except ValueError:
            warnings.warn('ValueError while repeating state component key "{}", omitting.'.format(key))
        except TypeError:
            warnings.warn('TypeError while repeating state component key "{}", omitting.'.format(key))

    return StateArray(x, shape, device=device)

def flatten_and_concat_states(list_of_states: List[State]):
    device = list_of_states[0].device
    shape = (sum(np.prod(s.shape) for s in list_of_states),)
    x = {}

    def _get_flattened_tensor(state, key):
        t = torch.tensor(state[key], device=device)
        return t.reshape(np.prod(state.shape), *t.shape[len(state.shape):])

    for key in list_of_states[0].keys():
        v = list_of_states[0][key]
        try:
            if isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
                x[key] = torch.cat([torch.stack(state.as_input(key)) for state in list_of_states], dim=0)
            elif torch.is_tensor(v):
                x[key] = torch.cat([state.as_input(key) for state in list_of_states], dim=0)
            else:
                x[key] = torch.cat([_get_flattened_tensor(state, key) for state in list_of_states], dim=0)
        except KeyError:
            warnings.warn('KeyError while creating StateArray for key "{}", omitting.'.format(key))
        except ValueError:
            warnings.warn('ValueError while creating StateArray for key "{}", omitting.'.format(key))
        except TypeError:
            warnings.warn('TypeError while creating StateArray for key "{}", omitting.'.format(key))

    return StateArray(x, shape, device=device)

def compute_returns(
        states: StateArray,
        discount_factor: float,
        dim: int):
    assert dim < len(states.shape)
    rewards = states['reward'].data.cpu().numpy()
    reversing_slices = (*(slice(None) for _ in range(dim)), slice(None, None, -1))
    returns = scipy.signal.lfilter([1], [1, float(-discount_factor)], rewards[reversing_slices], axis=dim)[reversing_slices]
    return torch.tensor(returns.copy(), device=states.device)

def apply_slices(states: StateArray, slices):
    assert len(slices) <= len(states.shape)
    x = {}
    for k, v in states.items():
        x[k] = v[slices]
        shape = x[k].shape[:(len(x[k].shape) - (len(v.shape) - len(states.shape)))]
    return StateArray(x, shape, device=states.device)

