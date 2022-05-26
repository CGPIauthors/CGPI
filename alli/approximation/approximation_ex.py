import hashlib
import pickle
import os
import sys

import torch

from all.approximation.approximation import Approximation
from alli.core.tensor_util import get_grad_norm

if '--debug_profile' in sys.argv:
    from rlkit.core.logging import profile
else:
    profile = lambda func: func

class ApproximationEx(Approximation):
    @profile
    def reinforce(self, loss, *, update_grad_norm=True):
        '''
        Backpropagate the loss through the model and make an update step.
        Internally, this will perform most of the activities associated with a control loop
        in standard machine learning environments, depending on the configuration of the object:
        Gradient clipping, learning rate schedules, logging, checkpointing, etc.

        Args:
            loss (torch.Tensor): The loss computed for a batch of inputs.

        Returns:
            self: The current Approximation object
        '''
        loss = self._loss_scaling * loss
        self._last_loss = loss.detach()
        self._writer.add_loss(self._name, self._last_loss)
        loss.backward()
        self.step(update_grad_norm=update_grad_norm)
        return self

    def __call__(self, *inputs, **kwargs):
        '''
        Run a forward pass of the model.
        '''
        return self.model(*inputs, **kwargs)

    def no_grad(self, *inputs, **kwargs):
        '''Run a forward pass of the model in no_grad mode.'''
        with torch.no_grad():
            return self.model(*inputs, **kwargs)

    def eval(self, *inputs, **kwargs):
        '''
        Run a forward pass of the model in eval mode with no_grad.
        The model is returned to its previous mode afer the forward pass is made.
        '''
        with torch.no_grad():
            # check current mode
            mode = self.model.training
            # switch model to eval mode
            self.model.eval()
            # run forward pass
            result = self.model(*inputs, **kwargs)
            # change to original mode
            self.model.train(mode)
            return result

    def target(self, *inputs, **kwargs):
        '''Run a forward pass of the target network.'''
        return self._target(*inputs, **kwargs)


    @profile
    def step(self, *, update_grad_norm=True):
        if update_grad_norm:
            self._last_grad_norm = get_grad_norm(self.model.parameters()).detach()
        self._writer.add_grad_norm(
            self._name,
            getattr(self, '_last_grad_norm', 0.0),
            dump_targets=['episode', 'frame'])
        return super().step()

    def step_dummy(self):
        self._writer.add_grad_norm(
            self._name,
            getattr(self, '_last_grad_norm', 0.0),
            dump_targets=['episode', 'frame'])
        self._writer.add_loss(self._name, getattr(self, '_last_loss', 0.0))

    def get_parameters_hash(self):
        return hashlib.md5(b''.join(
            p.data.cpu().numpy().tobytes() for p in self.model.parameters())).hexdigest()

