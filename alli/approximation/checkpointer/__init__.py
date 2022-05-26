import warnings

import torch

from all.approximation.checkpointer import Checkpointer

class PeriodicCheckpointerEx(Checkpointer):
    def __init__(self, frequency, filename_template):
        self.frequency = frequency
        self._updates = 1
        self._filename_template = filename_template
        self._model = None

    def init(self, model, _):
        self._model = model
        # Some builds of pytorch throw this unhelpful warning.
        # We can safely disable it.
        # https://discuss.pytorch.org/t/got-warning-couldnt-retrieve-source-code-for-container/7689/7
        warnings.filterwarnings("ignore", message="Couldn't retrieve source code")

    def __call__(self):
        if self._updates % self.frequency == 0:
            torch.save(self._model, self._filename_template.format(self._updates))
        self._updates += 1

