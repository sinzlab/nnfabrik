from pprint import pformat

from torch import nn
import torch
from torch.nn import ModuleDict

from . import logger as log


class GateGRU(nn.Module):
    def __init__(self, neurons, input_channels=3, hidden_channels=5, bias=True, offset=0, **kwargs):
        super().__init__()
        log.info("Ignoring input {} when creating {}".format(pformat(kwargs, indent=20), self.__class__.__name__))
        self.gru = nn.GRUCell(input_channels, hidden_channels, bias=bias)
        self.linear = nn.Linear(hidden_channels, neurons)
        self.hidden_states = hidden_channels
        self.offset = offset

    def regularizer(self, subs_idx=None):
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        return self.linear.weight[subs_idx, :].abs().mean()

    def initialize_state(self, batch_size, hidden_size, cuda):
        state_size = [batch_size, hidden_size]
        state = torch.zeros(*state_size)
        if cuda:
            state = state.cuda()
        return state

    def forward(self, input, readoutput=None, subs_idx=None):
        N, T, f = input.size()
        states = []

        hidden = self.initialize_state(N, self.hidden_states, input.is_cuda)
        x = input.transpose(0, 1)
        for t in range(T):
            hidden = self.gru(x[t, ...], hidden)
            states.append(self.linear(hidden))
        if readoutput is None:
            log.info("Nothing to modulate. Returning modulation only")
            return torch.exp(torch.stack(states, 1))
        else:
            lag = T - readoutput.size(1)
        states = torch.exp(torch.stack(states, 1)[:, lag:, :])
        if subs_idx is not None:
            states = states[..., subs_idx]
        return readoutput * (states + self.offset)


class GRUModulator(ModuleDict):
    _base_modulator = None

    def __init__(
        self, n_neurons, input_channels=3, hidden_channels=5, bias=True, gamma_modulator=0, offset=0, **kwargs
    ):
        log.info("Ignoring input {} when creating {}".format(pformat(kwargs, indent=20), self.__class__.__name__))
        super().__init__()
        self.gamma_modulator = gamma_modulator
        for k, n in n_neurons.items():
            self.add_module(k, self._base_modulator(n, input_channels, hidden_channels, bias, offset))

    def initialize(self):
        log.info("Initializing " + self.__class__.__name__)
        for k, mu in self.items():
            self[k].gru.reset_parameters()

    def regularizer(self, data_key, subs_idx=None):
        return self[data_key].regularizer(subs_idx=subs_idx) * self.gamma_modulator


class GateGRUModulator(GRUModulator):
    _base_modulator = GateGRU


def NoModulator(*args, **kwargs):
    """
    Dummy function to create an object that returns None
    Args:
        *args:   will be ignored
        *kwargs: will be ignored

    Returns:
        None
    """
    return None
