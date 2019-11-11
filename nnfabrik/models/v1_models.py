from mlutils.layers.readouts import PointPooled2d
from mlutils.layers.cores import Stacked2dCore
from torch import nn as nn
from ..utility.nn_helpers import get_io_dims, get_module_output, set_random_seed, get_dims_dict
from torch.nn import functional as F



class PointPooled2dReadout(nn.ModuleDict):
    def __init__(self, in_shape, n_neurons, gamma_readout):
        for k, neurons in n_neurons:
            self.add_module(k, PointPooled2d(in_shape,
                            neurons,
                            pool_steps=pool_steps,
                            pool_kern=pool_kern,
                            bias=readout_bias,
                            init_range=init_range))

        self.gamma_reaodut =gamma_readout

    def forward(self, *args, key=None, **kwargs):
        if key is None and len(self) == 1:
            key = list(self.keys())[0]
        return self[key](*args, **kwargs)


    def regularizer(self, data_key):
        return self[data_key].feature_l1() * self.gamma_readout



def stacked2d_core_point_readout(dataloaders, seed, hidden_channels=32, input_kern=15,
                                 hidden_kern=7, layers=3, gamma_hidden=0, gamma_input=0.1,
                                 skip=0, final_nonlinearity=True, core_bias=False, momentum=0.9,
                                 pad_input=True, batch_norm=True, hidden_dilation=1,
                                 pool_steps=2, pool_kern=7, readout_bias=True, init_range=0.1,
                                 gamma_readout=0.5, laplace_padding=None):
    # use all shape dict instead
    shape_dict = get_io_dims(list(dataloaders.values())[0])
    input_channels = shape_dict['inputs'][1]


    all_shape_dict = get_dims_dict(dataloaders)

    n_neurons = {k: v['targets'][1] for k, v in all_shape_dict}

    class Encoder(nn.Module):

        def __init__(self, core, readout):
            super().__init__()
            self.core = core
            self.readout = readout

        def forward(self, x, key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, key=key)
            return F.elu(x) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    # get a stacked2D core from mlutils
    core = Stacked2dCore(input_channels=input_channels,
                         hidden_channels=hidden_channels,
                         input_kern=input_kern,
                         hidden_kern=hidden_kern,
                         layers=layers,
                         gamma_hidden=gamma_hidden,
                         gamma_input=gamma_input,
                         skip=skip,
                         final_nonlinearity=final_nonlinearity,
                         bias=core_bias,
                         momentum=momentum,
                         pad_input=pad_input,
                         batch_norm=batch_norm,
                         hidden_dilation=hidden_dilation,
                         laplace_padding=laplace_padding)

    readout_in_shape = get_module_output(core, shape_dict['inputs'])

    # get a PointPooled Readout from mlutils
    readout = PointPooled2dReadout(readout_in_shape[1:],
                            n_neurons,
                            pool_steps=pool_steps,
                            pool_kern=pool_kern,
                            bias=readout_bias,
                            init_range=init_range)

    gamma_readout = 0.5
    def regularizer():
        return readout.feature_l1() * gamma_readout

    readout.regularizer = regularizer

    _, train_responses, weights = dataloader["train_loader"].dataset[:]
    # initialize readout bias by avg firing rate, scaled by how many images the neuron has seen.
    avg_responses = train_responses.mean(0) / weights.mean(0)

    model = Encoder(core, readout)
    model.core.initialize()
    model.readout.bias.data = avg_responses
    return model

