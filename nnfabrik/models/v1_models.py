from mlutils.layers.readouts import PointPooled2d
from mlutils.layers.cores import Stacked2dCore
from torch import nn as nn
from utility.nn_helpers import get_io_dims, get_module_output, set_random_seed
from torch.nn import functional as F


def stacked2d_core_point_readout(dataloader, seed, hidden_channels=32, input_kern=15,
                                 hidden_kern=7, layers=3, gamma_hidden=0, gamma_input=0.1,
                                 skip=0, final_nonlinearity=True, core_bias=False, momentum=0.9,
                                 pad_input=True, batch_norm=True, hidden_dilation=1,
                                 pool_steps=2, pool_kern=7, readout_bias=True, init_range=0.1,
                                 gamma_readout=0.5, laplace_padding=None):
    input_shape, output_shape = get_io_dims(dataloader["train_loader"])
    input_channels = input_shape[1]

    class Encoder(nn.Module):

        def __init__(self, core, readout):
            super().__init__()
            self.core = core
            self.readout = readout

        def forward(self, x):
            output_of_the_core = self.core(x)
            readout_output = self.readout(output_of_the_core)
            return F.elu(readout_output) + 1

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

    readout_in_shape = get_module_output(core, input_shape)
    num_neurons = output_shape[1]
    # get a PointPooled Readout from mlutils
    readout = PointPooled2d(readout_in_shape[1:],
                            num_neurons,
                            pool_steps=pool_steps,
                            pool_kern=int(pool_kern),
                            bias=readout_bias,
                            init_range=init_range)

    gamma_readout = gamma_readout = 0.5

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

