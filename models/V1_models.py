import numpy as np
import torch
from mlutils.layers.readouts import PointPooled2d
from mlutils.layers.cores import Stacked2dCore
from torch import nn as nn
from utility.helper_functions import *
from torch.nn import functional as F


def stacked2d_core_point_readout(dataloader, seed,
                   input_channels=1, hidden_channels=32, input_kern=15,
                   hidden_kern=7, layers=3, gamma_hidden=0, gamma_input=0.1,
                   skip=0, final_nonlinearity=True, core_bias=False, momentum=0.7,
                   pad_input=True, batch_norm=True, hidden_dilation=1,
                   pool_steps=2, pool_kern=7, readout_bias=True, init_range=0.1):

    class Encoder(nn.Module):

        def __init__(self, core, readout):
            super().__init__()
            self.core = core
            self.readout = readout

        def forward(self, x):
            x = self.core(x)
            x = self.readout(x)
            return F.elu(x) + 1

    np.random.seed(seed)
    torch.manual_seed(seed)

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
                         hidden_dilation=hidden_dilation)

    input_shape, output_shape = get_in_out_dimensions(dataloader)
    readout_in_shape = get_core_output_shape(core,input_shape)
    num_neurons=output_shape[1]

    # get a PointPooled Readout from mlutils
    readout = PointPooled2d(readout_in_shape,
                            num_neurons,
                            pool_steps=pool_steps,
                            pool_kern=pool_kern,
                            bias=readout_bias,
                            init_range=init_range)
    gamma_readout = 0.5
    def regularizer():
        return readout.feature_l1() * gamma_readout
    readout.regularizer = regularizer



    train_loader = dataloader["train_loader"]

    n_batches = 25
    train_responses = np.zeros((n_batches, num_neurons))
    train_valid_responses = np.zeros((n_batches, num_neurons))
    for i in range(n_batches):
        _, responses, valid_responses = next(iter(train_loader))
        train_responses[i, :] = responses.mean((0)).cpu().numpy()
        train_valid_responses[i, :] = valid_responses.mean((0)).cpu().numpy()

    avg_responses = np.mean(train_responses / train_valid_responses, axis=0)

    model = Encoder(core, readout)
    model.core.initialize()
    model.readout.bias.data = torch.as_tensor(avg_responses).to(torch.float).mean((0))
    return model



