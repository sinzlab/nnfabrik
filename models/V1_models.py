
import numpy as np

import torch

from mlutils.layers.readouts import PointPooled2d
from mlutils.layers.cores import Stacked2dCore
from torch import nn


def Core2D_PointRO(input_shape, output_shape, seed, param1=1):

    class Encoder(nn.Module):

        def __init__(self, core, readout):
            super().__init__()
            self.core = core
            self.readout = readout

        @staticmethod
        def get_readout_in_shape(core, input_shape):
            train_state = core.training
            core.eval()
            input_tensor = torch.ones(input_shape)
            tensor_out = core(input_tensor).shape[1:]
            core.train(train_state)
            return tensor_out

        def forward(self, x):
            x = self.core(x)
            x = self.readout(x)

    core = Stacked2dCore(input_channels=1,
                         hidden_channels=32,
                         input_kern=15,
                         hidden_kern=7,
                         layers=3,
                         gamma_hidden=0,
                         gamma_input=0.1,
                         skip=0,
                         final_nonlinearity=True,
                         bias=False,
                         momentum=0.75,
                         pad_input=True,
                         batch_norm=True,
                         hidden_dilation=1)


    readout_in_shape = Encoder.get_readout_in_shape(core,input_shape)
    print(readout_in_shape)
    num_neurons=output_shape[1]
    print("number of neurons")
    print(num_neurons)

    readout = PointPooled2d(readout_in_shape,
                            num_neurons,
                            pool_steps=2,
                            pool_kern=7,
                            bias=True,
                            init_range=0.1)


    # Readout
    gamma_readout = 0.5
    def regularizer():
        return readout.feature_l1() * gamma_readout
    readout.regularizer = regularizer

    np.random.seed(seed)
    model = Encoder(core, readout)
    # initialize the readout
    model.readout.bias.data = torch.as_tensor(responses).to(torch.float).mean((0))
    # initialize the core
    model.core.initialize()
    model = model.cuda()

    return model