from mlutils.layers.readouts import PointPooled2d
from mlutils.layers.cores import Core2d, Core
from ..utility.nn_helpers import get_io_dims, get_module_output, set_random_seed, get_dims_for_loader_dict

from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from torch import nn

import torchvision
from torchvision.models import vgg16, alexnet, vgg19

# class Core:
#     def initialize(self):
#         raise NotImplementedError('Not initializing')
#
#     def __repr__(self):
#         s = super().__repr__()
#         s += ' [{} regularizers: '.format(self.__class__.__name__)
#         ret = []
#         for attr in filter(lambda x: 'gamma' in x or 'skip' in x, dir(self)):
#             ret.append('{} = {}'.format(attr, getattr(self, attr)))
#         return s + '|'.join(ret) + ']\n'
#
#
# class Core2d(Core):
#     def initialize(self, cuda=False):
#         self.apply(self.init_conv)
#         self.put_to_cuda(cuda=cuda)
#
#     def put_to_cuda(self, cuda):
#         if cuda:
#             self = self.cuda()
#
#     @staticmethod
#     def init_conv(m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.xavier_normal_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.fill_(0)


class TransferLearningCore(Core2d, nn.Module):
    'Core feed forward Transfer Learning Model'
    def __init__(self, input_channels, tr_model_fn, model_layer, pretrained=True,
                 final_batchnorm=True, final_nonlinearity=True,
                 bias=False, momentum=0.1, fine_tune = False, **kwargs):
        print('Ignoring input {} when creating {}'.format(repr(kwargs), self.__class__.__name__))
        super().__init__()
        #getattr(self, tr_model_fn)
        tr_model_fn         = globals()[tr_model_fn] # take the string name of the function
        self.input_channels = input_channels
        self.tr_model_fn    = tr_model_fn    # torchvision.models functions (e.g. 'vgg16_bn')
        tr_model            = tr_model_fn(pretrained = pretrained) # model
        self.model_layer    = model_layer    # Number of output layer
        self.features = nn.Sequential()
        tr_features = nn.Sequential(*list(tr_model.features.children())[:model_layer])
        # Fix pretrained parameters during training parameters
        if not fine_tune:
            for param in tr_features.parameters():
                param.requires_grad = False
        self.features.add_module('TransferLearning', tr_features)
        print(self.features)
        if final_batchnorm:
            self.features.add_module('OutBatchNorm', nn.BatchNorm2d(self.outchannels, momentum=momentum))
        if final_nonlinearity:
            self.features.add_module('OutNonlin', nn.ReLU(inplace=True))
    def forward(self, input_):
        if self.input_channels == 1:
            input_ = input_.repeat(1,3,1,1)
        input_ = self.features(input_)
        return input_
    def regularizer(self):
        return 0
    @property
    def outchannels(self):
        'To consider: forward a random image and get output shape'
        found_out_channels = False
        i=1
        while not found_out_channels:
            if 'out_channels' in self.features.TransferLearning[-i].__dict__:
                found_out_channels = True
            else:
                i = i+1
        return self.features.TransferLearning[-i].out_channels


class PointPooled2dReadout(nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, pool_steps, pool_kern, bias, init_range, gamma_readout):
        # super init to get the _module attribute
        super(PointPooled2dReadout, self).__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, PointPooled2d(
                            in_shape,
                            n_neurons,
                            pool_steps=pool_steps,
                            pool_kern=pool_kern,
                            bias=bias,
                            init_range=init_range)
                            )

        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1() * self.gamma_readout


def vgg_core_point_readout(dataloaders, seed, pool_steps=1,
                           pool_kern=7, readout_bias=True, init_range=0.1,
                           gamma_readout=0.1):

    session_shape_dict = get_dims_for_loader_dict(dataloaders)

    n_neurons_dict = {k: v['targets'][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v['inputs'] for k, v in session_shape_dict.items()}
    input_channels = [v['inputs'][1] for _, v in session_shape_dict.items()]
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"

    class Encoder(nn.Module):

        def __init__(self, core, readout):
            super().__init__()
            self.core = core
            self.readout = readout

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x) + 1

        # core regularizer is omitted because it's pretrained
        def regularizer(self, data_key):
            return self.readout.regularizer(data_key=data_key)

    set_random_seed(seed)

    core = TransferLearningCore(input_channels=1, tr_model_fn='vgg16',
                                model_layer=11, momentum=0.1, final_batchnorm=True,
                                final_nonlinearity=True, bias=False)

    readout = PointPooled2dReadout(core, in_shape_dict=in_shapes_dict,
                                   n_neurons_dict=n_neurons_dict,
                                   pool_steps=pool_steps,
                                   pool_kern=pool_kern,
                                   bias=readout_bias,
                                   init_range=init_range,
                                   gamma_readout=gamma_readout)

    # initializing readout bias to mean response
    for k in dataloaders:
        readout[k].bias.data = dataloaders[k].dataset[:][1].mean(0)

    model = Encoder(core, readout)
    return model

