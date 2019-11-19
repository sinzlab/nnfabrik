from mlutils.layers.readouts import PointPooled2d
from mlutils.layers.cores import Core2d, Core
from ..utility.nn_helpers import get_io_dims, get_module_output, set_random_seed, get_dims_for_loader_dict

from torch.nn import functional as F
import numpy as np

from torch import nn

import torchvision
from torchvision.models import vgg16, alexnet, vgg19


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