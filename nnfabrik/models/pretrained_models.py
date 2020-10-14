from neuralpredictors.layers.readouts import PointPooled2d
from neuralpredictors.layers.cores import Core2d, Core
from ..utility.nn_helpers import get_io_dims, get_module_output, set_random_seed, get_dims_for_loader_dict

from itertools import count
import numpy as np

from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import vgg16, alexnet, vgg19


class TransferLearningCore(Core2d, nn.Module):
    """
    A Class to create a Core based on a model class from torchvision.models.
    """

    def __init__(
        self,
        input_channels,
        tr_model_fn,
        model_layer,
        pretrained=True,
        final_batchnorm=True,
        final_nonlinearity=True,
        bias=False,
        momentum=0.1,
        fine_tune=False,
        **kwargs
    ):
        """
        Args:
            input_channels: number of input channgels
            tr_model_fn: string to specify the pretrained model, as in torchvision.models, e.g. 'vgg16'
            model_layer: up onto which layer should the pretrained model be built
            pretrained: boolean, if pretrained weights should be used
            final_batchnorm: adds a batch norm layer
            final_nonlinearity: adds a nonlinearity
            bias: Adds a bias term. currently unused.
            momentum: batch norm momentum
            fine_tune: boolean, sets all weights to trainable if True
            **kwargs:
        """
        print("Ignoring input {} when creating {}".format(repr(kwargs), self.__class__.__name__))
        super().__init__()

        # getattr(self, tr_model_fn)
        tr_model_fn = globals()[tr_model_fn]

        self.input_channels = input_channels
        self.tr_model_fn = tr_model_fn

        tr_model = tr_model_fn(pretrained=pretrained)
        self.model_layer = model_layer
        self.features = nn.Sequential()

        tr_features = nn.Sequential(*list(tr_model.features.children())[:model_layer])

        # Fix pretrained parameters during training parameters
        if not fine_tune:
            for param in tr_features.parameters():
                param.requires_grad = False

        self.features.add_module("TransferLearning", tr_features)
        print(self.features)
        if final_batchnorm:
            self.features.add_module("OutBatchNorm", nn.BatchNorm2d(self.outchannels, momentum=momentum))
        if final_nonlinearity:
            self.features.add_module("OutNonlin", nn.ReLU(inplace=True))

    def forward(self, x):
        if self.input_channels == 1:
            x = x.expand(-1, 3, -1, -1)
        return self.features(x)

    def regularizer(self):
        return 0

    @property
    def outchannels(self):
        """
        Returns: dimensions of the output, after a forward pass through the model
        """
        found_out_channels = False
        i = 1
        while not found_out_channels:
            if "out_channels" in self.features.TransferLearning[-i].__dict__:
                found_out_channels = True
            else:
                i = i + 1
        return self.features.TransferLearning[-i].out_channels
