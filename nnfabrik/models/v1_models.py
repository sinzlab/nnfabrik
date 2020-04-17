import numpy as np
from torch import nn as nn
from torch.nn import functional as F

from mlutils.layers.readouts import PointPooled2d
from mlutils.layers.cores import Stacked2dCore
from mlutils.training import eval_state

from .pretrained_models import TransferLearningCore
from ..utility.nn_helpers import get_io_dims, get_module_output, set_random_seed, get_dims_for_loader_dict

class MultiplePointPooled2d(nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, pool_steps, pool_kern, bias, init_range, gamma_readout, readout_reg_avg):
        # super init to get the _module attribute
        super(MultiplePointPooled2d, self).__init__()
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
        self.readout_reg_avg = readout_reg_avg

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)


    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=self.readout_reg_avg) * self.gamma_readout


def stacked2d_core_point_readout(dataloaders, seed, hidden_channels=32, input_kern=13,          # core args
                                 hidden_kern=3, layers=3, gamma_hidden=0, gamma_input=0.1,
                                 skip=0, final_nonlinearity=True, core_bias=False, momentum=0.9,
                                 pad_input=False, batch_norm=True, hidden_dilation=1,
                                 laplace_padding=None, input_regularizer='LaplaceL2norm',
                                 pool_steps=2, pool_kern=7, readout_bias=True, init_range=0.1,  # readout args,
                                 gamma_readout=0.1,  elu_offset=0, stack=None, readout_reg_avg=False,
                                 use_avg_reg=False):
    """
    Model class of a stacked2dCore (from mlutils) and a pointpooled (spatial transformer) readout

    Args:
        dataloaders: a dictionary of dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: random seed
        elu_offset: Offset for the output non-linearity [F.elu(x + self.offset)]

        all other args: See Documentation of Stacked2dCore in mlutils.layers.cores and
            PointPooled2D in mlutils.layers.readouts

    Returns: An initialized model which consists of model.core and model.readout
    """
    

    # make sure trainloader is being used
    dataloaders = dataloaders.get("train", dataloaders)

    # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
    in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]
    
    assert np.unique(input_channels).size == 1, "all input channels must be of equal size"

    class Encoder(nn.Module):

        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.core.regularizer() + self.readout.regularizer(data_key=data_key)

        def _readout_regularizer_val(self):
            ret = 0
            with eval_state(model):
                for data_key in model.readout:
                    ret += self.readout.regularizer(data_key).detach().cpu().numpy()
            return ret

        def _core_regularizer_val(self):
            with eval_state(model):
                return self.core.regularizer().detach().cpu().numpy() if model.core.regularizer() else 0

        @property
        def tracked_values(self):
            return dict(readout_l1=self._readout_regularizer_val, 
                        core_reg=self._core_regularizer_val)

    set_random_seed(seed)

    # get a stacked2D core from mlutils
    core = Stacked2dCore(input_channels=input_channels[0],
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
                         laplace_padding=laplace_padding,
                         input_regularizer=input_regularizer,
                         stack=stack,
                         use_avg_reg=use_avg_reg)

    readout = MultiplePointPooled2d(core,
                                    in_shape_dict=in_shapes_dict,
                                    n_neurons_dict=n_neurons_dict,
                                    pool_steps=pool_steps,
                                    pool_kern=pool_kern,
                                    bias=readout_bias,
                                    init_range=init_range,
                                    gamma_readout=gamma_readout,
                                    readout_reg_avg=readout_reg_avg)

    # initializing readout bias to mean response
    if readout_bias:
        for k in dataloaders:
            readout[k].bias.data = dataloaders[k].dataset[:][1].mean(0)

    model = Encoder(core, readout, elu_offset)

    return model


def vgg_core_point_readout(dataloaders, seed,
                           input_channels=1, tr_model_fn='vgg16', # begin of core args
                           model_layer=11, momentum=0.1, final_batchnorm=True,
                           final_nonlinearity=True, bias=False,
                           pool_steps=1, pool_kern=7, readout_bias=True, # begin or readout args
                           init_range=0.1, gamma_readout=0.002, elu_offset=-1, readout_reg_avg=False):
    """
    A Model class of a predefined core (using models from torchvision.models). Can be initialized pretrained or random.
    Can also be set to be trainable or not, independent of initialization.

    Args:
        dataloaders: a dictionary of train-dataloaders, one loader per session
            in the format {'data_key': dataloader object, .. }
        seed: ..
        pool_steps:
        pool_kern:
        readout_bias:
        init_range:
        gamma_readout:

    Returns:
    """

    if "train" in dataloaders.keys():
        dataloaders = dataloaders["train"]

    in_name, out_name = next(iter(list(dataloaders.values())[0]))._fields

    session_shape_dict = get_dims_for_loader_dict(dataloaders)
    n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
    in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
    input_channels = [v[in_name][1] for v in session_shape_dict.values()]

    class Encoder(nn.Module):
        """
        helper nn class that combines the core and readout into the final model
        """
        def __init__(self, core, readout, elu_offset):
            super().__init__()
            self.core = core
            self.readout = readout
            self.offset = elu_offset

        def forward(self, x, data_key=None, **kwargs):
            x = self.core(x)
            x = self.readout(x, data_key=data_key)
            return F.elu(x + self.offset) + 1

        def regularizer(self, data_key):
            return self.readout.regularizer(data_key=data_key) + self.core.regularizer()

        def _readout_regularizer_val(self):
            ret = 0
            with eval_state(model):
                for data_key in model.readout:
                    ret += self.readout.regularizer(data_key).detach().cpu().numpy()
            return ret

        @property
        def tracked_values(self):
            return dict(readout_l1=self._readout_regularizer_val)
            

    set_random_seed(seed)

    core = TransferLearningCore(input_channels=input_channels[0],
                                tr_model_fn=tr_model_fn,
                                model_layer=model_layer,
                                momentum=momentum,
                                final_batchnorm=final_batchnorm,
                                final_nonlinearity=final_nonlinearity,
                                bias=bias)

    readout = MultiplePointPooled2d(core, in_shape_dict=in_shapes_dict,
                                    n_neurons_dict=n_neurons_dict,
                                    pool_steps=pool_steps,
                                    pool_kern=pool_kern,
                                    bias=readout_bias,
                                    init_range=init_range,
                                    gamma_readout=gamma_readout,
                                    readout_reg_avg=readout_reg_avg)

    # initializing readout bias to mean response
    if readout_bias:
        for k in dataloaders:
            readout[k].bias.data = dataloaders[k].dataset[:][1].mean(0)

    model = Encoder(core, readout, elu_offset)

    return model
