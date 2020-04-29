from pprint import pformat

from torch import nn
from torch.nn import functional as F, ModuleDict
from . import logger as log


class Shifter:
    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: "gamma" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


class StaticAffineShifter(Shifter, ModuleDict):
    def __init__(self, data_keys, input_channels, bias=True, gamma_shifter=0, **kwargs):
        log.info("Ignoring input {} when creating {}".format(pformat(kwargs, indent=20), self.__class__.__name__))
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(k, StaticAffine(input_channels, 2, bias=bias))

    def initialize(self, bias=None):
        log.info("Initializing affine weights")
        for k in self:
            if bias is not None:
                self[k].initialize(bias=bias[k])
            else:
                self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].weight.pow(2).mean() * self.gamma_shifter


class StaticAffine(nn.Linear):
    def __init__(self, input_channels, output_channels, bias=True, **kwargs):
        log.info("Ignoring input {} when creating {}".format(pformat(kwargs, indent=20), self.__class__.__name__))
        super().__init__(input_channels, output_channels, bias=bias)

    def forward(self, input):
        N, T, f = input.size()
        x = input.view(N * T, f)
        x = super().forward(x)
        return F.tanh(x.view(N, T, -1))

    def initialize(self, bias=None):
        self.weight.data.normal_(0, 1e-6)
        if self.bias is not None:
            if bias is not None:
                log.info("Setting bias to predefined value " + repr(bias))
                self.bias.data = bias
            else:
                self.bias.data.normal_(0, 1e-6)


def NoShifter(*args, **kwargs):
    """
    Dummy function to create an object that returns None
    Args:
        *args:   will be ignored
        *kwargs: will be ignored

    Returns:
        None
    """
    return None
