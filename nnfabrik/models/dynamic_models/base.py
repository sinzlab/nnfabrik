import numpy as np
import torch
from torch.nn.parallel import data_parallel

from .misc import log1exp, Elu1
from torch import nn as nn
from torch.autograd import Variable
from . import logger as log


class _CorePlusReadoutBase(nn.Module):
    def __init__(self, core, readout, modulator=None, nonlinearity=None, shifter=None):
        super().__init__()

        self.core = core
        self.readout = readout
        self.modulator = modulator
        self.shifter = shifter
        self.nonlinearity = Elu1 if nonlinearity is None else nonlinearity

        self._shift = shifter is not None
        self._modulate = modulator is not None
        self.readout_gpu = None

    @property
    def shift(self):
        return self._shift

    @shift.setter
    def shift(self, val):
        self._shift = val and self.shifter is not None

    @property
    def modulate(self):
        return self._modulate

    @modulate.setter
    def modulate(self, val):
        self._modulate = val and self.modulator is not None

    @staticmethod
    def get_readout_in_shape(core, in_shape):
        mov_shape = in_shape[1:]
        core.eval()
        tmp = Variable(torch.from_numpy(np.random.randn(1, *mov_shape).astype(np.float32)))
        nout = core(tmp).size()[1:]
        core.train(True)
        return nout


class CorePlusReadout3d(_CorePlusReadoutBase):
    def __init__(self, core, readout, modulator=None, nonlinearity=None, shifter=None, burn_in=15):
        super().__init__(core, readout, modulator=modulator, nonlinearity=nonlinearity, shifter=shifter)
        self.burn_in = burn_in

    @property
    def state(self):
        return dict(shift=self.shift, modulate=self.modulate, burn_in=self.burn_in)

    def forward(self, x, readout_key, behavior=None, eye_pos=None, subs_idx=None):
        timesteps = x.size(2)
        x = self.core(x)

        if eye_pos is not None and self.shifter is not None and self.shift:
            shift = self.shifter[readout_key](eye_pos)
            if isinstance(x, tuple):
                lag = shift.size(1) - x[0].size(2)
            else:
                lag = shift.size(1) - x.size(2)

            shift = shift[:, lag:, ...]
        else:
            shift = None

        if self.readout_gpu is not None:
            module_kwargs = dict(shift=shift.cuda(1) if shift is not None else None, subs_idx=subs_idx)
            n_gpu = torch.cuda.device_count()

            if x.size(0) > 1:
                device_ids = list(range(1, min(n_gpu, x.size(0))))
                x = data_parallel(
                    self.readout[readout_key], x.cuda(1), module_kwargs=module_kwargs, device_ids=device_ids
                ).cuda(0)
            else:
                x = self.readout[readout_key](x.cuda(1), **module_kwargs).cuda(0)
        else:
            x = self.readout[readout_key](x, shift=shift, subs_idx=subs_idx)

        x = self.nonlinearity(x)

        if behavior is not None and self.modulator is not None and self.modulate:
            x = self.modulator[readout_key](behavior, x, subs_idx=subs_idx)

        if self.burn_in < timesteps - x.size(1):
            log.warning("WARNING: burn in is smaller than induced lag")

        burn_in = max(0, self.burn_in - timesteps + x.size(1))

        return x[:, burn_in:, :]

    def cuda(self):
        n_gpu = torch.cuda.device_count()
        self = super().cuda()
        if n_gpu > 1:
            self.readout.cuda(1)
            self.readout_gpu = 1
        return self

    def __repr__(self):
        s = super().__repr__()
        s += " [{} parameters: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: "burn" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"
