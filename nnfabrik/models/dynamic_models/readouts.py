from pprint import pformat

from neuralpredictors.layers.readouts import SpatialTransformerPooled3d
from torch.nn import ModuleDict

from . import logger as log


class Readout:
    def initialize(self, *args, **kwargs):
        raise NotImplementedError("initialize is not implemented for ", self.__class__.__name__)

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(
            lambda x: not x.startswith("_") and ("gamma" in x or "pool" in x or "positive" in x or "component" in x),
            dir(self),
        ):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"


class PooledReadout(Readout):
    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value

    def initialize(self, mu_dict, grid=True):
        log.info(
            "Initializing with mu_dict: {}. Initialize grid={}".format(
                ", ".join(["{}: {}".format(k, len(m)) for k, m in mu_dict.items()]), grid
            )
        )
        for k, mu in mu_dict.items():
            self[k].initialize(grid=grid)
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key, subs_idx=None):
        return self[readout_key].feature_l1(subs_idx=subs_idx) * self.gamma_features

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        self._pool_steps = value
        for k in self:
            self[k].pool_steps = value


class STPool3dReadout(PooledReadout, ModuleDict):
    def __init__(
        self, in_shape, neurons, positive=False, gamma_features=0, pool_steps=0, stride=2, kernel_size=2, **kwargs
    ):
        log.info("Ignoring input {} when creating {}".format(pformat(kwargs, indent=20), self.__class__.__name__))
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        self._pool_steps = pool_steps
        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(
                k,
                SpatialTransformerPooled3d(
                    in_shape, neur, positive=positive, pool_steps=pool_steps, stride=stride, kernel_size=kernel_size
                ),
            )


#
# class STFactorizedPool3dReadout(PooledReadout, ModuleDict):
#     def __init__(self, in_shape, neurons, positive=False, gamma_features=0, pool_steps=0, stride=2, kernel_size=2,
#                  components=25, **kwargs):
#         log.info('Ignoring input {} when creating {}'.format(pformat(kwargs, indent=20), self.__class__.__name__))
#         super().__init__()
#
#         self.in_shape = in_shape
#         self.neurons = neurons
#         self._positive = positive
#         self.gamma_features = gamma_features
#         self._pool_steps = pool_steps
#         self._components = components
#         for k, neur in neurons.items():
#             if isinstance(self.in_shape, dict):
#                 in_shape = self.in_shape[k]
#             self.add_module(k, FactorizedSpatialTransformerPooled3d(in_shape, neur, positive=positive,
#                                                                     pool_steps=pool_steps,
#                                                                     stride=stride, kernel_size=kernel_size,
#                                                                     components=components))
#
#     @property
#     def components(self):
#         return self._components

#
# class STXPool3dReadout(PooledReadout, ModuleDict):
#     def __init__(self, in_shape, neurons, positive=False, gamma_features=0, gamma_grid=0,
#                  pool_steps=1, stride=4, kernel_size=4, grid_points=10,
#                  **kwargs):
#         log.info('Ignoring input {} when creating {}'.format(pformat(kwargs, indent=20), self.__class__.__name__))
#         super().__init__()
#
#         self.in_shape = in_shape
#         self.neurons = neurons
#         self._positive = positive
#         self.gamma_features = gamma_features
#         self.gamma_grid = gamma_grid
#         self._pool_steps = pool_steps
#         for k, neur in neurons.items():
#             if isinstance(self.in_shape, dict):
#                 in_shape = self.in_shape[k]
#             self.add_module(k, SpatialTransformerXPooled3d(in_shape, neur, positive=positive, pool_steps=pool_steps,
#                                                            stride=stride, kernel_size=kernel_size,
#                                                            grid_points=grid_points))
#
#     def regularizer(self, readout_key, subs_idx=None):
#         return self[readout_key].feature_l1(subs_idx=subs_idx) * self.gamma_features + \
#                self[readout_key].dgrid_l2(subs_idx=subs_idx) * self.gamma_grid
#


class STPool3dSharedGridReadout(PooledReadout, ModuleDict):
    def __init__(
        self, in_shape, neurons, positive=False, gamma_features=0, pool_steps=0, stride=2, kernel_size=2, **kwargs
    ):
        log.info("Ignoring input {} when creating {}".format(pformat(kwargs, indent=20), self.__class__.__name__))
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        self._pool_steps = pool_steps
        grid = None
        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            ro = SpatialTransformerPooled3d(
                in_shape,
                neur,
                positive=positive,
                pool_steps=pool_steps,
                grid=grid,
                stride=stride,
                kernel_size=kernel_size,
            )
            grid = ro.grid
            self.add_module(k, ro)


class STPool3dSharedGridStopGradReadout(PooledReadout, ModuleDict):
    def __init__(
        self,
        in_shape,
        neurons,
        positive=False,
        gamma_features=0,
        pool_steps=0,
        gradient_pass_mod=0,
        kernel_size=2,
        stride=2,
        **kwargs
    ):
        log.info("Ignoring input {} when creating {}".format(pformat(kwargs, indent=20), self.__class__.__name__))
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        self._pool_steps = pool_steps
        old_neur = -1
        for ro_index, (k, neur) in enumerate(neurons.items()):
            if old_neur != neur:
                log.info("Neuron change detected from {} to {}! Resetting grid!".format(old_neur, neur))
                grid = None
            old_neur = neur

            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]

            stop_grad = (ro_index % gradient_pass_mod) != 0
            if stop_grad:
                log.info("Gradient for {} will be blocked".format(k))
            else:
                log.info("Gradient for {} will pass".format(k))
            ro = SpatialTransformerPooled3d(
                in_shape,
                neur,
                positive=positive,
                pool_steps=pool_steps,
                grid=grid,
                stop_grad=stop_grad,
                kernel_size=kernel_size,
                stride=stride,
            )
            grid = ro.grid
            self.add_module(k, ro)
