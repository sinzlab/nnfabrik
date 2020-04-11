import logging
import sys
from collections import OrderedDict
from ...utility.nn_helpers import get_module_output
from .misc import Elu1


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(filename)-20s%(lineno)4d:\t %(message)s", datefmt="%d-%m-%Y:%H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)


from . import cores, modulators, readouts, shifters, base


def build_core(
    input_channels,
    hidden_channels=12,
    rec_channels=36,
    input_kern=13,
    hidden_kern=3,
    rec_kern=1,
    layers=3,
    gamma_hidden=0.1,
    gamma_input=20.0,
    gamma_rec=0.1,
    momentum=0.1,
    bias=True,
    skip=2,
    pad_input=1,
):
    return cores.StackedFeatureGRUCore(
        input_channels,
        hidden_channels=hidden_channels,
        rec_channels=rec_channels,
        input_kern=input_kern,
        hidden_kern=hidden_kern,
        rec_kern=rec_kern,
        layers=layers,
        gamma_hidden=gamma_hidden,
        gamma_input=gamma_input,
        gamma_rec=gamma_rec,
        momentum=momentum,
        bias=bias,
        skip=skip,
        pad_input=pad_input,
    )


def build_readout(in_shape, neurons, gamma_features=1.0, positive=False, pool_steps=2, kernel_size=4, stride=4):
    return readouts.STPool3dReadout(
        in_shape=in_shape,
        neurons=neurons,
        gamma_features=gamma_features,
        positive=positive,
        pool_steps=pool_steps,
        kernel_size=kernel_size,
        stride=stride,
    )


def build_modulator(data_keys, input_channels, hidden_channels=50, bias=True, gamma_modulator=0, offset=1):
    return modulators.GateGRUModulator(
        data_keys,
        input_channels,
        hidden_channels=hidden_channels,
        bias=bias,
        gamma_modulator=gamma_modulator,
        offset=offset,
    )


def build_shifter(data_keys, input_channels, gamma_shifter=1e-3, bias=True):
    return shifters.StaticAffineShifter(data_keys, input_channels, gamma_shifter=gamma_shifter, bias=bias)


def build_network(
    dataloaders, burn_in=15, core_config=None, readout_config=None, shifter_config=None, modulator_config=None
):

    trainloader = dataloaders["train_loader"]
    trainset = trainloader.dataset
    input_shape = trainset.img_shape
    input_channels = input_shape[1]
    eye_pos_channels = trainset[0].eye_position.shape[-1]
    beh_channels = trainset[0].behavior.shape[-1]
    n_neurons = {"dataset": trainset.n_neurons}

    core_config = {} if core_config is None else core_config
    readout_config = {} if readout_config is None else readout_config
    shifter_config = {} if shifter_config is None else shifter_config
    modulator_config = {} if modulator_config is None else modulator_config

    core = build_core(input_channels, **core_config)

    # ro_input_shape = base.CorePlusReadout3d.get_readout_in_shape(core, input_shape)
    ro_input_shape = get_module_output(core, (1,) + input_shape[1:])[1:]

    readout = build_readout(ro_input_shape, n_neurons, **readout_config)
    shifter = build_shifter(n_neurons, input_channels=eye_pos_channels, **shifter_config)
    modulator = build_modulator(n_neurons, input_channels=beh_channels, **modulator_config)

    model = base.CorePlusReadout3d(
        core, readout, nonlinearity=Elu1(), shifter=shifter, modulator=modulator, burn_in=burn_in
    )

    # initialize the model
    mu_dict = {"dataset": trainset.transformed_mean().responses}
    model.readout.initialize(mu_dict)
    model.core.initialize()

    if model.shifter is not None:
        biases = {"dataset": -trainset.transformed_mean().eye_position}
        model.shifter.initialize(bias=biases)

    if model.modulator is not None:
        model.modulator.initialize()

    return model
