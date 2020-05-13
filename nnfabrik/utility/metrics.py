import warnings

import numpy as np
import torch
from scipy import stats

from mlutils.measures import PoissonLoss, corr
from mlutils.training import eval_state


def model_predictions(loader, model, data_key, device):
    """
        computes model predictions for a given dataloader and a model
        Returns:
            target: ground truth, i.e. neuronal firing rates of the neurons
            output: responses as predicted by the network
        """
    target, output = torch.empty(0), torch.empty(0)
    for images, responses in loader[data_key]:
        output = torch.cat((output, (model(images.to(device), data_key=data_key).detach().cpu())), dim=0)
        target = torch.cat((target, responses.detach().cpu()), dim=0)

    return target.numpy(), output.numpy()


def corr_stop(model, loader, avg=True, device="cpu"):
    """
    Returns either the average correlation of all neurons or the the correlations per neuron.
        Gets called by early stopping and the model performance evaluation
    """

    n_neurons, correlations_sum = 0, 0
    if not avg:
        all_correlations = np.array([])

    for data_key in loader:
        with eval_state(model):
            target, output = model_predictions(loader, model, data_key, device)

        ret = corr(target, output, axis=0)

        if np.any(np.isnan(ret)):
            warnings.warn("{}% NaNs ".format(np.isnan(ret).mean() * 100))
        ret[np.isnan(ret)] = 0

        if not avg:
            all_correlations = np.append(all_correlations, ret)
        else:
            n_neurons += output.shape[1]
            correlations_sum += ret.sum()

    corr_ret = correlations_sum / n_neurons if avg else all_correlations
    return corr_ret


def poisson_stop(model, loader, avg=False, device="cpu"):
    poisson_losses = np.array([])
    n_neurons = 0
    for data_key in loader:
        with eval_state(model):
            target, output = model_predictions(loader, model, data_key, device)

        ret = output - target * np.log(output + 1e-12)
        if np.any(np.isnan(ret)):
            warnings.warn(" {}% NaNs ".format(np.isnan(ret).mean() * 100))

        poisson_losses = np.append(poisson_losses, np.nansum(ret, 0))
        n_neurons += output.shape[1]
    return poisson_losses.sum() / n_neurons if avg else poisson_losses.sum()
