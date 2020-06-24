import warnings
import numpy as np
import torch
from mlutils.measures import corr
from mlutils.training import eval_state, device_state
import types
import contextlib
import warnings
from .measure_helpers import get_subset_of_repeats, is_ensemble_function


def model_predictions_repeats(model, dataloader, data_key, device='cuda', broadcast_to_target=False):
    """
    Computes model predictions for a dataloader that yields batches with identical inputs along the first dimension.
    Unique inputs will be forwarded only once through the model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons as a list: [num_images][num_reaps, num_neurons]
        output: responses as predicted by the network for the unique images. If broadcast_to_target, returns repeated 
                outputs of shape [num_images][num_reaps, num_neurons] else (default) returns unique outputs of shape [num_images, num_neurons]
    """
    
    target = []
    unique_images = torch.empty(0)
    for images, responses in dataloader:
        if len(images.shape) == 5:
            images = images.squeeze(dim=0)
            responses = responses.squeeze(dim=0)
        
        assert torch.all(torch.eq(images[-1,], images[0,],)), "All images in the batch should be equal"
        unique_images = torch.cat((unique_images, images[0:1, ]), dim=0)
        target.append(responses.detach().cpu().numpy())
    
    # Forward unique images once:
    with eval_state(model) if not is_ensemble_function(model) else contextlib.nullcontext():
        with device_state(model, device) if not is_ensemble_function(model) else contextlib.nullcontext():
            output = model(unique_images.to(device), data_key=data_key).detach().cpu()
    
    output = output.numpy()   
        
    if broadcast_to_target:
        output = [np.broadcast_to(x, target[idx].shape) for idx, x in enumerate(output)]
    
    return target, output
    

def model_predictions(model, dataloader, data_key, device='cpu'):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """

    target, output = torch.empty(0), torch.empty(0)
    for images, responses in dataloader:
        if len(images.shape) == 5:
            images = images.squeeze(dim=0)
            responses = responses.squeeze(dim=0)
        with eval_state(model) if not is_ensemble_function(model) else contextlib.nullcontext():
            with device_state(model, device) if not is_ensemble_function(model) else contextlib.nullcontext():
                output = torch.cat((output, (model(images.to(device), data_key=data_key).detach().cpu())), dim=0)
            target = torch.cat((target, responses.detach().cpu()), dim=0)

    return target.numpy(), output.numpy()


def get_avg_correlations(model, dataloaders, device='cpu', as_dict=False, per_neuron=True, **kwargs):
    """
    Returns correlation between model outputs and average responses over repeated trials
    
    """
    if 'test' in dataloaders:
        dataloaders = dataloaders['test']
    
    correlations = {}
    for k, loader in dataloaders.items():

        # Compute correlation with average targets
        target, output = model_predictions_repeats(dataloader=loader, model=model, data_key=k, device=device, broadcast_to_target=False)
        target_mean = np.array([t.mean(axis=0) for t in target])
        correlations[k] = corr(target_mean, output, axis=0)
        
        # Check for nans
        if np.any(np.isnan(correlations[k])):
            warnings.warn('{}% NaNs , NaNs will be set to Zero.'.format(np.isnan(correlations[k]).mean() * 100))
        correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = np.hstack([v for v in correlations.values()]) if per_neuron else np.mean(np.hstack([v for v in correlations.values()]))
    return correlations


def get_correlations(model, dataloaders, device='cpu', as_dict=False, per_neuron=True, **kwargs):
    correlations = {}
    with eval_state(model) if not is_ensemble_function(model) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            target, output = model_predictions(dataloader=v, model=model, data_key=k, device=device)
            correlations[k] = corr(target, output, axis=0)

            if np.any(np.isnan(correlations[k])):
                warnings.warn('{}% NaNs , NaNs will be set to Zero.'.format(np.isnan(correlations[k]).mean() * 100))
            correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = np.hstack([v for v in correlations.values()]) if per_neuron else np.mean(np.hstack([v for v in correlations.values()]))
    return correlations


def get_poisson_loss(model, dataloaders, device='cpu', as_dict=False, avg=False, per_neuron=True, eps=1e-12):
    poisson_loss = {}
    with eval_state(model) if not is_ensemble_function(model) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            target, output = model_predictions(dataloader=v, model=model, data_key=k, device=device)
            loss = output - target * np.log(output + eps)
            poisson_loss[k] = np.mean(loss, axis=0) if avg else np.sum(loss, axis=0)
    if as_dict:
        return poisson_loss
    else:
        if per_neuron:
            return np.hstack([v for v in poisson_loss.values()])
        else:
            return np.mean(np.hstack([v for v in poisson_loss.values()])) if avg else np.sum(np.hstack([v for v in poisson_loss.values()]))


def get_repeats(dataloader, min_repeats=2):
    # save the responses of all neuron to the repeats of an image as an element in a list
    repeated_inputs = []
    repeated_outputs = []
    for inputs, outputs in dataloader:
        if len(inputs.shape) == 5:
            inputs = np.squeeze(inputs.cpu().numpy(), axis=0)
            outputs = np.squeeze(outputs.cpu().numpy(), axis=0)
        else:
            inputs = inputs.cpu().numpy()
            outputs = outputs.cpu().numpy()
        r, n = outputs.shape  # number of frame repeats, number of neurons
        if r < min_repeats:  # minimum number of frame repeats to be considered for oracle, free choice
            continue
        assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), "Images of oracle trials do not match"
        repeated_inputs.append(inputs)
        repeated_outputs.append(outputs)
    return np.array(repeated_inputs), np.array(repeated_outputs)


def get_oracles(dataloaders, as_dict=False, per_neuron=True):
    oracles = {}
    for k, v in dataloaders.items():
        _, outputs = get_repeats(v)
        oracles[k] = compute_oracle_corr(np.array(outputs))
    if not as_dict:
        oracles = np.hstack([v for v in oracles.values()]) if per_neuron else np.mean(np.hstack([v for v in oracles.values()]))
    return oracles


def get_oracles_corrected(dataloaders, as_dict=False, per_neuron=True):
    oracles = {}
    for k, v in dataloaders.items():
        _, outputs = get_repeats(v)
        oracles[k] = compute_oracle_corr_corrected(np.array(outputs))
    if not as_dict:
        oracles = np.hstack([v for v in oracles.values()]) if per_neuron else np.mean(np.hstack([v for v in oracles.values()]))
    return oracles


def compute_oracle_corr_corrected(repeated_outputs):
    """

    Args:
        repeated_outputs (list or array): array(images, repeats, responses), or a list of lists of repeats per image.

    Returns: the oracle correlations per neuron

    """
    if len(repeated_outputs.shape) == 3:
        var_noise = repeated_outputs.var(axis=1).mean(0)
        var_mean = repeated_outputs.mean(axis=1).var(0)
    else:
        var_noise, var_mean = [], []
        for repeat in repeated_outputs:
            var_noise.append(repeat.var(axis=0))
            var_mean.append(repeat.mean(axis=0))
        var_noise = np.mean(np.array(var_noise), axis=0)
        var_mean = np.var(np.array(var_mean), axis=0)
    return var_mean / np.sqrt(var_mean * (var_mean + var_noise))


def compute_oracle_corr(repeated_outputs):
    if len(repeated_outputs.shape) == 3:
        _, r, n = repeated_outputs.shape
        oracles = (repeated_outputs.mean(axis=1, keepdims=True) - repeated_outputs / r) * r / (r - 1)
        if np.any(np.isnan(oracles)):
            warnings.warn('{}% NaNs when calculating the oracle. NaNs will be set to Zero.'.format(np.isnan(oracles).mean() * 100))
        oracles[np.isnan(oracles)] = 0
        return corr(oracles.reshape(-1, n), repeated_outputs.reshape(-1, n), axis=0)
    else:
        oracles = []
        for outputs in repeated_outputs:
            r, n = outputs.shape
            # compute the mean over repeats, for each neuron
            mu = outputs.mean(axis=0, keepdims=True)
            # compute oracle predictor
            oracle = (mu - outputs / r) * r / (r - 1)

            if np.any(np.isnan(oracle)):
                warnings.warn('{}% NaNs when calculating the oracle. NaNs will be set to Zero.'.format(
                    np.isnan(oracle).mean() * 100))
                oracle[np.isnan(oracle)] = 0

            oracles.append(oracle)
        return corr(np.vstack(repeated_outputs), np.vstack(oracles), axis=0)


def get_fraction_oracles(model, dataloaders, device='cpu', corrected=False):
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    if corrected:
        oracles = get_oracles_corrected(dataloaders=dataloaders, as_dict=False, per_neuron=True)
    else:
        oracles = get_oracles(dataloaders=dataloaders, as_dict=False, per_neuron=True)
    test_correlation = get_correlations(model=model, dataloaders=dataloaders, device=device, as_dict=False, per_neuron=True)
    oracle_performance, _, _, _ = np.linalg.lstsq(np.hstack(oracles)[:, np.newaxis], np.hstack(test_correlation))
    return oracle_performance[0]


def get_explainable_var(dataloaders, as_dict=False, per_neuron=True, repeat_limit=None, randomize=True):
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    explainable_var = {}
    for k ,v in dataloaders.items():
        _, outputs = get_repeats(v)
        if repeat_limit is not None:
            outputs = get_subset_of_repeats(outputs=outputs, repeat_limit=repeat_limit, randomize=randomize)
        explainable_var[k] = compute_explainable_var(outputs)
    if not as_dict:
        explainable_var = np.hstack([v for v in explainable_var.values()]) if per_neuron else np.mean(np.hstack([v for v in explainable_var.values()]))
    return explainable_var


def compute_explainable_var(outputs, eps=1e-9):
    ImgVariance = []
    TotalVar = np.var(np.vstack(outputs), axis=0, ddof=1)
    for out in outputs:
        ImgVariance.append(np.var(out, axis=0, ddof=1))
    ImgVariance = np.vstack(ImgVariance)
    NoiseVar = np.mean(ImgVariance, axis=0)
    explainable_var = (TotalVar - NoiseVar) / (TotalVar + eps)
    return explainable_var


def get_FEV(model, dataloaders, device='cpu', as_dict=False, per_neuron=True, threshold=None):
    """
    Computes the fraction of explainable variance explained (FEVe) per Neuron, given a model and a dictionary of dataloaders.
    The dataloaders will have to return batches of identical images, with the corresponing neuronal responses.

    Args:
        model (object): PyTorch module
        dataloaders (dict): Dictionary of dataloaders, with keys corresponding to "data_keys" in the model
        device (str): 'cuda' or 'gpu
        as_dict (bool): Returns the scores as a dictionary ('data_keys': values) if set to True.
        per_neuron (bool): Returns the grand average if set to True.
        threshold (float): for the avg feve, excludes neurons with a explainable variance below threshold

    Returns:
        FEV (dict, or np.array, or float): Fraction of explainable varianced explained. Per Neuron or as grand average.
    """
    dataloaders = dataloaders["test"] if "test" in dataloaders else dataloaders
    FEV = {}
    with eval_state(model) if not is_ensemble_function(model) else contextlib.nullcontext():
        for data_key, dataloader in dataloaders.items():
            targets, outputs = model_predictions_repeats(model=model,
                                                         dataloader=dataloader,
                                                         data_key=data_key,
                                                         device=device,
                                                         broadcast_to_target=True)
            if threshold is None:
                FEV[data_key] = compute_FEV(targets=targets, outputs=outputs)
            else:
                fev, feve = compute_FEV(targets=targets, outputs=outputs, return_exp_var=True)
                FEV[data_key] = feve[fev>threshold]
    if not as_dict:
        FEV = np.hstack([v for v in FEV.values()]) if per_neuron else np.mean(np.hstack([v for v in FEV.values()]))
    return FEV


def compute_FEV(targets, outputs, return_exp_var=False):
    """

    Args:
        targets (list): Neuronal responses (ground truth) to image repeats. Dimensions: [num_images] np.array(num_reaps, num_neurons)
        outputs (list): Model predictions to the repeated images, with an identical shape as the targets
        return_exp_var (bool): returns the fraction of explainable variance per neuron if set to True

    Returns:
        FEVe (np.array): the fraction of explainable variance explained per neuron
        --- optional: FEV (np.array): the fraction

    """
    ImgVariance = []
    PredVariance = []
    for i, _ in enumerate(targets):
        PredVariance.append((targets[i] - outputs[i]) ** 2)
        ImgVariance.append(np.var(targets[i], axis=0, ddof=1))
    PredVariance = np.vstack(PredVariance)
    ImgVariance = np.vstack(ImgVariance)

    TotalVar = np.var(np.vstack(targets), axis=0, ddof=1)
    NoiseVar = np.mean(ImgVariance, axis=0)
    FEV = (TotalVar - NoiseVar) / TotalVar

    PredVar = np.mean(PredVariance, axis=0)
    FEVe = 1 - (PredVar - NoiseVar) / (TotalVar - NoiseVar)
    return [FEV, FEVe] if return_exp_var else FEVe


def get_model_rf_size(model_config):
    layers = model_config["layers"]
    input_kern = model_config["input_kern"]
    hidden_kern = model_config["hidden_kern"]
    dil = model_config["hidden_dilation"]
    rf_size = input_kern + ((hidden_kern-1) * dil)*(layers - 1)
    return rf_size


def get_predictions(model, dataloaders, device='cpu', as_dict=False, per_neuron=True, test_data=True, **kwargs):
    predictions = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            if test_data:
                _, output = model_predictions_repeats(dataloader=v, model=model, data_key=k, device=device)
            else:
                _, output = model_predictions(dataloader=v, model=model, data_key=k, device=device)
            predictions[k] = output.T

    if not as_dict:
        predictions = [v for v in predictions.values()]
    return predictions


def get_targets(model, dataloaders, device='cpu', as_dict=True, per_neuron=True, test_data=True, **kwargs):
    responses = {}
    with eval_state(model) if not isinstance(model, types.FunctionType) else contextlib.nullcontext():
        for k, v in dataloaders.items():
            if test_data:
                targets, _ = model_predictions_repeats(dataloader=v, model=model, data_key=k, device=device)
                targets_per_neuron = []
                for i in range(targets[0].shape[1]):
                    neuronal_responses = []
                    for repeats in targets:
                        neuronal_responses.append(repeats[:,i])
                    targets_per_neuron.append(neuronal_responses)
                responses[k] = targets_per_neuron
            else:
                targets, _ = model_predictions(dataloader=v, model=model, data_key=k, device=device)
                responses[k] = targets.T

    if not as_dict:
        responses = [v for v in responses.values()]
    return responses


def get_avg_firing(dataloaders, as_dict=False, per_neuron=True):
    """
    Returns average firing rate across the whole dataset
    """

    avg_firing = {}
    for k, dataloader in dataloaders.items():
        target = torch.empty(0)
        for images, responses in dataloader:
            if len(images.shape) == 5:
                responses = responses.squeeze(dim=0)
            target = torch.cat((target, responses.detach().cpu()), dim=0)
        avg_firing[k] = target.mean(0).numpy()

    if not as_dict:
        avg_firing = np.hstack([v for v in avg_firing.values()]) if per_neuron else np.mean(
            np.hstack([v for v in avg_firing.values()]))
    return avg_firing


def get_fano_factor(dataloaders, as_dict=False, per_neuron=True):
    """
    Returns average firing rate across the whole dataset
    """

    fano_factor = {}
    for k, dataloader in dataloaders.items():
        target = torch.empty(0)
        for images, responses in dataloader:
            if len(images.shape) == 5:
                responses = responses.squeeze(dim=0)
            target = torch.cat((target, responses.detach().cpu()), dim=0)
        fano_factor[k] = (target.var(0) / target.mean(0)).numpy()

    if not as_dict:
        fano_factor = np.hstack([v for v in fano_factor.values()]) if per_neuron else np.mean(
            np.hstack([v for v in fano_factor.values()]))
    return fano_factor