import warnings
from functools import partial

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from mlutils import measures
from mlutils.measures import *
from mlutils.training import early_stopping, MultipleObjectiveTracker, eval_state, cycle_datasets, Exhauster, LongCycler
from ..utility.nn_helpers import set_random_seed

from ..utility import metrics
from ..utility.metrics import corr_stop, poisson_stop


def early_stop_trainer(model, seed, stop_function='corr_stop',
                       loss_function='PoissonLoss', epoch=0, interval=1, patience=10, max_iter=75,
                       maximize=True, tolerance=0.001, device='cuda', restore_best=True,
                       lr_init=0.005, lr_decay_factor=0.3, min_lr=0.0001, optim_batch_step=True,
                       verbose=True, lr_decay_steps=3, dataloaders=None, **kwargs):
    """"
    Args:
        model: PyTorch nn module
        seed: random seed
        trainer_config:
            lr_schedule: list or ndarray that contains lr and lr decrements after early stopping kicks in
            stop_function: stop condition in early stopping, has to be one string of the following:
                'corr_stop'
                'gamma stop'
                'exp_stop'
                'poisson_stop'
            loss_function: has to be a string that gets evaluated with eval()
                Loss functions that are built in at mlutils that can
                be selected in the trainer config are:
                    'PoissonLoss'
                    'GammaLoss'
            device: Device that the model resides on. Expects arguments such as torch.device('')
                Examples: 'cpu', 'cuda:2' (0-indexed gpu)

        Pytorch Dataloaders are expanded into dictionary of individual loaders
            train: PyTorch DtaLoader -- training data
            val: validation data loader
            test: test data loader -- not used during training

    Returns:
        score: performance score of the model
        output: user specified validation object based on the 'stop function'
        model_state: the full state_dict() of the trained model
    """

    train = dataloaders["train"] if dataloaders else kwargs["train"]
    val = dataloaders["val"] if dataloaders else kwargs["val"]
    test = dataloaders["test"] if dataloaders else kwargs["test"]

    # --- begin of helper function definitions
    def model_predictions(loader, model, data_key):
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

    # all early stopping conditions
    def corr_stop(model, loader=None, avg=True):
        """
        Returns either the average correlation of all neurons or the the correlations per neuron.
            Gets called by early stopping and the model performance evaluation
        """
        loader = val if loader is None else loader
        n_neurons, correlations_sum = 0, 0
        if not avg:
            all_correlations = np.array([])

        for data_key in loader:
            with eval_state(model):
                target, output = model_predictions(loader, model, data_key)

            ret = corr(target, output, axis=0)

            if np.any(np.isnan(ret)):
                warnings.warn('{}% NaNs '.format(np.isnan(ret).mean() * 100))
            ret[np.isnan(ret)] = 0

            if not avg:
                all_correlations = np.append(all_correlations, ret)
            else:
                n_neurons += output.shape[1]
                correlations_sum += ret.sum()

        corr_ret = correlations_sum / n_neurons if avg else all_correlations
        return corr_ret

    def gamma_stop(model):
        with eval_state(model):
            target, output = model_predictions(val, model)

        ret = -stats.gamma.logpdf(target + 1e-7, output + 0.5).mean(axis=1) / np.log(2)
        if np.any(np.isnan(ret)):
            warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
        ret[np.isnan(ret)] = 0
        return ret.mean()

    def exp_stop(model, bias=1e-12, target_bias=1e-7):
        with eval_state(model):
            target, output = model_predictions(val, model)
        target = target + target_bias
        output = output + bias
        ret = (target / output + np.log(output)).mean(axis=1) / np.log(2)
        if np.any(np.isnan(ret)):
            warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
        ret[np.isnan(ret)] = 0
        # -- average if requested
        return ret.mean()

    def poisson_stop(model, loader=None, avg=False):
        poisson_losses = np.array([])
        loader = val if loader is None else loader
        n_neurons = 0
        for data_key in loader:
            with eval_state(model):
                target, output = model_predictions(loader, model, data_key)

            ret = output - target * np.log(output + 1e-12)
            if np.any(np.isnan(ret)):
                warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))

            poisson_losses = np.append(poisson_losses, np.nansum(ret, 0))
            n_neurons += output.shape[1]
        return poisson_losses.sum()/n_neurons if avg else poisson_losses.sum()

    def readout_regularizer_stop(model):
        ret = 0
        with eval_state(model):
            for data_key in val:
                ret += model.readout.regularizer(data_key).detach().cpu().numpy()
        return ret

    def core_regularizer_stop(model):
        with eval_state(model):
            if model.core.regularizer():
                return model.core.regularizer().detach().cpu().numpy()
            else:
                return 0

    def full_objective(model, data_key, inputs, targets, **kwargs):
        """
        Computes the training loss for the model and prespecified criterion.
            Default: PoissonLoss, summed over Neurons and Batches, scaled by dataset
                        size and batch size to account for batch noise.

        Args:
            inputs: i.e. images
            targets: neuronal responses that the model should predict

        Returns: training loss summed over all neurons. Summed over batches and Neurons

        """
        m = len(train[data_key].dataset)
        k = inputs.shape[0]
            
        return np.sqrt(m / k) * criterion(model(inputs.to(device), data_key=data_key, **kwargs), targets.to(device)).sum() \
               + model.regularizer(data_key)
        

    def run(model, full_objective, optimizer, scheduler, stop_closure, train_loader,
            epoch, interval, patience, max_iter, maximize, tolerance,
            restore_best, tracker, optim_step_count, lr_decay_steps):

        for epoch, val_obj in early_stopping(model, stop_closure,
                                             interval=interval, patience=patience,
                                             start=epoch, max_iter=max_iter, maximize=maximize,
                                             tolerance=tolerance, restore_best=restore_best,
                                             tracker=tracker, scheduler=scheduler, lr_decay_steps=lr_decay_steps):
            optimizer.zero_grad()

            # reports the entry of the current epoch for all tracked objectives
            if verbose:
                for key in tracker.log.keys():
                    print(key, tracker.log[key][-1])

            # Beginning of main training loop
            for batch_no, (data_key, data) in tqdm(enumerate(LongCycler(train_loader)),
                                                      desc='Epoch {}'.format(epoch)):

                loss = full_objective(model, data_key, *data)
                if (batch_no+1) % optim_step_count == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                loss.backward()

        # End of training
        return model, epoch

    # model setup
    set_random_seed(seed)
    model.to(device)
    model.train()

    # current criterium is supposed to be poisson loss. Only for that loss, the additional arguments are defined
    criterion = eval(loss_function)(per_neuron=True, avg=False)

    # get stopping criterion from helper functions based on keyword
    stop_closure = eval(stop_function)

    tracker = MultipleObjectiveTracker(correlation=partial(corr_stop, model),
                                       poisson_loss=partial(poisson_stop, model),
                                       poisson_loss_val=partial(poisson_stop, model, val),
                                       readout_l1=partial(readout_regularizer_stop, model),
                                       core_regularizer=partial(core_regularizer_stop, model))

    trainable_params = [p for p in list(model.parameters()) if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max' if maximize else 'min',
                                                           factor=lr_decay_factor,
                                                           patience=patience,
                                                           threshold=tolerance,
                                                           min_lr=min_lr,
                                                           verbose=verbose,
                                                           threshold_mode='abs',
                                                           )

    optim_step_count = len(train.keys()) if optim_batch_step else 1

    model, epoch = run(model=model,
                       full_objective=full_objective,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       stop_closure=stop_closure,
                       train_loader=train,
                       epoch=epoch,
                       interval=interval,
                       patience=patience,
                       max_iter=max_iter,
                       lr_decay_steps=lr_decay_steps,
                       maximize=maximize,
                       tolerance=tolerance,
                       restore_best=restore_best,
                       tracker=tracker,
                       optim_step_count=optim_step_count)

    model.eval()
    tracker.finalize()

    # compute average test correlations as the score
    avg_corr = corr_stop(model, test, avg=True)

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()}
    return avg_corr, output, model.state_dict()


def standard_early_stop_trainer(model, dataloaders, seed, avg_loss=True, scale_loss=True,   # trainer args
                                loss_function='PoissonLoss', stop_function='corr_stop',
                                loss_accum_batch_n=None, device='cuda', verbose=True,
                                interval=1, patience=5, epoch=0, lr_init=0.005,             # early stopping args
                                max_iter=100, maximize=True, tolerance=1e-6,
                                restore_best=True, lr_decay_steps=3,
                                lr_decay_factor=0.3, min_lr=0.0001,                         # lr scheduler args
                                cb=None, **kwargs):

    def full_objective(model, data_key, inputs, targets):
        if scale_loss:
            m = len(trainloaders[data_key].dataset)
            k = inputs.shape[0]
            loss_scale = np.sqrt(m / k)
        else: 
            loss_scale = 1.0
        
        return loss_scale * criterion(model(inputs.to(device), data_key), targets.to(device)) + model.regularizer(data_key)

    trainloaders = dataloaders["train"]
    valloaders = dataloaders.get("validation", dataloaders["val"] if "val" in dataloaders.keys() else None)
    testloaders = dataloaders["test"]
    
    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()
    
    criterion = getattr(measures, loss_function)(avg=avg_loss)
    stop_closure = partial(getattr(metrics, stop_function), model, valloaders, device=device)

    n_iterations = len(LongCycler(trainloaders))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if maximize else 'min', 
                                                           factor=lr_decay_factor, patience=patience, threshold=tolerance, 
                                                           min_lr=min_lr, verbose=verbose, threshold_mode='abs')
    
    # set the number of iterations over which you would like to accummulate gradients
    optim_step_count = len(trainloaders.keys()) if loss_accum_batch_n is None else loss_accum_batch_n
    
    # define some trackers
    tracker_dict = dict(correlation=partial(corr_stop, model, valloaders, device=device),
                        poisson_loss=partial(poisson_stop, model, valloaders, device=device), 
                        poisson_loss_val=partial(poisson_stop, model, valloaders, device=device))
    
    if hasattr(model, 'tracked_values'):
        tracker_dict.update(model.tracked_values)
    
    tracker = MultipleObjectiveTracker(**tracker_dict)
    
    # train over epochs
    for epoch, val_obj in early_stopping(model, stop_closure, interval=interval, patience=patience, 
                                         start=epoch, max_iter=max_iter, maximize=maximize, 
                                         tolerance=tolerance, restore_best=restore_best, tracker=tracker, 
                                         scheduler=scheduler, lr_decay_steps=lr_decay_steps):

        # print the quantities from tracker
        if verbose and tracker is not None:
            print("=======================================")
            for key in tracker.log.keys():
                print(key, tracker.log[key][-1], flush=True)

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad()
        for batch_no, (data_key, data) in tqdm(enumerate(LongCycler(trainloaders)), total=n_iterations, desc="Epoch {}".format(epoch)):               

            loss = full_objective(model, data_key, *data)
            loss.backward()
            if (batch_no+1) % optim_step_count == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        
    ##### Model evaluation ####################################################################################################
    model.eval()
    tracker.finalize()
    
    # Compute avg validation and test correlation
    avg_val_corr = corr_stop(model, valloaders, avg=True, device=device)
    avg_test_corr = corr_stop(model, testloaders, avg=True, device=device)

    # return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()}
    
    return avg_test_corr, output, model.state_dict()
