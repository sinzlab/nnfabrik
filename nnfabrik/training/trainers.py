import torch
from functools import partial
from mlutils.measures import *
from mlutils.training import early_stopping, MultipleObjectiveTracker, eval_state, cycle_datasets
from scipy import stats
from tqdm import tqdm
import warnings
from ..utility.nn_helpers import set_random_seed
import numpy as np

def early_stop_trainer(model, seed, stop_function='corr_stop',
                       loss_function='PoissonLoss', epoch=0, interval=1, patience=10, max_iter=75,
                       maximize=True, tolerance=1e-5, device='cuda', restore_best=True,
                       lr_init=0.005, lr_decay_factor=0.3, lr_decay_patience=5, lr_decay_threshold=0.001,
                       min_lr=0.0001, optim_batch_step=True, pretrained_core=False, verbose=True,
                       train=None, val=None, test=None):
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

            poisson_losses = np.append(poisson_losses, np.nanmean(ret, 0))
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
        Computes the training loss for the model and prespecified criterion
        Args:
            inputs: i.e. images
            targets: neuronal responses that the model should predict

        Returns: training loss summed over all neurons
        """
        return criterion(model(inputs.to(device), data_key=data_key, **kwargs), targets.to(device)).sum() \
               + model.regularizer(data_key)


    def run(model, full_objective, optimizer, scheduler, stop_closure, train_loader,
            epoch, interval, patience, max_iter, maximize, tolerance,
            restore_best, tracker, optim_step_count):

        for epoch, val_obj in early_stopping(model, stop_closure,
                                             interval=interval, patience=patience,
                                             start=epoch, max_iter=max_iter, maximize=maximize,
                                             tolerance=tolerance, restore_best=restore_best,
                                             tracker=tracker):
            optimizer.zero_grad()
            scheduler.step(val_obj)

            # reports the entry of the current epoch for all tracked objectives
            if verbose:
                for key in tracker.log.keys():
                    print(key, tracker.log[key][-1])

            # Beginning of main training loop
            for batch_no, (data_key, data) in tqdm(enumerate(cycle_datasets(train_loader)),
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
    criterion = eval(loss_function)(per_neuron=True)

    # get stopping criterion from helper functions based on keyword
    stop_closure = eval(stop_function)

    tracker = MultipleObjectiveTracker(correlation=partial(corr_stop, model),
                                       poisson_loss=partial(poisson_stop, model),
                                       readout_l1=partial(readout_regularizer_stop, model),
                                       core_regularizer=partial(core_regularizer_stop, model))

    trainable_params = [p for p in list(model.parameters()) if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr_init)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=lr_decay_factor,
                                                           patience=lr_decay_patience,
                                                           threshold=lr_decay_threshold,
                                                           min_lr=min_lr,
                                                           verbose=verbose,
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
                       maximize=maximize,
                       tolerance=tolerance,
                       restore_best=restore_best,
                       tracker=tracker,
                       optim_step_count=optim_step_count)

    model.eval()
    tracker.finalize()

    # compute average test correlations as the score
    avg_corr = corr_stop(model, test, avg=True)

    #return the whole tracker output as a dict
    output = {k: v for k, v in tracker.log.items()}
    return avg_corr, output, model.state_dict()

