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
                       loss_function='PoissonLoss', epoch=0, interval=1, patience=10, max_iter=50,
                       maximize=True, tolerance=1e-5, device='cuda', restore_best=True,
                       lr_init=0.003, lr_decay_factor=0.5, lr_decay_patience=5, lr_decay_threshold=0.001,
                       min_lr=0.0001, optim_batch_step=True, train=None, val=None, test=None):
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
        train: PyTorch DtaLoader -- training data
        val: validation data loader
        test: test data loader -- not used during training

    Returns:
        loss: training loss after each epoch
            Expected format: ndarray or dict
        output: user specified output of the training procedure.
            Expected format: ndarray or dict
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
            output = torch.cat((output, model(images, data_key)), dim=0)
            target = torch.cat((target, responses), dim=0)

        return target.detach().cpu().numpy(), output.detach().cpu().numpy()

    # all early stopping conditions
    def corr_stop(model, loader=None, avg=True):

        loader = val if loader is None else loader
        correlations = np.zeros((len(loader.keys()), 1))
        n_neurons = np.zeros((1, len(loader.keys())))
        if not avg:
            all_correlations = np.array([])

        for i, data_key, in enumerate(loader):
            with eval_state(model):
                target, output = model_predictions(loader, model, data_key)

            ret = corr(target, output, axis=0)

            if np.any(np.isnan(ret)):
                warnings.warn('{}% NaNs '.format(np.isnan(ret).mean() * 100))
            ret[np.isnan(ret)] = 0

            if not avg:
                all_correlations = np.append(all_correlations, ret)
            else:
                n_neurons[0,i] = output.shape[1]
                correlations[i,0] = ret.mean()

        corr_ret = ((n_neurons@correlations) / n_neurons.sum()).item() if avg else all_correlations
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

    def poisson_stop(model):
        with eval_state(model):
            target, output = model_predictions(val, model)

        ret = (output - target * np.log(output + 1e-12))
        if np.any(np.isnan(ret)):
            warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
        ret[np.isnan(ret)] = 0
        # -- average if requested
        return ret.mean()

    def full_objective(model, data_key, inputs, targets, **kwargs):
        """
        Computes the training loss for the model and prespecified criterion
        Args:
            inputs: i.e. images
            targets: neuronal responses that the model should predict

        Returns: training loss of the model
        """
        return criterion(model(inputs.to(device), data_key=data_key, **kwargs), targets.to(device)) \
                + model.regularizer(data_key)

    def run(model, full_objective, optimizer, scheduler, stop_closure, train,
            epoch, interval, patience, max_iter, maximize, tolerance,
            restore_best, tracker, optim_step_count):

        for epoch, val_obj in early_stopping(model, stop_closure,
                                             interval=interval, patience=patience,
                                             start=epoch, max_iter=max_iter, maximize=maximize,
                                             tolerance=tolerance, restore_best=restore_best,
                                             tracker=tracker):
            optimizer.zero_grad()
            scheduler.step(val_obj)

            for batch_no, (data_key, data) in tqdm(enumerate(cycle_datasets(train)),
                                                      desc='Epoch {}'.format(epoch)):

                loss = full_objective(model, data_key, *data)
                if (batch_no+1) % optim_step_count == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                loss.backward()
            print('Training loss: {}'.format(loss))

        return model, epoch

    # --- end of helper function definitions

    # set up the model and the loss/early_stopping functions
    set_random_seed(seed)
    model.to(device)
    model.train()
    criterion = eval(loss_function)()
    # get stopping criterion from helper functions based on keyword
    stop_closure = eval(stop_function)

    # full tracker init
    # tracker = MultipleObjectiveTracker(poisson=partial(poisson_stop, model),
    #                                    gamma=partial(gamma_stop, model),
    #                                    correlation=partial(corr_stop, model),
    #                                    exponential=partial(exp_stop, model))

    # minimal tracker init
    tracker = MultipleObjectiveTracker(correlation=partial(corr_stop, model))

    # reduce on plateau feature from pytorch 1.2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           factor=lr_decay_factor,
                                                           patience=lr_decay_patience,
                                                           threshold=lr_decay_threshold,
                                                           min_lr=min_lr,
                                                           )
    optim_step_count = len(train.keys()) if optim_batch_step else 1

    model, epoch = run(model=model,
                       full_objective=full_objective,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       stop_closure=stop_closure,
                       train=train,
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

    val_output = tracker.log["correlation"]

    # compute average test correlations as the score
    avg_corr = corr_stop(model, test, avg=False)
    return avg_corr, val_output, model.state_dict()

