from functools import partial

# TrainedModels table calls the trainer as follows:
# loss, output, model_state = trainer(model, seed, **trainer_config, **dataloader)

def early_stop_trainer(model, seed, lr_schedule,stop_function ='corr_stop',
                     loss_function ='PoissonLoss', epoch=0, interval=1, patience=10, max_iter=50,
                     maximize=True, tolerance=1e-5, cuda=True, restore_best=True, tracker=None,
                     train_loader, val_loader, test_loader):
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
            loss_function: selects the loss function from mlutils.measures.py. Has to be a string
                            of the following:
                'PoissonLoss'
                'GammaLoss'
        train_loader: PyTorch DtaLoader -- training data
        val_loader: validation data loader
        test_loader: test data loader

    Returns:
        loss: training loss after each epoch
            Expected format: ndarray or dict
        output: user specified output of the training procedure.
            Expected format: ndarray or dict
    """

    # --- begin of helper function definitions

    def model_predictions(loader, model):
        """
        computes model predictions for a given dataloader and a model
        Returns:
            target: ground truth, i.e. neuronal firing rates of the neurons
            output: responses as predicted by the network
        """
        target, output = [], []
        for images, responses, val_responses in loader:
            output.append(model(images).detach().cpu().numpy() * val_responses.detach().cpu().numpy())
            target.append(responses.detach().cpu().numpy() * val_responses.detach().cpu().numpy())
        target, output = map(np.vstack, (target, output))
        return target, output

    # all early stopping conditions
    def corr_stop(model):
        with eval_state(model):
            target, output = model_predictions(val_loader, model)

        ret = corr(target, output, axis=0)

        if np.any(np.isnan(ret)):
            warnings.warn('{}% NaNs '.format(np.isnan(ret).mean() * 100))
        ret[np.isnan(ret)] = 0

        return ret.mean()

    def gamma_stop(model):
        with eval_state(model):
            target, output = model_predictions(val_loader, model)

        ret = -stats.gamma.logpdf(target + 1e-7, output + 0.5).mean(axis=1) / np.log(2)
        if np.any(np.isnan(ret)):
            warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
        ret[np.isnan(ret)] = 0
        return ret.mean()

    def exp_stop(model, bias=1e-12, target_bias=1e-7):
        with eval_state(model):
            target, output = model_predictions(val_loader, model)
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
            target, output = model_predictions(val_loader, model)

        ret = (output - target * np.log(output + 1e-12))
        if np.any(np.isnan(ret)):
            warnings.warn(' {}% NaNs '.format(np.isnan(ret).mean() * 100))
        ret[np.isnan(ret)] = 0
        # -- average if requested
        return ret.mean()

    def full_objective(inputs, targets, val_responses):
        """
        Args:
            inputs: i.e. images
            targets: neuronal responses that the model should predict
            val_responses: valid responses. Target and output are only valid for images,
                that the particular neuron in its session has actually seen

        Returns: loss of the model based on the user specified criterion
        """
        return criterion(model(inputs) * val_responses, targets) + model.core.regularizer() + model.readout.regularizer()

    def run(model, full_objective, optimizer, stop_closure, train_loader,
            epoch, interval, patience, max_iter, maximize, tolerance,
            restore_best, tracker):

        optimizer.zero_grad()
        for epoch, val_obj in early_stopping(model=model, stop_closure=stop_closure,
                                             interval=interval, patience=patience,
                                             start=epoch, max_iter=max_iter, maximize=maximize,
                                             tolerance=tolerance, restore_best=restore_best,
                                             tracker=tracker):
            for data in tqdm(train_loader, desc='Epoch {}'.format(epoch)):
                loss = full_objective(*data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(loss)

            # store the training loss/output after each epoch loop
            loss_ret.append(loss)
            val_output.append(val_obj)


        return model, epoch

    # --- end of helper function definitions

    # set up the model and the loss/early_stopping functions

    model.train(True)
    # get loss function from mlutils.measures based on the keyword
    criterion = eval(loss_function)()
    # get stopping criterion from helper functions based on keyword
    stop_closure = partial(eval(stop_function), model)
    tracker = MultipleObjectiveTracker(poisson=partial(poisson_stop, model),
                                       gamma=partial(gamma_stop, model),
                                       correlation=partial(corr_stop, model),
                                       exponential=partial(exp_stop, model),
                                       )
    val_output = []
    loss_ret = []

    # Run the Training
    for opt, lr in zip(repeat(torch.optim.Adam), lr_schedule):
        print('Training with learning rate {}'.format(lr))
        optimizer = opt(model.parameters(), lr=lr)

        model, epoch = run(model=model,
                           full_objective=full_objective,
                           optimizer=optimizer,
                           stop_closure=stop_closure,
                           train_loader=train_loader,
                           epoch=epoch,
                           interval=interval,
                           patience=patience,
                           max_iter=max_iter,
                           maximize=maximize,
                           tolerance=tolerance,
                           restore_best=restore_best,
                           tracker=tracker,
                           )
    model.eval()
    tracker.finalize()

    return loss_ret, val_output, model.state_dict()