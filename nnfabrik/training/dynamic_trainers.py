from collections import namedtuple

from neuralpredictors.measures import PoissonLoss3d


def slice_iter(n, step):
    k = 0
    while k < n - step:
        yield slice(k, k + step)
        k += step
    yield slice(k, None)


class BadConfigException(Exception):
    pass


def corr(y1, y2, axis=-1, eps=1e-8, **kwargs):
    """
    Compute the correlation between two matrices along certain dimensions.

    Args:
        y1:      first matrix
        y2:      second matrix
        axis:    dimension along which the correlation is computed.
        eps:     offset to the standard deviation to make sure the correlation is well defined (default 1e-8)
        **kwargs passed to final mean of standardized y1 * y2

    Returns: correlation vector

    """
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (y1.std(axis=axis, keepdims=True, ddof=1) + eps)
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (y2.std(axis=axis, keepdims=True, ddof=1) + eps)
    return (y1 * y2).mean(axis=axis, **kwargs)


def ptcorr(y1, y2, axis=-1, eps=1e-8, **kwargs):
    """
    Compute the correlation between two matrices along certain dimensions.

    Args:
        y1:      first matrix
        y2:      second matrix
        axis:    dimension along which the correlation is computed.
        eps:     offset to the standard deviation to make sure the correlation is well defined (default 1e-8)
        **kwargs passed to final mean of standardized y1 * y2

    Returns: correlation vector

    """
    y1 = (y1 - y1.mean(dim=axis, keepdim=True)) / (y1.std(dim=axis, keepdim=True) + eps)
    y2 = (y2 - y2.mean(dim=axis, keepdim=True)) / (y2.std(dim=axis, keepdim=True) + eps)
    return (y1 * y2).mean(dim=axis, **kwargs)


def compute_predictions(loader, model, readout_key, reshape=True, stack=True, subsamp_size=None, return_lag=False):
    y, y_hat = [], []
    if subsamp_size is not None:
        loader = tqdm(loader)
    for x_val, beh_val, eye_val, y_val in loader:
        neurons = y_val.size(-1)
        if subsamp_size is None:
            y_mod = model(x_val, readout_key, eye_pos=eye_val, behavior=beh_val).detach().cpu().numpy()
        else:
            y_mod = []
            for subs_idx in slice_iter(neurons, subsamp_size):
                y_mod.append(
                    model(x_val, readout_key, eye_pos=eye_val, behavior=beh_val, subs_idx=subs_idx)
                    .detach()
                    .cpu()
                    .numpy()
                )
            y_mod = np.concatenate(y_mod, axis=-1)
        lag = y_val.shape[1] - y_mod.shape[1]
        if reshape:
            y.append(y_val[:, lag:, :].cpu().numpy().reshape((-1, neurons)))
            y_hat.append(y_mod.reshape((-1, neurons)))
        else:
            y.append(y_val[:, lag:, :].cpu().numpy())
            y_hat.append(y_mod)
    if stack:
        y, y_hat = np.vstack(y), np.vstack(y_hat)
    if not return_lag:
        return y, y_hat
    else:
        return y, y_hat, lag


def correlation_closure(mod, loaders, avg=True, subsamp_size=None):
    ret = []
    train = mod.training
    mod.eval()
    for readout_key, loader in loaders.items():
        y, y_hat = compute_predictions(loader, mod, readout_key, reshape=True, stack=True, subsamp_size=subsamp_size)
        co = corr(y, y_hat, axis=0)
        print(readout_key + "correlation: {:.4f}".format(co.mean()))
        ret.append(co)
    ret = np.hstack(ret)
    mod.train(train)

    if avg:
        return ret.mean()
    else:
        return ret


PerformanceScores = namedtuple("PerformanceScores", ["pearson"])


def compute_scores(y, y_hat, axis=0):
    pearson = corr(y, y_hat, axis=axis)
    return PerformanceScores(pearson=pearson)


def dynamic_model_trainer(
    model,
    dataloaders,
    schedule=(0.005, 0.001),
    max_epoch=500,
    n_subsample_test=500,
    val_interval=4,
    patience=5,
    stop_tolerance=1e-6,
):

    trainloaders = dataloaders["train_loader"]
    valloaders = dataloaders["val_loader"]
    testloaders = dataloaders["test_loader"]

    criterion = PoissonLoss3d()
    if len(trainloaders) > 1:
        raise BadConfigException("TrainConfig.SingleScan only accepts single scan datasets")

    def objective(model, readout_key, inputs, beh, eye_pos, targets):
        outputs = model(inputs, readout_key, eye_pos=eye_pos, behavior=beh)
        return (
            criterion(outputs, targets)
            + (model.core.regularizer() if not model.readout[readout_key].stop_grad else 0)
            + model.readout.regularizer(readout_key).cuda(0)
            + (model.shifter.regularizer(readout_key) if model.shift else 0)
            + (model.modulator.regularizer(readout_key) if model.modulate else 0)
        )

    def run(model, objective, optimizer, stop_closure, trainloaders, epoch=0):
        log.info("Training models with {} and state {}".format(optimizer.__class__.__name__, repr(model.state)))
        optimizer.zero_grad()
        iteration = 0

        for epoch, val_obj in early_stopping(
            model,
            stop_closure,
            interval=val_interval,
            patience=patience,
            start=epoch,
            max_iter=max_epoch,
            maximize=True,
            tolerance=stop_tolerance,
            restore_best=True,
        ):
            for batch_no, (readout_key, data) in tqdm(
                enumerate(cycle_datasets(trainloaders)), desc=self.__class__.__name__ + "  | Epoch {}".format(epoch)
            ):
                obj = objective(model, readout_key, *data)
                obj.backward()
                optimizer.step()
                optimizer.zero_grad()
                iteration += 1
        return model, epoch

    # --- train
    log.info("Shipping model to GPU")
    model = model.cuda()
    model.train(True)
    print(model)
    epoch = 0

    model.shift = True
    for opt, lr in zip(repeat(torch.optim.Adam), schedule):
        log.info("Training with learning rate {}".format(lr))

        optimizer = opt(model.parameters(), lr=lr)

        model, epoch = run(
            model, objective, optimizer, partial(correlation_closure, loaders=valloaders), trainloaders, epoch=epoch
        )
    model.eval()

    def compute_test_score_tuples(key, testloaders, model, **kwargs):
        scores, unit_scores = [], []
        for readout_key, testloader in testloaders.items():
            log.info("Computing test scores for " + readout_key)

            y, y_hat = compute_predictions(testloader, model, readout_key, **kwargs)
            perf_scores = compute_scores(y, y_hat)

            member_key = (MovieMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)
            member_key.update(key)

            unit_ids = testloader.dataset.neurons.unit_ids
            member_key["neurons"] = len(unit_ids)
            member_key["pearson"] = perf_scores.pearson.mean()

            scores.append(member_key)
            unit_scores.extend([dict(member_key, unit_id=u, pearson=c) for u, c in zip(unit_ids, perf_scores.pearson)])
        return scores, unit_scores

    stop_closure = partial(correlation_closure, loaders=valloaders)
    updated_key = dict(
        key,
        val_corr=np.nanmean(stop_closure(model, avg=False)),
        model={k: v.cpu().numpy() for k, v in model.state_dict().items()},
    )

    scores, unit_scores = compute_test_score_tuples(
        key, testloaders, model, reshape=True, stack=True, subsamp_size=n_subsample_test
    )
