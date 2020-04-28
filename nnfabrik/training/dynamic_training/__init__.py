from mlutils.measures import PoissonLoss3d
from .utils import (
    corr,
    ptcorr,
    compute_predictions,
    slice_iter,
    BadConfigException,
    correlation_closure,
    compute_scores,
)
from mlutils.training import cycle_datasets, early_stopping
from tqdm import tqdm
from logging import getLogger
from itertools import product, repeat
import torch
from functools import partial
from collections import OrderedDict


log = getLogger(__name__)


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

    trainloaders = OrderedDict(dataset=dataloaders["train_loader"])
    valloaders = OrderedDict(dataset=dataloaders["val_loader"])
    testloaders = OrderedDict(dataset=dataloaders["test_loader"])

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
                enumerate(cycle_datasets(trainloaders)), desc="Epoch {}".format(epoch)
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
