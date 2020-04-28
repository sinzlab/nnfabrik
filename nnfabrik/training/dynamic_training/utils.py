import numpy as np
from collections import namedtuple


class BadConfigException(Exception):
    pass


def slice_iter(n, step):
    k = 0
    while k < n - step:
        yield slice(k, k + step)
        k += step
    yield slice(k, None)


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
