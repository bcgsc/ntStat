import sys

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
import tqdm
from histogram import NtCardHistogram


def update_components(components, params):
    updated = []
    i_param = 0
    for _, rv in components:
        n_args = len(rv.args)
        updated_rv = rv.dist(*params[i_param + 1 : i_param + n_args + 1])
        updated.append([params[i_param], updated_rv])
        i_param += n_args + 1
    return updated


def pdf(x, components, params):
    i_param = 0
    y = np.zeros(x.shape)
    for _, rv in components:
        n_args = len(rv.args)
        w = params[i_param]
        p_rv = params[i_param + 1 : i_param + n_args + 1]
        y += w * rv.dist.pdf(x, *p_rv)
        i_param += n_args + 1
    return y


def loss(params, x, y, components):
    fx = pdf(x, components, params)
    huber = scipy.special.huber(0.001, np.abs(y - fx).sum())
    kl_div = scipy.stats.entropy(fx, y)
    return huber + kl_div


def log_iteration(
    intermediate_result: scipy.optimize.OptimizeResult,
    history: list,
    progress_bar: tqdm.tqdm,
):
    history.append((intermediate_result.x, intermediate_result.fun))
    progress_bar.update()
    progress_bar.set_postfix(loss=intermediate_result.fun)


class Model:

    @staticmethod
    def from_params(params):
        model = Model()
        model.__components = update_components(model.__components, params)
        model.__converged = True
        return model

    def __init__(self) -> None:
        self.__components = [
            (1, scipy.stats.burr(0.5, 0.5, 0.5)),
            (1, scipy.stats.skewnorm(1, 1, 1)),
            (1, scipy.stats.skewnorm(1, 1, 1)),
        ]
        self.__converged = False

    @property
    def converged(self) -> bool:
        return self.__converged

    @property
    def err_rv(self):
        return self.__components[0]

    @property
    def heterozygous_rv(self):
        return self.__components[1]

    @property
    def homozygous_rv(self):
        return self.__components[2]

    @property
    def coverage(self):
        return self.peaks[1]

    @property
    def peaks(self):
        x = np.linspace(1, self.__hist_max_count + 1, self.__hist_max_count * 10)
        return x[self.score_components(x)[1:, :].argmax(axis=1)]

    def pdf(self, x):
        return self.score_components(x).sum(axis=0)

    def score_components(self, x):
        scores = np.zeros(shape=(len(self.__components), len(x)))
        for i, (w, rv) in enumerate(self.__components):
            scores[i, :] = w * rv.pdf(x)
        return scores

    def get_weak_robust_crossover(self, x):
        scores = self.score_components(x)
        y1 = scores[0, :].reshape(-1)
        y2 = scores[1:, :].sum(axis=0).reshape(-1)
        i = np.where(y2 >= y1)[0]
        return x[i[0]] if i.shape[0] > 0 else 0

    def fit(self, hist: NtCardHistogram, config: dict = dict()) -> int:
        self.__converged = False
        self.__hist_max_count = hist.max_count
        d = hist.as_distribution()[hist.first_minima :].argmax() + hist.first_minima + 1
        bounds = [
            (0, 1),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 1),
            (0, 2),
            (hist.first_minima, 3 * d / 4),
            (d / 6, d * 6),
            (0, 1),
            (0, 2),
            (3 * d / 4, 3 * d / 2),
            (d / 6, d * 6),
        ]
        p0 = [(ub + lb) / 2 for lb, ub in bounds]
        x, y = np.arange(1, hist.max_count + 1), hist.as_distribution()
        history = []
        progress_bar = tqdm.tqdm(
            desc="Fitting model",
            total=Model.MAX_ITERS,
            disable=None,
            file=sys.stdout,
            leave=False,
        )
        callback = lambda intermediate_result: log_iteration(
            intermediate_result,
            history,
            progress_bar=progress_bar,
        )
        opt = scipy.optimize.differential_evolution(
            func=loss,
            popsize=config.get("popsize", 3),
            init=config.get("init", "sobol"),
            bounds=bounds,
            args=(x, y, self.__components),
            x0=p0,
            seed=config.get("seed", 42),
            updating="deferred",
            workers=config.get("workers", -1),
            maxiter=config.get("maxiter", 1000),
            callback=callback,
            mutation=config.get("mutation", (0.2, 1.0)),
            recombination=config.get("recombination", 0.8),
            strategy=config.get("strategy", "best1exp"),
        )
        progress_bar.close()
        components = update_components(self.__components, opt.x)
        if components[1][1].mean() > components[2][1].mean():
            components[1], components[2] = components[2], components[1]
        self.__components = components
        self.__converged = True
        return opt.nit, history
