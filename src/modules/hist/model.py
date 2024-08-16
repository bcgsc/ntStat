import warnings

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
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


def pdf(x, components):
    y = np.zeros(x.shape)
    for w, rv in components:
        y += w * rv.pdf(x)
    return y


def fitness(params, x, y_true, components):
    y_pred = pdf(x, update_components(components, params))
    return np.square(y_pred - y_true).sum()


class Model:

    MAX_ITERS = 5000

    def __init__(self) -> None:
        self.__components = []

    @property
    def converged(self) -> bool:
        return len(self.__components) > 0

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

    def fit(self, hist: NtCardHistogram) -> int:
        self.__hist_max_count = hist.max_count
        components = [
            (1, scipy.stats.burr(0.5, 0.5, 0.5)),
            (1 / 2, scipy.stats.norm(hist.otsu_thresholds[0], 1)),
            (1 / 2, scipy.stats.norm(hist.otsu_thresholds[1], 1)),
        ]
        bounds = [
            (0, 1),
            (0, 2),
            (0, 2),
            (0, 2),
            (0, 1),
            (0, hist.max_count),
            (0, hist.max_count),
            (0, 1),
            (0, hist.max_count),
            (0, hist.max_count),
        ]
        p0 = [p for w, rv in components for p in [w] + list(rv.args)]
        x, y = np.arange(1, hist.max_count + 1), hist.as_distribution()
        opt = scipy.optimize.basinhopping(
            fitness,
            minimizer_kwargs={"args": (x, y, components)},
            x0=p0,
            seed=42,
            disp=True,
        )
        p, num_iters = opt.x, opt.nit
        try:
            p_cf, c, info, *_ = scipy.optimize.curve_fit(
                lambda x, *params: pdf(x, update_components(components, params)),
                x,
                y,
                p0=opt.x,
                full_output=True,
                maxfev=Model.MAX_ITERS,
            )
            if np.isfinite(np.linalg.cond(c)):
                p = p_cf
                num_iters += info["nfev"]
            else:
                raise RuntimeError()
        except RuntimeError:
            warnings.warn("LM failed, using results from differential evolution")
        components = update_components(components, p)
        if components[1][1].mean() > components[2][1].mean():
            components[1], components[2] = components[2], components[1]
        self.__components = components
        return num_iters
