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
        p0 = [p for w, rv in components for p in [w] + list(rv.args)]
        p, _, info, *_ = scipy.optimize.curve_fit(
            lambda x, *p: pdf(x, update_components(components, p)),
            np.arange(1, hist.max_count + 1),
            hist.as_distribution(),
            p0=p0,
            full_output=True,
            maxfev=Model.MAX_ITERS,
        )
        self.__components = update_components(components, p)
        return info["nfev"]
