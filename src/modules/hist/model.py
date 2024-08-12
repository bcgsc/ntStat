import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
from histogram import NtCardHistogram


def gmm(x, w1, m1, s1, w2, m2, s2):
    p1 = scipy.stats.norm.pdf(x, m1, s1)
    p2 = scipy.stats.norm.pdf(x, m2, s2)
    return w1 * p1 + w2 * p2


class Model:

    MAX_ITERS = 5000

    @property
    def err_rv(self):
        return self.__err_rv

    @property
    def heterozygous_rv(self):
        return self.__het

    @property
    def homozygous_rv(self):
        return self.__hom

    @property
    def coverage(self):
        return self.peaks[1]

    @property
    def peaks(self):
        x = []
        for rv in [self.heterozygous_rv[1], self.homozygous_rv[1]]:
            r = np.linspace(self.__x_fit[0], self.__x_fit[-1], 1_000_000)
            x.append(r[rv.pdf(r).argmax()])
        return np.array(x)

    def pdf(self, x):
        return self.score_components(x).sum(axis=0)

    def score_components(self, x):
        w_err, rv_err = self.err_rv
        w_het, rv_het = self.heterozygous_rv
        w_hom, rv_hom = self.homozygous_rv
        scores = np.zeros(shape=(3, len(x)))
        scores[0, :] = w_err * rv_err.pdf(x)
        scores[1, :] = w_het * rv_het.pdf(x)
        scores[2, :] = w_hom * rv_hom.pdf(x)
        return scores

    def get_weak_robust_crossover(self, x):
        scores = self.score_components(x)
        y1 = scores[0, :].reshape(-1)
        y2 = scores[1:, :].sum(axis=0).reshape(-1)
        i = np.where(y2 >= y1)[0]
        return x[i[0]] if i.shape[0] > 0 else 0

    def fit(self, hist: NtCardHistogram) -> int:
        num_iters = 0
        self.__x_fit = np.arange(1, hist.max_count + 1)
        y = hist.as_distribution()
        err_rv = scipy.stats.burr
        p_err, _, info, *_ = scipy.optimize.curve_fit(
            err_rv.pdf,
            self.__x_fit,
            y,
            [0.5, 0.5, 0.5],
            full_output=True,
            maxfev=5000,
        )
        self.__err_rv = (1.0, err_rv(*p_err))
        num_iters += info["nfev"]
        y = np.clip(hist.as_distribution() - self.__err_rv[1].pdf(self.__x_fit), 0, 1)
        d = y[hist.first_minima :].argmax() + hist.first_minima + 1
        b = ([0] * 6, [1, np.inf, np.inf, 1, np.inf, np.inf])
        b[0][1], b[1][1] = d / 2 * 0.9, d / 2 * 1.1
        b[0][4], b[1][4] = d * 0.9, d * 1.1
        p, _, info, *_ = scipy.optimize.curve_fit(
            gmm,
            self.__x_fit,
            y,
            full_output=True,
            bounds=b,
            maxfev=Model.MAX_ITERS,
            loss="arctan",
        )
        c1 = (p[0], scipy.stats.norm(p[1], p[2]))
        c2 = (p[3], scipy.stats.norm(p[4], p[5]))
        self.__het, self.__hom = (c1, c2) if p[1] < p[4] else (c2, c1)
        num_iters += info["nfev"]
        return num_iters
