import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
import skimage.filters
from histogram import NtCardHistogram


def gmm(x, w1, a1, m1, s1, w2, a2, m2, s2):
    p1 = scipy.stats.skewnorm.pdf(x, a1, m1, s1)
    p2 = scipy.stats.skewnorm.pdf(x, a2, m2, s2)
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
        w_het, rv_het = self.heterozygous_rv
        w_hom, rv_hom = self.homozygous_rv
        scores = np.zeros(shape=(3, len(x)))
        scores[0, :] = self.__err_rv.pdf(x)
        scores[1, :] = w_het * rv_het.pdf(x)
        scores[2, :] = w_hom * rv_hom.pdf(x)
        return scores

    def get_solid_weak_intersection(self, x):
        scores = self.score_components(x)
        y1 = scores[0, :].reshape(-1)
        y2 = scores[1:, :].sum(axis=0).reshape(-1)
        i = np.where(y2 >= y1)[0]
        return x[i[0]] if i.shape[0] > 0 else 0

    def fit(self, hist: NtCardHistogram) -> int:
        num_iters = 0
        x = np.arange(1, hist.max_count + 1)
        self.__x_fit = x.copy()
        y = hist.as_distribution()
        err_rv = scipy.stats.burr
        p_err, _, info, *_ = scipy.optimize.curve_fit(
            err_rv.pdf, x, y, [0.5, 0.5, 0.5], full_output=True, maxfev=5000
        )
        self.__err_rv = err_rv(*p_err)
        num_iters += info["nfev"]
        d = skimage.filters.threshold_multiotsu(hist=(y, x))[-1]
        y = np.clip(hist.as_distribution() - self.__err_rv.pdf(x), 0, None)
        b = ([0] * 8, [np.inf] * 8)
        b[0][2], b[1][2] = hist.first_minima, 2 * d / 2 - hist.first_minima
        b[0][6], b[1][6] = d / 2, 3 * d / 2
        p, _, info, *_ = scipy.optimize.curve_fit(
            gmm, x, y, bounds=b, method="trf", full_output=True, maxfev=Model.MAX_ITERS
        )
        i = len(p) // 2
        self.__het = (p[0], scipy.stats.skewnorm(*p[1:i]))
        self.__hom = (p[i], scipy.stats.skewnorm(*p[i + 1 :]))
        num_iters += info["nfev"]
        return num_iters
