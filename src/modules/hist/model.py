import numpy as np
import scipy.optimize
import scipy.stats
import skimage.filters
from histogram import NtCardHistogram


def gmm(x, w1, mu1, sigma1, w2, mu2, sigma2):
    p1 = scipy.stats.norm.pdf(x, mu1, sigma1)
    p2 = scipy.stats.norm.pdf(x, mu2, sigma2)
    return w1 * p1 + w2 * p2


class Model:

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
        return self.peaks.max()

    @property
    def peaks(self):
        return np.array([self.__hom[1].args[0], self.__het[1].args[0]])

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
        y1 = scores[0, :]
        y2 = scores[1:, :].sum(axis=0)
        return x[np.where(y2 >= y1)[0][0]]

    def fit(self, hist: NtCardHistogram) -> int:
        num_iters = 0
        x = np.arange(1, hist.max_count + 1)
        y = hist.as_distribution()
        err_rv = scipy.stats.burr
        p_err, _, info, *_ = scipy.optimize.curve_fit(
            err_rv.pdf, x, y, [0.5, 0.5, 0.5], full_output=True, maxfev=5000
        )
        self.__err_rv = err_rv(*p_err)
        num_iters += info["nfev"]
        y = np.clip(hist.as_distribution() - self.__err_rv.pdf(x), 0, None)
        d = skimage.filters.threshold_multiotsu(hist=(hist.values, x - 1))[-1]
        p0 = [0.5, d / 2, 1, 0.5, d, 1]
        p, _, info, *_ = scipy.optimize.curve_fit(
            gmm, x, y, p0, full_output=True, maxfev=5000
        )
        self.__het = (p[0], scipy.stats.norm(p[1], p[2]))
        self.__hom = (p[3], scipy.stats.norm(p[4], p[5]))
        num_iters += info["nfev"]
        return num_iters
