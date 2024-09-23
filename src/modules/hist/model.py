import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats


def update_components(params):
    num_components = (len(params) - 4) // 3
    components = [(params[0], scipy.stats.burr(*params[1:4]))]
    for i in range(num_components):
        i_w = 4 + i * 3
        w, d, p = params[i_w : i_w + 3]
        r = d * p / (1 - p + np.finfo(float).eps) + 1
        components.append((w, scipy.stats.nbinom(r, p)))
    return components


def score(x, components):
    scores = np.empty(shape=(len(components), x.shape[0]))
    for i, (w, rv) in enumerate(components):
        y = rv.pdf(x) if isinstance(rv.dist, scipy.stats.rv_continuous) else rv.pmf(x)
        scores[i, :] = w * y
    return scores


def loss(params, x, y):
    fx = score(x, update_components(params)).sum(axis=0)
    return np.abs(y - fx).sum()


def log_iteration(
    intermediate_result: scipy.optimize.OptimizeResult,
    history: list,
):
    components = update_components(intermediate_result.x)
    history.append((components, intermediate_result.fun))


def nbinom_mode(rv):
    r, p = rv.args
    return (r - 1) * (1 - p) / p


class Model:

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
    def copy_rvs(self):
        return self.__components[3:]

    @property
    def coverage(self):
        return self.peaks[1]

    @property
    def peaks(self):
        return np.array([rv.mean() for _, rv in self.__components[1:]])

    def pdf(self, x):
        return self.score_components(x).sum(axis=0)

    def score_components(self, x):
        return score(x, self.__components)

    def get_weak_robust_crossover(self, x):
        scores = self.score_components(x)
        y1 = scores[0, :].reshape(-1)
        y2 = scores[1:, :].sum(axis=0).reshape(-1)
        i = np.where(y2 >= y1)[0]
        return x[i[0]] if i.shape[0] > 0 else 0

    def fit(self, hist, num_components=2, config=dict()):
        bounds = [
            (0, 1),
            (0, 2),
            (0, 2),
            (0, 2),
        ]
        d = hist.as_distribution()[hist.first_minima :].argmax() + hist.first_minima + 1
        for _ in range(num_components):
            bounds.extend([(0, 1), (hist.first_minima, d * 3), (0, 1)])
        x, y = np.arange(1, hist.max_count + 1), hist.as_distribution()
        history = []
        max_iters = config.get("maxiter", 1000)
        callback = lambda intermediate_result: log_iteration(
            intermediate_result,
            history,
        )
        opt = scipy.optimize.differential_evolution(
            func=loss,
            popsize=config.get("popsize", 3),
            init=config.get("init", "sobol"),
            x0=[(a + b) / 2 for a, b in bounds],
            bounds=bounds,
            args=(x, y),
            seed=config.get("seed", 42),
            updating="deferred",
            workers=config.get("workers", -1),
            maxiter=max_iters,
            callback=callback,
            mutation=config.get("mutation", (0.3, 1.2)),
            recombination=config.get("recombination", 0.8),
            strategy=config.get("strategy", "best1bin"),
            polish=True,
        )
        components = update_components(opt.x)
        sorted_nbinoms = sorted(components[1:], key=lambda c: c[1].mean())
        self.__components = [components[0]] + list(sorted_nbinoms)
        return opt.nit, loss(opt.x, x, y), history
