import numpy as np
import numpy.typing
import pandas as pd
import utils


class NtCardHistogram:

    def __init__(self, path: str) -> None:
        data = pd.read_csv(path, delimiter=r"\s+", header=None)[1].values
        self.__total = int(data[0])
        self.__distinct = int(data[1])
        self.__hist = data[2:]
        self.__hist.setflags(write=False)
        self.__elbow = utils.find_elbow(self.__hist)
        self.__min0 = utils.find_first_minima(self.__hist)
        self.__otsu = utils.find_otsu_thresholds(self.__hist)
        self.__otsu.setflags(write=False)

    def __getitem__(self, count):
        return self.__hist[count - 1]

    @property
    def values(self) -> numpy.typing.NDArray[np.uint64]:
        return self.__hist

    @property
    def num_total(self) -> int:
        return self.__total

    @property
    def num_distinct(self) -> int:
        return self.__distinct

    @property
    def max_count(self) -> int:
        return self.__hist.shape[0]

    @property
    def elbow(self) -> int:
        return self.__elbow

    @property
    def first_minima(self) -> int:
        return self.__min0

    @property
    def mode_after_first_minima(self) -> int:
        return self.values[self.first_minima :].argmax() + self.first_minima

    @property
    def otsu_thresholds(self) -> numpy.typing.NDArray[np.uint]:
        return self.__otsu

    def as_distribution(self) -> numpy.typing.NDArray[np.float64]:
        return self.values / self.num_distinct
