import numpy as np
import numpy.typing
import scipy.signal


class NtCardHistogram:

    def __init__(self, path: str) -> None:
        with open(path) as hist_file:
            data = [int(row.strip().split()[1]) for row in hist_file.readlines()]
        self.__total = int(data[0])
        self.__distinct = int(data[1])
        self.__hist = np.array(data[2:])
        self.__hist.setflags(write=False)
        self.__min0 = scipy.signal.argrelextrema(self.__hist, np.less)[0][0]

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
    def first_minima(self) -> int:
        return self.__min0

    @property
    def mode_after_first_minima(self) -> int:
        return self.values[self.first_minima :].argmax() + self.first_minima

    def as_distribution(self) -> numpy.typing.NDArray[np.float64]:
        return self.values / self.num_distinct
