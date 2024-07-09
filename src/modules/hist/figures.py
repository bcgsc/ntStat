import os

import numpy as np
import numpy.typing
import termplotlib as tpl


def print_hist(hist: numpy.typing.NDArray[np.uint64]) -> None:
    w = 3 * os.get_terminal_size().columns // 4
    x = np.arange(1, hist.shape[0] + 1)
    y = np.add.reduceat(hist, range(0, hist.shape[0], hist.shape[0] // w + 1))
    y = np.log(y + 1)
    y = y - y.min()
    fig = tpl.figure()
    fig.hist(y, x, max_width=w)
    fig.show()
    print()
