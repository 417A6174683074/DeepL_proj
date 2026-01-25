import numpy as np
from numpy.typing import NDArray


def random_exemplar_selection(X_task: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    """Select m random exemplars from X_task"""
    if m > X_task.shape[0]:
        return X_task
    n_samples: int = X_task.shape[0]
    indices: NDArray[np.int64] = np.random.choice(n_samples, size=m, replace=False)
    ret = X_task[indices]
    assert len(ret) == m
    return ret
