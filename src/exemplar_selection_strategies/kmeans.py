import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def kmeans_exemplar_selection(X_task: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    if m > X_task.shape[0]:
        return X_task

    kmeans = KMeans(n_clusters=m)
    kmeans.fit(X_task)

    # Get the samples in the data closest to the kmeans centers
    selected_indices = np.array([np.argmin(np.linalg.norm(X_task - kmeans.cluster_centers_[i], axis=1)) for i in range(m)])
    ret = X_task[selected_indices]
    assert len(ret) == m
    return ret
