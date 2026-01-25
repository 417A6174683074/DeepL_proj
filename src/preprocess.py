import os
import numpy as np
from numpy.typing import NDArray
from typing import Literal


def pre_divide_data_by_task(total_classes: int, data_path: str, divided_into_tasks_data_path: str, tag: Literal["train", "test", "valid"]):
    """Step in data processing where we already divide it into tasks to make fetching the data faster and easier.

    Args:
        total_classes (int): number of classes in the data
        data_path (str): folder leading to .npy files
        tag: train test or valid
    """

    if not os.path.exists(divided_into_tasks_data_path):
        os.makedirs(divided_into_tasks_data_path)

    # data path
    X_path: str = f"{data_path}/X_{tag}.npy"
    y_path: str = f"{data_path}/y_{tag}.npy"
    X_data: NDArray[np.float64] = np.load(X_path)
    y_data: NDArray[np.int64] = np.load(y_path)

    print(f"{y_data.shape} rows")

    for i in range(total_classes):
        task_indices = np.where(y_data == i)[0]  # Every index i for which y_data[i] is in the array g

        X_i: NDArray[np.float64] = X_data[task_indices]

        # sanity check
        # y_i: NDArray[np.int64] = y_data[task_indices]
        # assert (y_i == y_i[0]).all()

        np.save(f"{divided_into_tasks_data_path}/X_{i}_{tag}.npy", X_i)


if __name__ == "__main__":
    data_path = "./scratch/Malware/CICAndMal/processed_data/"  # folder containing the npy files
    divided_into_tasks_data_path: str = os.path.join(data_path, "divided_by_tasks")
    total_classes = 42  # number of labels for our NN
    pre_divide_data_by_task(total_classes, data_path, divided_into_tasks_data_path, "train")
    pre_divide_data_by_task(total_classes, data_path, divided_into_tasks_data_path, "test")
