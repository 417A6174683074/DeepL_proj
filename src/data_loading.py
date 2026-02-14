import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle  # TODO implement pickalable training loop
import pandas as pd
import random
import os
import copy

# static typing - vscode has good built in extensions for it
from numpy.typing import NDArray
from typing import Literal, Generator, Callable, TypeAlias


def load_task_i(divided_into_tasks_data_path: str, task: int, tag: Literal["train", "test", "valid"]) -> NDArray[np.float64]:
    X: NDArray[np.float64] = np.load(f"{divided_into_tasks_data_path}/X_{task}_{tag}.npy")
    return X


class GroupTrainingSet:
    """Just making this class to add clarity with the names from the paper
    Corresponds to D_t from the algorithm on page 4.
    """

    def __init__(self, X_g_list: list[NDArray[np.float64]], y_g_list: list[int]) -> None:
        self.X_g_list: list[NDArray[np.float64]] = X_g_list
        self.y_g_list: list[int] = y_g_list

    def __len__(self):
        return len(self.X_g_list)

    def nb_entries(self):
        return sum([X.shape[0] for X in self.X_g_list])

    def get_tensors(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        X_combined: NDArray[np.float64] = np.concatenate(self.X_g_list, axis=0)
        y_combined: NDArray[np.int64] = np.concatenate(
            [np.full(X.shape[0], task, dtype=np.int64) for X, task in zip(self.X_g_list, self.y_g_list)], axis=0
        )
        return X_combined, y_combined


def load_groups_data(
    divided_into_tasks_data_path: str, groups: list[NDArray[np.int64]], tag: Literal["train", "test", "valid"]
) -> Generator[GroupTrainingSet, None, None]:
    """For each group, load and yield the corresponding data from pre-divided task files

    Args:
        divided_into_tasks_data_path (str): path to predivided data
        groups (list[NDArray[np.int64]]): groups
        tag: train test or valid

    Yields:
        Generator[GroupTrainingSet], None, None]: For each group, yield a GroupTrainingSet containing the data from every task of the group
    """

    for g in groups:
        X_g_list: list[NDArray[np.float64]] = [load_task_i(divided_into_tasks_data_path, task, tag) for task in g]
        yield GroupTrainingSet(X_g_list, g.tolist())


def tests_dataloader(divided_into_tasks_data_path, tasks: list[int]) -> DataLoader:
    """Dataloader for the tasks already trained on

    Args:
        divided_into_tasks_data_path (_type_): path to divided data
        tasks (list[int]): list of trained tasks

    Returns:
        DataLoader: DataLoader
    """
    X_list = [load_task_i(divided_into_tasks_data_path, task, "test") for task in tasks]
    print(tasks)
    X_combined: NDArray[np.float64] = np.concatenate(X_list, axis=0)
    y_combined: NDArray[np.int64] = np.concatenate([np.full(X.shape[0], task, dtype=np.int64) for X, task in zip(X_list, tasks)], axis=0)
    dataset = TensorDataset(torch.from_numpy(X_combined).float(), torch.from_numpy(y_combined).long())
    return DataLoader(dataset, batch_size=32, shuffle=False)


def train_dataloader(divided_into_tasks_data_path, tasks: list[int]) -> DataLoader:
    """Dataloader for the tasks already trained on

    Args:
        divided_into_tasks_data_path (_type_): path to divided data
        tasks (list[int]): list of trained tasks

    Returns:
        DataLoader: DataLoader
    """
    X_list = [load_task_i(divided_into_tasks_data_path, task, "train") for task in tasks]
    X_combined: NDArray[np.float64] = np.concatenate(X_list, axis=0)
    y_combined: NDArray[np.int64] = np.concatenate([np.full(X.shape[0], task, dtype=np.int64) for X, task in zip(X_list, tasks)], axis=0)
    dataset = TensorDataset(torch.from_numpy(X_combined).float(), torch.from_numpy(y_combined).long())
    return DataLoader(dataset, batch_size=32, shuffle=False)
