import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle  # TODO implement pickalable training loop
import pandas as pd
import random
import os
import copy
import time
from datetime import timedelta

# static typing - vscode has good built in extensions for it
from numpy.typing import NDArray
from typing import Literal, Generator, Callable, TypeAlias

from data_loading import GroupTrainingSet, tests_dataloader, load_groups_data

# models
from models.ember_transformer import EmberTransformer
from models.simple_mpl import SimpleMLP
from models.simplest_mlp import SimplestMLP

# Exemplar selection strategy
from exemplar_selection_strategies.random_selection import random_exemplar_selection
from exemplar_selection_strategies.kmeans import kmeans_exemplar_selection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
ExamplarSelectionStrategy: TypeAlias = Callable[[NDArray[np.float64], int], NDArray[np.float64]]


def divide_classes_into_groups(total_classes: int, nb_cl_first_group: int, nb_groups: int, nb_cl_per_group: int) -> list[NDArray[np.int64]]:
    """return the classes divided into random groups of predermined size

    Args:
        total_classes (int): number of classes
        nb_cl_first_group (int): number of classes in first group
        nb_groups (int): nb of groups including the first one
        nb_cl_per_group (int): number of classes per group except the first one

    Returns:
        list[NDArray[np.int64]]: _description_
    """
    assert total_classes == nb_cl_first_group + (
        (nb_groups - 1) * nb_cl_per_group
    ), "verify we have the correct number of classes in the group division"

    classes_order: NDArray[np.int64] = np.arange(total_classes)
    np.random.shuffle(classes_order)

    groups: list[NDArray[np.int64]] = []
    groups.append(classes_order[:nb_cl_first_group])
    for i in range(nb_groups - 1):
        group_order_index_start: int = nb_cl_first_group + i * nb_cl_per_group
        groups.append(classes_order[group_order_index_start : group_order_index_start + nb_cl_per_group])
    return groups


class Buffer:
    def __init__(
        self,
        buffer_truncate_class: ExamplarSelectionStrategy,
        new_task_select_m_exemplars: ExamplarSelectionStrategy,
        K: int,
    ) -> None:
        # TODO initialize non-empty buffer with pickle
        """Initialize empty buffer

        Args:
            buffer_truncate_class (ExamplarSelectionStrategy): Function to update each element of the buffer
            buffer_update (ExamplarSelectionStrategy): Function to decide what m elements to add in the buffer for each class, strategies in 4.1
            K: buffer capacity in 4.2
        """
        self.X_tasks_list: list[NDArray[np.float64]] = []
        self.y_tasks_list: list[int] = []
        # TODO initialize non-empty buffer with pickle

        self.K: int = K
        self.buffer_truncate_class: ExamplarSelectionStrategy = buffer_truncate_class
        self.new_task_select_m_exemplars: ExamplarSelectionStrategy = new_task_select_m_exemplars

        self.last_added_tasks: list[int] = []  # Keep track of the last group of tasks added to the buffer for refinement

    def get_tensors(self):
        if not self.X_tasks_list:
            return None, None

        X_combined: NDArray[np.float64] = np.concatenate(self.X_tasks_list, axis=0)
        y_combined: NDArray[np.int64] = np.concatenate(
            [np.full(X.shape[0], task, dtype=np.int64) for X, task in zip(self.X_tasks_list, self.y_tasks_list)], axis=0
        )

        return X_combined, y_combined

    def update_buffer(self, D_t: GroupTrainingSet):
        nb_classes_in_buffer: int = len(self.X_tasks_list)
        nb_classes_in_new_dataset = len(D_t)

        nb_classes = nb_classes_in_buffer + nb_classes_in_new_dataset
        m: int = self.K // nb_classes

        self.X_tasks_list = [self.buffer_truncate_class(class_in_buffer, m) for class_in_buffer in self.X_tasks_list]
        new_X_tasks = [self.new_task_select_m_exemplars(new_task_in_buffer, m) for new_task_in_buffer in D_t.X_g_list]
        self.X_tasks_list.extend(new_X_tasks)

        self.y_tasks_list.extend(D_t.y_g_list)
        self.last_added_tasks = D_t.y_g_list

    def dataloader(self, batch_size=256, shuffle=True) -> DataLoader:
        X, y = self.get_tensors()
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataloader_from_data_and_buffer(D_t: GroupTrainingSet, E: Buffer, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
    """Returns a dataloader with the buffer included as specified in phase 1 initial training of the paper

    Args:
        D_t (GroupTrainingSet): Data for the training of the group
        E (Buffer): Data from the buffer
        batch_size (int, optional): batch size. Defaults to 256.
        shuffle (bool, optional): shuffle the data. Defaults to True.

    Returns:
        DataLoader: dataloader of the tensors.
    """
    X_buffer, y_buffer = E.get_tensors()
    X_D_t, y_D_t = D_t.get_tensors()
    if X_buffer is None or y_buffer is None:  # Edge case when buffer still empty
        X = X_D_t
        y = y_D_t
    else:
        X: NDArray[np.float64] = np.concatenate([X_D_t, X_buffer], axis=0)
        y: NDArray[np.int64] = np.concatenate([y_D_t, y_buffer], axis=0)

    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Section 4.*
from torch._tensor import Tensor
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer


class TraMEL:
    def __init__(self, model: Module, E: Buffer, optimizer: torch.optim.Optimizer, groups: list[NDArray[np.int64]]) -> None:
        """TraMEL algorithm keeps track of model and a buffer E that updates over time"""
        self.model: Module = model.to(device)  # We will try different models, relevant to section 4.4
        self.f_i_m1: Module = copy.deepcopy(self.model).to(device)  # refinement section 4.3
        self.f_prime: Module = copy.deepcopy(self.model).to(device)  # refinement section 4.3
        self.E: Buffer = E
        self.optimizer: Optimizer = optimizer
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.group = 0
        self.epoch = 0
        self.phase: Literal[1, 2, 3, 4] = 1
        self.groups: list[NDArray[np.int64]] = groups

        self.runtime: timedelta = timedelta(0)
        self.last_dump_time: float = time.time()

    def range(self, epochs: int):
        """Custom range method, just to keep the current epoch in pickle and keep the *phases* methods clean with no pickle management"""
        for i in range(self.epoch, epochs):
            self.epoch = i
            self.dump()
            yield i
        self.epoch = 0

    def phase1_train(
        self, D_t: GroupTrainingSet, *, epochs: int = 50, writer: SummaryWriter
    ):  # LIST AS PLACEHOLDER, TODO find correct data type D_t represents the t-th dataset
        """Train model on D_t and E"""
        self.model.train()
        dataloader: DataLoader = get_dataloader_from_data_and_buffer(D_t, self.E)
        print(f"len dataloader {len(dataloader)}")
        for epoch in self.range(epochs):
            epoch_losses = []

            for X, y in dataloader:
                X: Tensor = X.to(device)
                y: Tensor = y.to(device)

                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.loss_fn(logits, y)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses[-100:])
            writer.add_scalar("Phase1_Train/loss", avg_loss, self.group * epochs + epoch)
            print(epoch, "epoch", "batch", f"Average loss: {avg_loss:.6f}")

    def phase2_examplar_selection(self, D_t: GroupTrainingSet):
        self.E.update_buffer(D_t)

    def phase3_refinement(self, t: int, *, alpha: int = 1, beta: int = 1, epochs=20, writer: SummaryWriter):
        """Refine the model with the buffer which contains examples from the current dataset and the previous ones.

        Args:
            t (int): timestep/index of current group
            alpha (int, optional): Section 4.3. Defaults to 1.
            beta (int, optional): Section 4.3. Defaults to 1.
        """
        if t == 0:
            self.f_i_m1.load_state_dict(self.model.state_dict())
            return

        self.f_prime.load_state_dict(self.model.state_dict())
        dataloader: DataLoader = self.E.dataloader()

        mse = nn.MSELoss(reduction="mean")
        ce = nn.CrossEntropyLoss(reduction="mean")

        last_added_tensor: Tensor = torch.tensor(self.E.last_added_tasks, dtype=torch.long).to(device)

        self.model.train()
        self.f_i_m1.eval()
        self.f_prime.eval()

        for epoch in self.range(epochs):
            epoch_losses = []

            for X, y in dataloader:
                X: Tensor = X.to(device)
                y: Tensor = y.to(device)

                E_i_mask: Tensor = torch.isin(y, last_added_tensor)  # get a boolean tensor telling us which tasks were just added to the buffer
                E_bi_mask: Tensor = (
                    ~E_i_mask
                )  # get a boolean tensor telling us which tasks were in the buffer before, (on which self.f_i_m1 is trained)

                fix: Tensor = self.model(X)  # f^{i}(x)
                with torch.no_grad():
                    fim1x: Tensor = self.f_i_m1(X)  # f^{i-1}(x)
                    fip: Tensor = self.f_prime(X)  # f^{i}'(x)

                l_ce: Tensor = ce(fix, y)
                # if there are entries corresponding to the old tasks in this batch
                l_past: Tensor = mse(fix[E_bi_mask], fim1x[E_bi_mask]) if E_bi_mask.any() else torch.tensor(0.0).to(device)
                l_current: Tensor = mse(fix[E_i_mask], fip[E_i_mask]) if E_i_mask.any() else torch.tensor(0.0).to(device)

                loss: Tensor = l_ce + alpha * l_past + beta * l_current

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses[-100:])
            writer.add_scalar("Phase3_Refinement/loss", loss.item(), self.group * epochs + epoch)  # type: ignore
            print(epoch, "epoch", "batch", f"Average loss: {avg_loss:.6f}")

        self.f_i_m1.load_state_dict(self.model.state_dict())

    def phase4_test(self, divided_into_tasks_data_path: str, *, writer: SummaryWriter) -> float:
        """Test the model on all seen tasks and compute overall accuracy

        Args:
            divided_into_tasks_data_path: Path to divided task data

        Returns:
            float: Overall accuracy on all seen tasks
        """
        self.model.eval()
        seen_tasks = self.E.y_tasks_list
        test_dataloader = tests_dataloader(divided_into_tasks_data_path, seen_tasks)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)

                logits = self.model(X)
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        accuracy = np.mean(all_preds == all_labels)
        writer.add_scalar("Phase4_Test/accuracy", accuracy, self.group)
        print(f"Test Accuracy: {accuracy:.4f}")

        return accuracy

    ################ Pickle ####################
    def dump(self, name: str = "tramel.pickle") -> None:
        now: float = time.time()
        self.runtime += timedelta(seconds=now - self.last_dump_time)
        self.last_dump_time = now
        print(f"dumped, runtime: {self.runtime}")
        with open(name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, name: str = "tramel.pickle") -> "TraMEL":
        with open(name, "rb") as f:
            tramel = pickle.load(f)
            tramel.last_dump_time = time.time()
            return tramel

    def dump_first_group(self):
        """Since first group specifically takes a long time to train
        and does not test what the paper is about (catastrophic forgetting). We dont train it too much"""
        self.dump("first_group.pickle")

    @classmethod
    def load_first_group(cls) -> "TraMEL":
        return cls.load("first_group.pickle")

    def __str__(self):
        return f"""group: {self.group}
epoch: {self.epoch}
phase: {self.phase}
model: {self.model} 
runtime: {self.runtime}
groups: {"\n        ".join([str(g) for g in self.groups])}
"""


def main():

    # Parameters of the dataset for the dataset
    data_path = "./scratch/Malware/CICAndMal/processed_data/"  # folder containing the npy files
    divided_into_tasks_data_path: str = os.path.join(data_path, "divided_by_tasks")
    nb_cl_first_group = 22  # number of classes in the first training, mentioned in 5.2. Should be tested
    nb_groups = 5  # nb of groups of classes
    nb_cl_per_group = 5  # nb of classes per group (except first group)
    total_classes = 42  # number of labels for our NN
    in_features = 85  # X_data has 85 features for CICAndMal

    try:
        tramel: TraMEL = TraMEL.load()
    except FileNotFoundError:
        try:
            tramel: TraMEL = TraMEL.load_first_group()
            print("Pickle not found, starting from first group")
        except FileNotFoundError:
            print("No pickle found starting from fresh instance")

            # Hyperparameters of the training and model
            model = EmberTransformer()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
            K = 30000
            E = Buffer(random_exemplar_selection, kmeans_exemplar_selection, K)
            groups: list[NDArray[np.int64]] = divide_classes_into_groups(total_classes, nb_cl_first_group, nb_groups, nb_cl_per_group)
            tramel = TraMEL(model, E, optimizer, groups)

    print(tramel)

    writer = SummaryWriter("./runs/tramel")
    try:
        for t, D_t in enumerate(load_groups_data(divided_into_tasks_data_path, tramel.groups, "train")):
            if t < tramel.group:
                # skip the first few groups if they are already trained in the tramel
                continue
            else:
                tramel.group = t

            print(f"nb of entries in group: {D_t.nb_entries()}")
            if tramel.phase == 1:
                tramel.phase1_train(D_t, epochs=50, writer=writer)
                tramel.phase = 2
                tramel.dump()

            if tramel.phase == 2:
                tramel.phase2_examplar_selection(D_t)
                tramel.phase = 3
                tramel.dump()

            if tramel.phase == 3:
                tramel.phase3_refinement(t, epochs=20, writer=writer)
                tramel.phase = 4
                tramel.dump()

            if tramel.phase == 4:
                tramel.phase4_test(divided_into_tasks_data_path, writer=writer)
                tramel.phase = 1
                tramel.dump()

            if t == 0:
                tramel.dump_first_group()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
