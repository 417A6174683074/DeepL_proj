import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_files(x_path, y_path, mixing, order, nb_groups, nb_cl, nb_cl_first):
    X_data = np.load(x_path)
    y_data = np.load(y_path)
    
    labels_old = np.array([mixing[label] for label in y_data])
    
    files_train = [[] for _ in range(nb_groups + 1)]
    
    # first task
    for i2 in range(nb_cl_first):
        tmp_ind = np.where(labels_old == order[i2])[0]
        np.random.shuffle(tmp_ind)
        
        train_ind = tmp_ind[:]
        files_train[0].extend(train_ind)
    
    # rest tasks
    for i in range(nb_groups):
        for i2 in range(nb_cl):
            current_cl = nb_cl_first + i * nb_cl + i2
            tmp_ind = np.where(labels_old == order[current_cl])[0]
            np.random.shuffle(tmp_ind)

            train_ind = tmp_ind[:]
            files_train[i+1].extend(train_ind)
    
    return files_train

def read_data(x_path, y_path, mixing, indices):
    X = np.load(x_path)
    y = np.load(y_path)
    
    X_data = X[indices]
    y_data = y[indices]
    
    # mixing
    y_data = np.array([mixing[label] for label in y_data])
    
    return X_data, y_data

import numpy as np
from collections import defaultdict

def split_classes(files_train, y_path, mixing, task_id, frac_keep=0.9, seed=0):

    rng = np.random.default_rng(seed)
    y = np.load(y_path)
    task_indices = files_train[task_id]

    # grouper les indices par classe
    class_to_indices = defaultdict(list)
    for idx in task_indices:
        cls = mixing[y[idx]]
        class_to_indices[cls].append(idx)

    new_task_indices = []
    held_out = {}

    for cls, indices in class_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)

        n_total = len(indices)
        n_keep = int(np.floor(frac_keep * n_total))

        keep = indices[:n_keep].tolist()
        leftover = indices[n_keep:].tolist()

        new_task_indices.extend(keep)
        if len(leftover) > 0:
            held_out[cls] = leftover

    # new_files_train = files_train[task_id].copy()
    # new_files_train[task_id] = new_task_indices

    return new_task_indices, held_out



def read_data3(X_data, y_data, mixing, indices):
    X_data_list = X_data[indices]
    y_data_list = y_data[indices]
    
    y_data_list = np.array([mixing[label] for label in y_data_list])
    
    # Ensure y_data has the right shape
    if len(y_data_list.shape) > 1:
        y_data_list = y_data_list.reshape(-1)
    
    return X_data_list, y_data_list
