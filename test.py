import random
import h5py
import numpy as np
import sympy
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import math
import torchvision
import pandas as pd
import glob
import os
import sys
from scipy.spatial.distance import squareform
import scipy as sp
import networkx
import torch.distributed as dist
import time
import timeit
from datetime import date
import struct
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.MNISTModel import MNISTModel
from model.CIFAR10Model import CIFAR10Model
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import RandomState, SeedSequence
from numpy.random import MT19937
from model.model import *
from model.CIFAR10Model import *
from sklearn.cluster import KMeans
import itertools
import json
from sympy import Symbol
from sympy.solvers.inequalities import reduce_rational_inequalities
from trans_matrix import *


def moving_average(input_data, window_size):
    moving_average = [[] for i in range(len(input_data))]
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if j < window_size - 1:
                if type(input_data[i][j + 1]) == str:
                    input_data[i][j + 1] = float(input_data[i][j + 1])
                if input_data[i][j + 1] == 'nan' or 'inf':
                    input_data[i][j + 1] = float(input_data[i][j])
                moving_average[i].append(sum(input_data[i][:j + 1]) / len(input_data[i][:j + 1]))
            else:
                input_data[i][j - window_size + 1:j + 1][-1] = float(input_data[i][j - window_size + 1:j + 1][-1])
                # print(input_data[i][j - window_size + 1:j + 1])
                moving_average[i].append(sum(input_data[i][j - window_size + 1:j + 1]) / len(input_data[i][j - window_size + 1:j + 1]))
    moving_average_means = []
    for i in range(len(moving_average[0])):
        sum_data = []
        for j in range(len(moving_average)):
            sum_data.append(moving_average[j][i])
        moving_average_means.append(sum(sum_data) / len(sum_data))
    return np.array(moving_average), moving_average_means

def matrix(nodes, num_neighbor):
    upper = int(nodes / 2) - 2
    bottom = 1
    matrix = np.ones((nodes,), dtype=int)
    while True:
        org_matrix = np.diag(matrix)
        org_target = np.arange(nodes, dtype=int)
        for i in range(nodes):
            if np.count_nonzero(org_matrix[i]) < num_neighbor + 1:
                if np.count_nonzero(org_matrix[i]) < num_neighbor + 1 and np.count_nonzero(
                        org_matrix.transpose()[i]) < num_neighbor + 1:
                    target = np.setdiff1d(org_target, i)
                    target_set = []
                    for k in range(len(target)):
                        if np.count_nonzero(org_matrix[target[k]]) < num_neighbor + 1:
                            target_set.append(target[k])
                    if num_neighbor + 1 - int(np.count_nonzero(org_matrix[i])) <= len(target_set):
                        target = np.random.choice(target_set, num_neighbor + 1 - int(np.count_nonzero(org_matrix[i])),
                                                  replace=False)
                    for j in range(len(target)):
                        org_matrix[i][target[j]] = 1
                        org_matrix.transpose()[i][target[j]] = 1
            else:
                pass
        if np.count_nonzero(
                np.array([np.count_nonzero(org_matrix[i]) for i in range(nodes)]) - (num_neighbor + 1)) == 0:
            break
    return org_matrix

def Ring_network(nodes):
    matrix = np.ones((nodes,), dtype=int)
    conn_matrix = np.diag(matrix)
    neighbors = []
    for i in range(nodes):
        connected = [(i - 1) % nodes, i, (i + 1) % nodes]
        for j in connected:
            conn_matrix[i][j] = 1
            conn_matrix.transpose()[i][j] = 1
        neighbors.append(connected)
    factor = 1 / len(neighbors[0])
    conn_matrix = conn_matrix * factor
    return conn_matrix,

def Check_Matrix(client, matrix):
    count = 0
    for i in range(client):
        if np.count_nonzero(matrix[i] - matrix.transpose()[i]) == 0:
            pass
        else:
            count += 1
    if count != 0:
        raise Exception('The Transfer Matrix Should be Symmetric')
    else:
        print('Transfer Matrix is Symmetric Matrix')


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'arial'

num_nodes = 10
num_neighbors = 5
num_classes = 10
seed = 24

if num_neighbors == 2:
    network = 'Ring'
else:
    network = 'Random'

conn_matrix = Transform(num_nodes=num_nodes, num_neighbors=num_neighbors, seed=seed, network=network)
# print('W: ', conn_matrix.matrix)
Check_Matrix(client=num_nodes, matrix=conn_matrix.matrix)

eigenvalues = np.linalg.eigvals(conn_matrix.factor * conn_matrix.matrix)
# print(eigenvalues)
eigenvalues = np.sort(eigenvalues, axis=-1)[::-1]

rho = 1 - eigenvalues[1]
if rho == 1:
    rho = 0
mu = np.sort(1-eigenvalues, axis=-1)[::-1][0]

print(rho, mu)
gammas = np.arange(0.02, 1, 0.02)
UPPER = []
GAMMA = []

for gamma in gammas:
    a = rho**2*(1-gamma**2)
    b = a / (a+mu**2)
    if b > 0:
        GAMMA.append(gamma)
        UPPER.append(b)

print(GAMMA, UPPER)
plt.plot(GAMMA, UPPER)
# plt.ylim([0, 1])
plt.show()


# neighbors = list(conn_matrix.neighbors)
# print('Neighbors: ', neighbors)
#
# neighbors_weights = [[0 for i in range(num_neighbors+1)] for i in range(num_nodes)]
# print('Initial neighbor weights: ', neighbors_weights)
# for n in range(num_nodes):
#     node_weights = n
#     for m in range(num_nodes):
#         if n in neighbors[m]:
#             neighbors_weights[m][neighbors[m].index(n)] = node_weights
# print('Updated neighbor weights: ', neighbors_weights)

# alpha = 0.01
# Alpha = [alpha for i in range(num_classes)]
# samples = np.random.dirichlet(Alpha, size=num_nodes)
# print(len(samples), samples)
# summation = np.sum(samples, axis=1)
# print(summation)
# for i in range(len(samples)):
#     data_samples = np.array(6000 * samples[i], dtype=np.int16)
#     print(data_samples, sum(data_samples))

# As = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
# LOWER = []
# UPPER = []
# DIFFER = []
#
# for A in As:
#     lower = np.sqrt(1/A + 4/A**2) - 2/A
#     upper = np.sqrt(2/A + 4/A**2) - 2/A
#     LOWER.append(lower)
#     UPPER.append(upper)
#     DIFFER.append(upper - lower)
#
# plt.plot(As, LOWER, color='red')
# plt.plot(As, UPPER, color='blue')
# plt.plot(As, DIFFER, color='green')
# print(LOWER[0], UPPER[0])
# print(min(LOWER), max(UPPER), min(DIFFER), max(DIFFER))
#
# plt.show()

# import numpy as np
# import random
# from typing import List, Sequence, Tuple, Optional, Union
#
# def partition_dirichlet_equal_return_data(
#     dataset: Sequence,                 # e.g., list/array of samples or (x, y) tuples
#     labels: Union[Sequence[int], np.ndarray],  # length N vector of class ids in [0, num_classes-1]
#     num_clients: int = 20,
#     num_classes: int = 10,
#     alpha: float = 0.1,
#     seed: int = 42,
#     drop_remainder: bool = True,
#     return_indices: bool = False,      # set True if you want indices instead of samples
# ) -> Tuple[List[List], Optional[List[List[int]]]]:
#     """
#     Partition `dataset` into `num_clients` shards using Dirichlet(alpha) over `num_classes`,
#     ensuring:
#       - equal number of samples per client
#       - no overlap across clients
#       - heterogeneity controlled by `alpha`
#
#     Returns:
#       client_data:  list of length num_clients, each a list of *samples* (or indices if return_indices=True)
#       client_indices (optional): if return_indices=False => also returns the indices; else returns None
#     """
#     rng = np.random.default_rng(seed)
#     random.seed(seed)
#
#     labels = np.asarray(labels)
#     N = len(labels)
#     assert len(dataset) == N, "dataset and labels must have the same length."
#
#     # Build class -> indices pools (no overlap by construction)
#     class_pools = {c: [] for c in range(num_classes)}
#     for idx, y in enumerate(labels):
#         y = int(y)
#         if y < 0 or y >= num_classes:
#             raise ValueError(f"Label {y} outside [0, {num_classes-1}] at index {idx}.")
#         class_pools[y].append(idx)
#
#     # Shuffle each class pool
#     for c in range(num_classes):
#         rng.shuffle(class_pools[c])
#
#     # Compute equal per-client size; optionally drop the remainder
#     total_samples = sum(len(class_pools[c]) for c in range(num_classes))
#     remainder = total_samples % num_clients
#     if remainder != 0:
#         if not drop_remainder:
#             raise ValueError(
#                 f"Total N={total_samples} not divisible by num_clients={num_clients}. "
#                 f"Set drop_remainder=True or adjust data."
#             )
#         # Drop 'remainder' items from the global pool (proportionally by class for fairness)
#         # Compute how many to drop per class roughly proportional to class size
#         to_drop = remainder
#         class_sizes = np.array([len(class_pools[c]) for c in range(num_classes)], dtype=float)
#         if class_sizes.sum() == 0:
#             raise RuntimeError("Empty dataset after filtering.")
#         # initial proportional plan
#         drop_plan = np.floor(to_drop * (class_sizes / class_sizes.sum())).astype(int)
#         # adjust to match exact to_drop
#         while drop_plan.sum() < to_drop:
#             # give +1 to the largest remaining class
#             c = int(np.argmax(class_sizes - drop_plan))
#             drop_plan[c] += 1
#         # drop from the end of each class pool
#         for c in range(num_classes):
#             k = int(drop_plan[c])
#             if k > 0 & k <= len(class_pools[c]):
#                 class_pools[c] = class_pools[c][:-k]
#         total_samples = sum(len(class_pools[c]) for c in range(num_classes))
#
#     samples_per_client = total_samples // num_clients
#     assert samples_per_client * num_clients == total_samples, "Internal size mismatch after trimming."
#
#     # Helper: draw class counts for a client subject to availability caps
#     def draw_counts_with_caps(quota: int, avail: np.ndarray, probs: np.ndarray) -> np.ndarray:
#         counts = np.zeros_like(avail, dtype=int)
#         remaining = quota
#         avail_left = avail.copy().astype(int)
#
#         # build initial weights (zero where unavailable)
#         w = probs * (avail_left > 0)
#         if w.sum() == 0:
#             w = (avail_left > 0).astype(float)
#         w = w / w.sum()
#
#         while remaining > 0:
#             proposal = rng.multinomial(remaining, w)
#             proposal = np.minimum(proposal, avail_left)
#             taken = proposal.sum()
#             if taken == 0:
#                 # fallback purely by availability
#                 w = (avail_left > 0).astype(float)
#                 if w.sum() == 0:
#                     break
#                 w = w / w.sum()
#                 continue
#             counts += proposal
#             avail_left -= proposal
#             remaining -= taken
#             if remaining == 0:
#                 break
#             w = probs * (avail_left > 0)
#             if w.sum() == 0:
#                 w = (avail_left > 0).astype(float)
#             w = w / w.sum()
#
#         if counts.sum() != quota:
#             # top off if tiny mismatch (very rare)
#             need = quota - counts.sum()
#             if need > 0:
#                 where = np.where(avail_left > 0)[0]
#                 if where.size < need:
#                     raise RuntimeError("Insufficient availability to reach exact quota.")
#                 # greedily take 1 from the first 'need' classes with availability
#                 for j in where[:need]:
#                     counts[j] += 1
#                     avail_left[j] -= 1
#         return counts
#
#     # Prepare outputs
#     client_indices: List[List[int]] = [[] for _ in range(num_clients)]
#
#     # Iteratively assign to each client
#     for client in range(num_clients):
#         # For each client, sample Dirichlet over classes
#         probs = rng.dirichlet(alpha * np.ones(num_classes))
#         # Current availability
#         avail = np.array([len(class_pools[c]) for c in range(num_classes)], dtype=int)
#         if avail.sum() < samples_per_client:
#             raise RuntimeError(f"Insufficient availability left to fill client {client}.")
#
#         counts = draw_counts_with_caps(samples_per_client, avail, probs)
#
#         # Pop indices from each class pool
#         for c in range(num_classes):
#             k = int(counts[c])
#             if k > 0:
#                 # take last k (pools already shuffled)
#                 picked = class_pools[c][-k:]
#                 del class_pools[c][-k:]
#                 client_indices[client].extend(picked)
#
#         rng.shuffle(client_indices[client])
#         assert len(client_indices[client]) == samples_per_client, "Client quota mismatch."
#
#     # Build actual per-client data lists (samples)
#     if return_indices:
#         client_data = client_indices
#         return client_data, None
#     else:
#         client_data = [[dataset[i] for i in idxs] for idxs in client_indices]
#         return client_data, client_indices
#
#
# # ---------------- Example usage ----------------
# if __name__ == "__main__":
#     # Toy dataset: (x, y) pairs, 10 classes
#     N = 50000
#     num_classes = 10
#     rng = np.random.default_rng(0)
#     y = rng.integers(0, num_classes, size=N)
#     X = [("sample_%d" % i, int(y[i])) for i in range(N)]  # pretend each sample is a tuple
#
#     client_data, client_indices = partition_dirichlet_equal_return_data(
#         dataset=X,
#         labels=y,
#         num_clients=20,
#         num_classes=num_classes,
#         alpha=0.02,
#         seed=123,
#         drop_remainder=True,
#         return_indices=False,   # set True if you only want indices
#     )
#
#     # sanity checks
#     print("Per-client sizes:", [len(cd) for cd in client_data])
#
#     # show class histogram for client 0
#     from collections import Counter
#     for i in range(20):
#         print(f"Client {i} class hist:", Counter([s[1] for s in client_data[i]]))
#


