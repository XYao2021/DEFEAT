import numpy as np
import random
import copy
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from typing import List, Sequence, Tuple, Optional, Union

def sampler(dataset, num_nodes):
    pass

def MINIST_sample(num_classes, num_clients):  # For number of clients greater or equal to number of classes
    org_set = np.arange(num_classes, dtype=int)
    samples = np.array([], dtype=int)
    while num_clients > 0:
        np.random.shuffle(org_set)
        if num_clients <= num_classes:
            sample = org_set[:num_clients]
        else:
            sample = org_set
        samples = np.append(samples, sample, axis=0)
        num_clients -= num_classes
    return samples

def split_data(sample, train_data):
    data = [[] for _ in range(len(sample))]
    for i in range(len(train_data.targets)):
        for j in range(len(sample)):
            if train_data.targets[i] == sample[j]:
                data[j].append(train_data[i])
    return data

class Sampling:
    def __init__(self, num_class, num_client, train_data, method, seed, name):
        super().__init__()
        self.num_class = num_class
        self.num_client = num_client
        self.train_data = train_data
        self.method = method
        self.seed = seed
        self.name = name

        self.partition = None
        self.set_length = int(len(self.train_data)/self.num_class)
        self.sample_data = [[] for _ in range(num_class)]
        self.classes = np.arange(self.num_class)
        self.target = None
        self.dataset = []

        self._initialize()

    def _initialize(self):
        if self.method == 'uniform':
            self.partition = np.ones(self.num_client) / self.num_client
        elif self.method == 'random':  # TODO: Need to consider the relationship with BATCH_SIZE
            self.partition = np.random.random(self.num_client)
            self.partition /= np.sum(self.partition)
        if self.name == 'SVHN':
            for i in range(self.num_class):
                tmp_data = []
                for j in range(len(self.train_data)):
                    if self.train_data.labels[j] == self.classes[i]:
                        tmp_data.append(self.train_data[j])
                self.dataset.append(tmp_data)
                # print(i, len(self.dataset[i]))
            # print(len(self.dataset), len(self.dataset[0]))
        else:
            for i in range(self.num_class):
                tmp_data = []
                for j in range(len(self.train_data)):
                    if self.train_data.targets[j] == self.classes[i]:
                        tmp_data.append(self.train_data[j])
                self.dataset.append(tmp_data)

    def DL_sampling_single(self):
        np.random.seed(self.seed)
        Sampled_data = [[] for _ in range(self.num_client)]
        self.target = np.arange(self.num_client)
        self.target %= self.num_class
        np.random.shuffle(self.target)
        # print(self.target)
        for i in range(self.num_client):
            for j in range(len(self.train_data)):
                if self.train_data.targets[j] == self.target[i]:
                    Sampled_data[i].append(self.train_data[j])
        return Sampled_data

    def Complete_Random(self):
        np.random.seed(self.seed)
        Sampled_data = [[] for _ in range(self.num_client)]
        indices = np.arange(len(self.train_data), dtype=int)
        np.random.shuffle(indices)
        k = int(len(self.train_data)/self.num_client)
        for i in range(self.num_client):
            client_indices = indices[i*k: (i+1)*k]
            for index in client_indices:
                Sampled_data[i].append(self.train_data[index])
        return Sampled_data

    def Synthesize_sampling(self, alpha):
        Alpha = [alpha for i in range(self.num_class)]
        # Generate samples from the Dirichlet distribution
        samples = np.random.dirichlet(Alpha, size=self.num_client)
        print(samples)
        # Print the generated samples
        num_samples = []
        for i in range(self.num_client):
            num_sample = []
            for j in range(self.num_class):
                sample = np.array(samples[i][j]) * len(self.dataset[j])
                num_sample.append(int(sample))
            num_samples.append(num_sample)
        print(num_samples)
        Sample_data = [[] for i in range(self.num_client)]
        for client in range(self.num_client):
            for i in range(self.num_class):
                class_samples = num_samples[client][i]
                tmp_data = random.sample(self.dataset[i], k=class_samples)
                Sample_data[client] += tmp_data
            np.random.shuffle(Sample_data[client])
        return Sample_data

def average_weights(weights):
    for i in range(len(weights)):
        if i == 0:
            total = weights[i]
        else:
            total += weights[i]
    total = torch.div(total, torch.tensor(len(weights)))
    return total

def Check_Matrix(client, matrix):
    count = 0
    for i in range(client):
        if np.count_nonzero(matrix[i] - matrix.transpose()[i]) == 0:
            pass
        else:
            count += 1
    return count

def shuffle_dataset(dataset):
    return random.shuffle(dataset)

def partition_dirichlet_equal_return_data(device,
    dataset: torch.Tensor,                 # e.g., list/array of samples or (x, y) tuples
    labels: torch.Tensor,  # length N vector of class ids in [0, num_classes-1]
    num_clients: int = 20,
    num_classes: int = 10,
    alpha: float = 0.1,
    seed: int = 42,
    drop_remainder: bool = True,
    return_indices: bool = False,      # set True if you want indices instead of samples
) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Partition `dataset` into `num_clients` shards using Dirichlet(alpha) over `num_classes`,
    ensuring:
      - equal number of samples per client
      - no overlap across clients
      - heterogeneity controlled by `alpha`

    Returns:
      client_data:  list of length num_clients, each a list of *samples* (or indices if return_indices=True)
      client_indices (optional): if return_indices=False => also returns the indices; else returns None
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    labels = labels.to(device)
    N = len(labels)
    assert len(dataset) == N, "dataset and labels must have the same length."

    # Build class -> indices pools (no overlap by construction)
    class_pools = {c: [] for c in range(num_classes)}
    for idx, y in enumerate(labels):
        y = int(y)
        if y < 0 or y >= num_classes:
            raise ValueError(f"Label {y} outside [0, {num_classes-1}] at index {idx}.")
        class_pools[y].append(idx)

    # Shuffle each class pool
    for c in range(num_classes):
        rng.shuffle(class_pools[c])

    # Compute equal per-client size; optionally drop the remainder
    total_samples = sum(len(class_pools[c]) for c in range(num_classes))
    remainder = total_samples % num_clients
    if remainder != 0:
        if not drop_remainder:
            raise ValueError(
                f"Total N={total_samples} not divisible by num_clients={num_clients}. "
                f"Set drop_remainder=True or adjust data."
            )
        # Drop 'remainder' items from the global pool (proportionally by class for fairness)
        # Compute how many to drop per class roughly proportional to class size
        to_drop = remainder
        class_sizes = np.array([len(class_pools[c]) for c in range(num_classes)], dtype=float)
        if class_sizes.sum() == 0:
            raise RuntimeError("Empty dataset after filtering.")
        # initial proportional plan
        drop_plan = np.floor(to_drop * (class_sizes / class_sizes.sum())).astype(int)
        # adjust to match exact to_drop
        while drop_plan.sum() < to_drop:
            # give +1 to the largest remaining class
            c = int(np.argmax(class_sizes - drop_plan))
            drop_plan[c] += 1
        # drop from the end of each class pool
        for c in range(num_classes):
            k = int(drop_plan[c])
            if k > 0 & k <= len(class_pools[c]):
                class_pools[c] = class_pools[c][:-k]
        total_samples = sum(len(class_pools[c]) for c in range(num_classes))

    samples_per_client = total_samples // num_clients
    assert samples_per_client * num_clients == total_samples, "Internal size mismatch after trimming."

    # Helper: draw class counts for a client subject to availability caps
    def draw_counts_with_caps(quota: int, avail: np.ndarray, probs: np.ndarray) -> np.ndarray:
        counts = np.zeros_like(avail, dtype=int)
        remaining = quota
        avail_left = avail.copy().astype(int)

        # build initial weights (zero where unavailable)
        w = probs * (avail_left > 0)
        if w.sum() == 0:
            w = (avail_left > 0).astype(float)
        w = w / w.sum()

        while remaining > 0:
            proposal = rng.multinomial(remaining, w)
            proposal = np.minimum(proposal, avail_left)
            taken = proposal.sum()
            if taken == 0:
                # fallback purely by availability
                w = (avail_left > 0).astype(float)
                if w.sum() == 0:
                    break
                w = w / w.sum()
                continue
            counts += proposal
            avail_left -= proposal
            remaining -= taken
            if remaining == 0:
                break
            w = probs * (avail_left > 0)
            if w.sum() == 0:
                w = (avail_left > 0).astype(float)
            w = w / w.sum()

        if counts.sum() != quota:
            # top off if tiny mismatch (very rare)
            need = quota - counts.sum()
            if need > 0:
                where = np.where(avail_left > 0)[0]
                if where.size < need:
                    raise RuntimeError("Insufficient availability to reach exact quota.")
                # greedily take 1 from the first 'need' classes with availability
                for j in where[:need]:
                    counts[j] += 1
                    avail_left[j] -= 1
        return counts

    # Prepare outputs
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    # Iteratively assign to each client
    for client in range(num_clients):
        # For each client, sample Dirichlet over classes
        probs = rng.dirichlet(alpha * np.ones(num_classes))
        # Current availability
        avail = np.array([len(class_pools[c]) for c in range(num_classes)], dtype=int)
        if avail.sum() < samples_per_client:
            raise RuntimeError(f"Insufficient availability left to fill client {client}.")

        counts = draw_counts_with_caps(samples_per_client, avail, probs)

        # Pop indices from each class pool
        for c in range(num_classes):
            k = int(counts[c])
            if k > 0:
                # take last k (pools already shuffled)
                picked = class_pools[c][-k:]
                del class_pools[c][-k:]
                client_indices[client].extend(picked)

        rng.shuffle(client_indices[client])
        assert len(client_indices[client]) == samples_per_client, "Client quota mismatch."

    # Build actual per-client data lists (samples)
    if return_indices:
        client_data = client_indices
        return client_data, None
    else:
        client_data = [[dataset[i] for i in idxs] for idxs in client_indices]
        return client_data, client_indices


