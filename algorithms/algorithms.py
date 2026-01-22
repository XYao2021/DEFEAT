# import copy
# import numpy as np
# import torch
# import random
# import copy
# import time
#
# class Algorithms:
#     def __init__(self, algorithm=None, compression=None, num_nodes=None, neighbors=None, models=None, data_transform=None,
#                  device=None, self_weights=None, neighbor_weights=None, data_loader=None, learning_rate=None,
#                  compressors=None, gamma=None, residual_errors=None, self_accumulate_update=None, neighbor_accumulate_update=None,
#                  self_H=None, neighbor_H=None, self_G=None, neighbor_G=None, lamda=None, self_update=None,
#                  neighbor_update=None, average_rate=None, normalization=None):
#         super().__init__()
#
#         "Common"
#         self.name = algorithm
#         self.compression = compression
#         self.num_clients = num_nodes
#         self.neighbors = neighbors
#         self.device = device
#         self.data_transform = data_transform
#
#         self.models = models
#         self.self_weights = self_weights
#         self.data_loaders = data_loader
#         self.learning_rate = learning_rate
#         self.compressors = compressors
#
#         "DEFEAT / DEFEAT_Ada"
#         self.gamma = gamma
#         self.residual_errors = residual_errors
#         self.neighbor_weights = neighbor_weights
#
#         "DCD"
#
#         "CHOCO"
#         self.self_accumulate_update = self_accumulate_update
#         self.neighbor_accumulate_update = neighbor_accumulate_update
#         self.consensus = gamma
#
#         "BEER"
#         self.self_H = self_H
#         self.neighbor_H = neighbor_H
#         self.self_G = self_G
#         self.neighbor_G = neighbor_G
#         self.self_V = []
#         self.previous_gradient = []
#
#         "MOTEF"
#         self.self_M = []
#         self.lamda = lamda
#
#         "DeepSqueeze"
#         self.self_update = self_update
#         self.neighbor_update = neighbor_update
#         self.average_rate = average_rate
#         self.tmp_weights = self_update
#
#         "Time cost recording"
#         self.communication_cost = 0
#         self.computation_cost = 0
#
#     def _logger(self):
#         print(' compression method:', self.compression, '\n',
#               'running algorithm: ', self.name, '\n')
#
#     def _training(self, data_loader, client_weights, model):  # Only consider 1 inner iteration per aggregation
#         model.assign_weights(weights=client_weights)
#         model.model.train()
#
#         images, labels = data_loader
#         images, labels = images.to(self.device), labels.to(self.device)
#
#         # if self.data_transform is not None:
#         #     images = self.data_transform(images)
#
#         model.optimizer.zero_grad()
#         pred = model.model(images)
#         loss = model.loss_function(pred, labels)
#         loss.backward()
#         model.optimizer.step()
#
#         trained_model = model.get_weights()  # x_t - \eta * gradients
#         return trained_model
#
#     def _average_updates(self, updates):
#         Averaged_weights = []
#         for i in range(self.num_clients):
#             Averaged_weights.append(sum(updates[i]) / len(updates[i]))
#         return Averaged_weights
#
#     def _averaged_choco(self, updates, update):
#         Averaged = []
#         for i in range(self.num_clients):
#             summation = torch.zeros_like(update[0])
#             for j in range(len(updates[i])):
#                 summation += (1/len(updates[i])) * (updates[i][j] - update[i])
#             Averaged.append(summation)
#         return Averaged
#
#     def _check_weights(self, client_weights, neighbors_weights):
#         checks = 0
#         for n in range(self.num_clients):
#             neighbors = self.neighbors[n]
#             neighbors_models = neighbors_weights[n]
#
#             check = 0
#             for m in range(len(neighbors)):
#                 if torch.equal(neighbors_models[m], client_weights[neighbors[m]]):
#                     check += 1
#                 else:
#                     pass
#             if check == len(self.neighbors[n]):
#                 checks += 1
#             else:
#                 pass
#         if checks == self.num_clients:
#             return True
#         else:
#             return False
#
#     def DEFEAT(self, iter_num):
#         weighted_average = self._average_updates(updates=self.neighbor_weights)
#         for n in range(self.num_clients):
#             images, labels = next(iter(self.data_loaders[n]))
#             trained_weights = self._training(data_loader=[images, labels],
#                                            client_weights=self.self_weights[n], model=self.models[n])
#             gradients = trained_weights - self.self_weights[n]  # - eta * gradients
#             # print(torch.sum(torch.square(gradients)))
#
#             b_t = weighted_average[n] - self.self_weights[n] + gradients + self.gamma * self.residual_errors[n]
#             v_t, current_residual = self.compressors[n].get_trans_bits_and_residual(w_tmp=b_t)
#
#             self.residual_errors[n] = current_residual + (1 - self.gamma) * self.residual_errors[n]
#             self.self_weights[n] += v_t
#             for m in range(self.num_clients):
#                 if n in self.neighbors[m]:
#                     self.neighbor_weights[m][self.neighbors[m].index(n)] += v_t
#
#     def DEFEAT_ada(self, iter_num):
#         epsilon = 0.000000000001
#         weighted_average = self._average_updates(updates=self.neighbor_weights)
#         for n in range(self.num_clients):
#             images, labels = next(iter(self.data_loaders[n]))
#             trained_weights = self._training(data_loader=[images, labels],
#                                              client_weights=self.self_weights[n], model=self.models[n])
#             gradients = trained_weights - self.self_weights[n]  # - eta * gradients
#
#             if iter_num == 0:
#                 error_norm = 1
#             else:
#                 error_norm = torch.sum(torch.square(self.residual_errors[n])).item()
#
#             # gamma = min(np.sqrt(torch.sum(torch.square(gradients)).item() / (error_norm + epsilon)), 0.3)
#             # print(iter_num, n, gamma)
#             gamma = min(max(np.sqrt(torch.sum(torch.square(gradients)).item() / (error_norm + epsilon)), 0.3), 1.7)
#             # gamma = min(max(np.sqrt(torch.sum(torch.square(gradients)).item() / (error_norm + epsilon)), 0.3), 1.0)
#
#             b_t = weighted_average[n] - self.self_weights[n] + gradients + gamma * self.residual_errors[n]
#             v_t, current_residual = self.compressors[n].get_trans_bits_and_residual(w_tmp=b_t)
#
#             self.residual_errors[n] = current_residual + (1 - gamma) * self.residual_errors[n]
#             self.self_weights[n] += v_t
#             for m in range(self.num_clients):
#                 if n in self.neighbors[m]:
#                     self.neighbor_weights[m][self.neighbors[m].index(n)] += v_t
#
#     def DCD(self, iter_num):
#         weighted_average = self._average_updates(updates=self.neighbor_weights)
#         for n in range(self.num_clients):
#             images, labels = next(iter(self.data_loaders[n]))
#             trained_weights = self._training(data_loader=[images, labels],
#                                              client_weights=self.self_weights[n], model=self.models[n])
#             gradients = trained_weights - self.self_weights[n]  # - eta * gradients
#
#             x_tmp = weighted_average[n] + gradients
#             z_t = x_tmp - self.self_weights[n]
#             z_t, current_residual = self.compressors[n].get_trans_bits_and_residual(w_tmp=z_t)
#
#             self.self_weights[n] += z_t
#             for m in range(self.num_clients):
#                 if n in self.neighbors[m]:
#                     self.neighbor_weights[m][self.neighbors[m].index(n)] += z_t
#
#     def CHOCO(self, iter_num):
#         weighted_average = self._averaged_choco(updates=self.neighbor_accumulate_update, update=self.self_accumulate_update)
#         for n in range(self.num_clients):
#             images, labels = next(iter(self.data_loaders[n]))
#             tmp_weight = self._training(data_loader=[images, labels],
#                                         client_weights=self.self_weights[n], model=self.models[n])
#             self.self_weights[n] = tmp_weight + self.consensus * weighted_average[n]
#
#             q_t = self.self_weights[n] - self.self_accumulate_update[n]
#             q_t, current_residual = self.compressors[n].get_trans_bits_and_residual(w_tmp=q_t)
#
#             self.self_accumulate_update[n] += q_t
#             for m in range(self.num_clients):
#                 if n in self.neighbors[m]:
#                     self.neighbor_accumulate_update[m][self.neighbors[m].index(n)] += q_t
#
#     def BEER(self, iter_num):
#         weighted_average_H = self._averaged_choco(updates=self.neighbor_H, update=self.self_H)
#         weighted_average_G = self._averaged_choco(updates=self.neighbor_G, update=self.self_G)
#
#         for n in range(self.num_clients):
#             if iter_num == 0:
#                 images, labels = next(iter(self.data_loaders[n]))
#                 initial_trained_weight = self._training(data_loader=[images, labels],
#                                                         client_weights=self.self_weights[n], model=self.models[n])
#                 initial_gradient = (self.self_weights[n] - initial_trained_weight) / self.learning_rate
#                 self.self_V.append(initial_gradient)
#                 self.previous_gradient.append(initial_gradient)
#
#             self.self_weights[n] = self.self_weights[n] + self.gamma * weighted_average_H[n] - self.learning_rate * self.self_V[n]
#             q_h = self.self_weights[n] - self.self_H[n]
#             q_h, _ = self.compressors[n].get_trans_bits_and_residual(w_tmp=q_h)
#
#             self.self_H[n] += q_h
#             for m in range(self.num_clients):
#                 if n in self.neighbors[m]:
#                     self.neighbor_H[m][self.neighbors[m].index(n)] += q_h
#
#             images, labels = next(iter(self.data_loaders[n]))
#             next_trained_weight = self._training(data_loader=[images, labels],
#                                                  client_weights=self.self_weights[n], model=self.models[n])
#             next_gradient = (self.self_weights[n] - next_trained_weight) / self.learning_rate
#
#             self.self_V[n] = self.self_V[n] + self.gamma * weighted_average_G[n] + next_gradient - self.previous_gradient[n]
#             self.previous_gradient[n] = next_gradient
#
#             q_g = self.self_V[n] - self.self_G[n]
#             q_g, _ = self.compressors[n].get_trans_bits_and_residual(w_tmp=q_g)
#
#             self.self_G[n] += q_g
#             for m in range(self.num_clients):
#                 if n in self.neighbors[m]:
#                     self.neighbor_G[m][self.neighbors[m].index(n)] += q_g
#
#     def MoTEF(self, iter_num):
#         if iter_num == 0:
#             for n in range(self.num_clients):
#                 images, labels = next(iter(self.data_loaders[n]))
#                 initial_trained_weight = self._training(data_loader=[images, labels],
#                                                         client_weights=self.self_weights[n], model=self.models[n])
#                 initial_gradient = (self.self_weights[n] - initial_trained_weight) / self.learning_rate
#                 self.self_V.append(initial_gradient)
#                 self.self_M.append(initial_gradient)
#                 self.self_G[n] = initial_gradient
#                 self.self_H[n] = self.self_weights[n]
#                 for m in range(self.num_clients):
#                     if n in self.neighbors[m]:
#                         self.neighbor_G[m][self.neighbors[m].index(n)] = initial_gradient
#                         self.neighbor_H[m][self.neighbors[m].index(n)] = self.self_weights[n]
#
#         weighted_average_H = self._averaged_choco(updates=self.neighbor_H, update=self.self_H)
#         weighted_average_G = self._averaged_choco(updates=self.neighbor_G, update=self.self_G)
#         for n in range(self.num_clients):
#             self.self_weights[n] += self.gamma * weighted_average_H[n] - self.learning_rate * self.self_V[n]
#             q_h = self.self_weights[n] - self.self_H[n]
#             q_h, _ = self.compressors[n].get_trans_bits_and_residual(w_tmp=q_h)
#
#             self.self_H[n] += q_h
#             for m in range(self.num_clients):
#                 if n in self.neighbors[m]:
#                     self.neighbor_H[m][self.neighbors[m].index(n)] += q_h
#
#             images, labels = next(iter(self.data_loaders[n]))
#             next_trained_weight = self._training(data_loader=[images, labels],
#                                                  client_weights=self.self_weights[n], model=self.models[n])
#             next_gradient = (self.self_weights[n] - next_trained_weight) / self.learning_rate
#             current_M = copy.deepcopy(self.self_M[n])
#             self.self_M[n] = (1 - self.lamda) * self.self_M[n] + self.lamda * next_gradient
#             self.self_V[n] += self.gamma * weighted_average_G[n] + self.self_M[n] - current_M
#
#             q_g = self.self_V[n] - self.self_G[n]
#             q_g, _ = self.compressors[n].get_trans_bits_and_residual(w_tmp=q_g)
#
#             self.self_G[n] += q_g
#             for m in range(self.num_clients):
#                 if n in self.neighbors[m]:
#                     self.neighbor_G[m][self.neighbors[m].index(n)] += q_g
#
#     def DeepsSqueeze(self, iter_num):  # Need to be checked, the original DeepSqueeze algorithm is not correct.
#         for n in range(self.num_clients):
#             images, labels = next(iter(self.data_loaders[n]))
#             current_trained_weight = self._training(data_loader=[images, labels],
#                                                     client_weights=self.self_weights[n], model=self.models[n])
#             current_gradient = (self.self_weights[n] - current_trained_weight) / self.learning_rate
#
#             self.tmp_weights[n] = self.self_weights[n] - self.learning_rate * current_gradient
#             v_t = self.self_weights[n] - self.learning_rate * current_gradient + self.residual_errors[n]
#             compressed_v, self.residual_errors[n] = self.compressors[n].get_trans_bits_and_residual(w_tmp=v_t)
#
#             self.self_update[n] = compressed_v
#             for m in range(self.num_clients):
#                 if n in self.neighbors[m]:
#                     self.neighbor_update[m][self.neighbors[m].index(n)] = compressed_v
#
#         weighted_average_update = self._averaged_choco(updates=self.neighbor_update, update=self.self_update)
#
#         for n in range(self.num_clients):
#             self.self_weights[n] = self.tmp_weights[n] + self.average_rate * weighted_average_update[n]
#
#     def CEDAS(self):
#         pass
#
#     def DeCoM(self):
#         pass

import copy
import numpy as np
import torch
import random
import copy
import time
import os

def save_gradient(
    grad_tensor,
    algorithm,
    iteration,
    var_name,
    var_value,
    base_dir="logs"
):
    """
    Save gradient vector into a SINGLE directory whose name
    encodes algorithm, iteration, and one variable.
    """

    # Build ONE directory name
    # dir_name = f"{algorithm}_iter{iteration:06d}_{var_name}_{var_value}"
    save_dir = os.path.join(base_dir)

    # ✅ Create directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Convert gradient to NumPy
    grad_np = grad_tensor.detach().cpu().numpy().astype(np.float32)

    # Save gradient
    save_path = os.path.join(save_dir, "{}_{}_{}.npy".format(algorithm, var_value, iteration))
    np.save(save_path, grad_np)

class Algorithms:
    def __init__(self, algorithm=None, compression=None, num_nodes=None, neighbors=None, models=None, data_transform=None,
                 device=None, self_weights=None, neighbor_weights=None, data_loader=None, learning_rate=None,
                 compressors=None, gamma=None, residual_errors=None, self_accumulate_update=None, neighbor_accumulate_update=None,
                 self_H=None, neighbor_H=None, self_G=None, neighbor_G=None, lamda=None, self_update=None,
                 neighbor_update=None, average_rate=None, normalization=None):
        super().__init__()

        "Common"
        self.name = algorithm
        self.compression = compression
        self.num_clients = num_nodes
        self.neighbors = neighbors
        self.device = device
        self.data_transform = data_transform

        self.models = models
        self.self_weights = self_weights
        self.data_loaders = data_loader
        self.learning_rate = learning_rate
        self.compressors = compressors

        "DEFEAT / DEFEAT_Ada"
        self.gamma = gamma
        self.residual_errors = residual_errors
        self.neighbor_weights = neighbor_weights

        "DCD"

        "CHOCO"
        self.self_accumulate_update = self_accumulate_update
        self.neighbor_accumulate_update = neighbor_accumulate_update
        self.consensus = gamma

        "BEER"
        self.self_H = self_H
        self.neighbor_H = neighbor_H
        self.self_G = self_G
        self.neighbor_G = neighbor_G
        self.self_V = []
        self.previous_gradient = []

        "MOTEF"
        self.self_M = []
        self.lamda = lamda

        "DeepSqueeze"
        self.self_update = self_update
        self.neighbor_update = neighbor_update
        self.average_rate = average_rate
        self.tmp_weights = self_update

        "Debugging"
        self.gradient_norm = []
        self.error_norm = []
        self.update_norm = []
        self.gradient_cos = []
        self.update_cos = []
        self.vectors = []

    def _logger(self):
        print(' compression method:', self.compression, '\n',
              'running algorithm: ', self.name, '\n')

    def _training(self, data_loader, client_weights, model):  # Only consider 1 inner iteration per aggregation
        model.assign_weights(weights=client_weights)
        model.model.train()

        images, labels = data_loader
        images, labels = images.to(self.device), labels.to(self.device)

        # if self.data_transform is not None:
        #     images = self.data_transform(images)

        model.optimizer.zero_grad()
        pred = model.model(images)
        loss = model.loss_function(pred, labels)
        loss.backward()
        model.optimizer.step()

        trained_model = model.get_weights()  # x_t - \eta * gradients
        return trained_model

    def _average_updates(self, updates):
        Averaged_weights = []
        for i in range(self.num_clients):
            Averaged_weights.append(sum(updates[i]) / len(updates[i]))
        return Averaged_weights

    def _averaged_choco(self, updates, update):
        Averaged = []
        for i in range(self.num_clients):
            summation = torch.zeros_like(update[0])
            for j in range(len(updates[i])):
                summation += (1/len(updates[i])) * (updates[i][j] - update[i])
            Averaged.append(summation)
        return Averaged

    def _check_weights(self, client_weights, neighbors_weights):
        checks = 0
        for n in range(self.num_clients):
            neighbors = self.neighbors[n]
            neighbors_models = neighbors_weights[n]

            check = 0
            for m in range(len(neighbors)):
                if torch.equal(neighbors_models[m], client_weights[neighbors[m]]):
                    check += 1
                else:
                    pass
            if check == len(self.neighbors[n]):
                checks += 1
            else:
                pass
        if checks == self.num_clients:
            return True
        else:
            return False

    def cosine_similarity(self, gradient, error, eps=1e-12):
        """
        Compute cosine similarity between two vectors g and e.
        g, e: torch.Tensor with the same shape
        """
        dot = torch.dot(gradient.view(-1), error.view(-1))
        g_norm = torch.norm(gradient)
        e_norm = torch.norm(error)
        return dot / (g_norm * e_norm + eps)

    def DEFEAT(self, iter_num):
        weighted_average = self._average_updates(updates=self.neighbor_weights)
        for n in range(self.num_clients):
            images, labels = next(iter(self.data_loaders[n]))
            trained_weights = self._training(data_loader=[images, labels],
                                           client_weights=self.self_weights[n], model=self.models[n])
            gradients = trained_weights - self.self_weights[n]  # - eta * gradients
            # print(torch.sum(torch.square(gradients)))

            b_t = weighted_average[n] - self.self_weights[n] + gradients + self.gamma * self.residual_errors[n]
            v_t, current_residual = self.compressors[n].get_trans_bits_and_residual(w_tmp=b_t)

            if n == 0:
                save_gradient(
                    gradients + self.gamma * self.residual_errors[n],
                    self.name,
                    iter_num,
                    "fix_gamma",
                    self.gamma,
                    base_dir="{}_{}".format(self.name, self.gamma)
                )

            self.residual_errors[n] = current_residual + (1 - self.gamma) * self.residual_errors[n]
            self.self_weights[n] += v_t
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_weights[m][self.neighbors[m].index(n)] += v_t

    def DEFEAT_ada(self, iter_num):
        epsilon = 0.000000000001
        weighted_average = self._average_updates(updates=self.neighbor_weights)
        for n in range(self.num_clients):
            images, labels = next(iter(self.data_loaders[n]))
            trained_weights = self._training(data_loader=[images, labels],
                                             client_weights=self.self_weights[n], model=self.models[n])
            gradients = trained_weights - self.self_weights[n]  # - eta * gradients

            if iter_num == 0:
                error_norm = 1
            else:
                error_norm = torch.sum(torch.square(self.residual_errors[n])).item()

            # gamma = min(np.sqrt(torch.sum(torch.square(gradients)).item() / (error_norm + epsilon)), 0.3)
            # print(iter_num, n, gamma)
            gamma = min(max(np.sqrt(torch.sum(torch.square(gradients)).item() / (error_norm + epsilon)), 0.3), 1.7)
            # gamma = min(max(np.sqrt(torch.sum(torch.square(gradients)).item() / (error_norm + epsilon)), 0.3), 1.0)

            b_t = weighted_average[n] - self.self_weights[n] + gradients + gamma * self.residual_errors[n]
            v_t, current_residual = self.compressors[n].get_trans_bits_and_residual(w_tmp=b_t)

            if n == 0:
                save_gradient(
                    gradients + gamma * self.residual_errors[n],
                    self.name,
                    iter_num,
                    "ada_gamma",
                    "ada",
                    base_dir="{}_ada_no_comp_grad".format(self.name)
                )

            self.residual_errors[n] = current_residual + (1 - gamma) * self.residual_errors[n]
            self.self_weights[n] += v_t
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_weights[m][self.neighbors[m].index(n)] += v_t

    def DCD(self, iter_num):
        weighted_average = self._average_updates(updates=self.neighbor_weights)
        for n in range(self.num_clients):
            images, labels = next(iter(self.data_loaders[n]))
            trained_weights = self._training(data_loader=[images, labels],
                                             client_weights=self.self_weights[n], model=self.models[n])
            gradients = trained_weights - self.self_weights[n]  # - eta * gradients

            x_tmp = weighted_average[n] + gradients
            z_t = x_tmp - self.self_weights[n]
            z_t, current_residual = self.compressors[n].get_trans_bits_and_residual(w_tmp=z_t)

            self.self_weights[n] += z_t
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_weights[m][self.neighbors[m].index(n)] += z_t

    def CHOCO(self, iter_num):
        weighted_average = self._averaged_choco(updates=self.neighbor_accumulate_update, update=self.self_accumulate_update)
        for n in range(self.num_clients):
            images, labels = next(iter(self.data_loaders[n]))
            tmp_weight = self._training(data_loader=[images, labels],
                                        client_weights=self.self_weights[n], model=self.models[n])
            self.self_weights[n] = tmp_weight + self.consensus * weighted_average[n]

            q_t = self.self_weights[n] - self.self_accumulate_update[n]
            q_t, current_residual = self.compressors[n].get_trans_bits_and_residual(w_tmp=q_t)

            self.self_accumulate_update[n] += q_t
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_accumulate_update[m][self.neighbors[m].index(n)] += q_t

    def BEER(self, iter_num):
        weighted_average_H = self._averaged_choco(updates=self.neighbor_H, update=self.self_H)
        weighted_average_G = self._averaged_choco(updates=self.neighbor_G, update=self.self_G)

        for n in range(self.num_clients):
            if iter_num == 0:
                images, labels = next(iter(self.data_loaders[n]))
                initial_trained_weight = self._training(data_loader=[images, labels],
                                                        client_weights=self.self_weights[n], model=self.models[n])
                initial_gradient = (self.self_weights[n] - initial_trained_weight) / self.learning_rate
                self.self_V.append(initial_gradient)
                self.previous_gradient.append(initial_gradient)

            self.self_weights[n] = self.self_weights[n] + self.gamma * weighted_average_H[n] - self.learning_rate * self.self_V[n]
            q_h = self.self_weights[n] - self.self_H[n]
            q_h, _ = self.compressors[n].get_trans_bits_and_residual(w_tmp=q_h)

            self.self_H[n] += q_h
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_H[m][self.neighbors[m].index(n)] += q_h

            images, labels = next(iter(self.data_loaders[n]))
            next_trained_weight = self._training(data_loader=[images, labels],
                                                 client_weights=self.self_weights[n], model=self.models[n])
            next_gradient = (self.self_weights[n] - next_trained_weight) / self.learning_rate

            self.self_V[n] = self.self_V[n] + self.gamma * weighted_average_G[n] + next_gradient - self.previous_gradient[n]
            self.previous_gradient[n] = next_gradient

            q_g = self.self_V[n] - self.self_G[n]
            q_g, _ = self.compressors[n].get_trans_bits_and_residual(w_tmp=q_g)

            self.self_G[n] += q_g
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_G[m][self.neighbors[m].index(n)] += q_g

    def MoTEF(self, iter_num):
        if iter_num == 0:
            for n in range(self.num_clients):
                images, labels = next(iter(self.data_loaders[n]))
                initial_trained_weight = self._training(data_loader=[images, labels],
                                                        client_weights=self.self_weights[n], model=self.models[n])
                initial_gradient = (self.self_weights[n] - initial_trained_weight) / self.learning_rate
                self.self_V.append(initial_gradient)
                self.self_M.append(initial_gradient)
                self.self_G[n] = initial_gradient
                self.self_H[n] = self.self_weights[n]
                for m in range(self.num_clients):
                    if n in self.neighbors[m]:
                        self.neighbor_G[m][self.neighbors[m].index(n)] = initial_gradient
                        self.neighbor_H[m][self.neighbors[m].index(n)] = self.self_weights[n]

        weighted_average_H = self._averaged_choco(updates=self.neighbor_H, update=self.self_H)
        weighted_average_G = self._averaged_choco(updates=self.neighbor_G, update=self.self_G)
        for n in range(self.num_clients):
            self.self_weights[n] += self.gamma * weighted_average_H[n] - self.learning_rate * self.self_V[n]
            q_h = self.self_weights[n] - self.self_H[n]
            q_h, _ = self.compressors[n].get_trans_bits_and_residual(w_tmp=q_h)

            self.self_H[n] += q_h
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_H[m][self.neighbors[m].index(n)] += q_h

            images, labels = next(iter(self.data_loaders[n]))
            next_trained_weight = self._training(data_loader=[images, labels],
                                                 client_weights=self.self_weights[n], model=self.models[n])
            next_gradient = (self.self_weights[n] - next_trained_weight) / self.learning_rate
            current_M = copy.deepcopy(self.self_M[n])
            self.self_M[n] = (1 - self.lamda) * self.self_M[n] + self.lamda * next_gradient
            self.self_V[n] += self.gamma * weighted_average_G[n] + self.self_M[n] - current_M

            q_g = self.self_V[n] - self.self_G[n]
            q_g, _ = self.compressors[n].get_trans_bits_and_residual(w_tmp=q_g)

            self.self_G[n] += q_g
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_G[m][self.neighbors[m].index(n)] += q_g

    def DeepsSqueeze(self, iter_num):  # Need to be checked, the original DeepSqueeze algorithm is not correct.
        for n in range(self.num_clients):
            images, labels = next(iter(self.data_loaders[n]))
            current_trained_weight = self._training(data_loader=[images, labels],
                                                    client_weights=self.self_weights[n], model=self.models[n])
            current_gradient = (self.self_weights[n] - current_trained_weight) / self.learning_rate

            self.tmp_weights[n] = self.self_weights[n] - self.learning_rate * current_gradient
            v_t = self.self_weights[n] - self.learning_rate * current_gradient + self.residual_errors[n]
            compressed_v, self.residual_errors[n] = self.compressors[n].get_trans_bits_and_residual(w_tmp=v_t)

            self.self_update[n] = compressed_v
            for m in range(self.num_clients):
                if n in self.neighbors[m]:
                    self.neighbor_update[m][self.neighbors[m].index(n)] = compressed_v

        weighted_average_update = self._averaged_choco(updates=self.neighbor_update, update=self.self_update)

        for n in range(self.num_clients):
            self.self_weights[n] = self.tmp_weights[n] + self.average_rate * weighted_average_update[n]

    def CEDAS(self):
        pass

    def DeCoM(self):
        pass

