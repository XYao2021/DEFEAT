import matplotlib.pyplot as plt
import torch
import random
import copy
import numpy as np
from torch.utils.data import DataLoader
from model.model import Model
from util.util import *
from compression import *
from config import *
from dataset.dataset import *
from trans_matrix import *
import time
from datetime import date
import os
from algorithms.algorithms import Algorithms


if device != 'cpu':
    current_device = torch.cuda.current_device()
    torch.cuda.set_device(current_device)
    device = 'cuda:{}'.format(CUDA_ID)

if __name__ == '__main__':
    ACC = []
    ACC_T = []
    LOSS = []
    for seed in Seed_set:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        if dataset == 'FashionMNIST' or 'CIFAR10' or 'MNIST' or 'CIFAR100' or 'EMNIST':
            train_data, test_data = loading(dataset_name=dataset, data_path=dataset_path, device=device)
            NUM_CLASS = len(train_data.classes)

            client_data, client_indices = partition_dirichlet_equal_return_data(
                device=device,
                dataset=train_data,
                labels=train_data.target_transformed,
                num_clients=CLIENTS,
                num_classes=NUM_CLASS,
                alpha=ALPHA,
                seed=seed,
                drop_remainder=True,
                return_indices=False,  # set True if you only want indices
            )

            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)
        else:
            raise Exception('Unrecognized dataset !!!')

        print("......DATA LOADING COMPLETE......")
        # Sample = Sampling(num_client=CLIENTS, num_class=10, train_data=train_data, method='uniform', seed=seed, name=dataset)
        # if DISTRIBUTION == 'Dirichlet':
        #     if ALPHA == 0:
        #         client_data = Sample.DL_sampling_single()
        #     elif ALPHA > 0:
        #         client_data = Sample.Synthesize_sampling(alpha=ALPHA)
        # else:
        #     raise Exception('This data distribution method has not been embedded')

        "Print the details of data distribution"
        # print("Per-client sizes:", [len(cd) for cd in client_data])
        for i in range(CLIENTS):
            count = [0 for i in range(NUM_CLASS)]
            for j in range(len(client_data[i])):
                count[int(client_data[i][j][1])] += 1
            print(i, count)

        models = []
        self_weights = []
        neighbor_weights = []
        data_loader = []
        compressors = []

        "DEFEAT / DEFEAT_Ada"
        residual_errors = []

        "DCD"

        "CHOCO"
        self_accumulate_update = []
        neighbor_accumulate_update = []

        "BEER"
        self_H = []
        neighbor_H = []
        self_G = []
        neighbor_G = []

        "MOTEF"

        "DeepSqueeze"
        self_update = []
        neighbor_update = []

        Transfer = Transform(num_nodes=CLIENTS, num_neighbors=NEIGHBORS, seed=seed, network=NETWORK)
        check = Check_Matrix(CLIENTS, Transfer.matrix)
        if check != 0:
            raise Exception('The Transfer Matrix Should be Symmetric')
        else:
            print(NETWORK, 'Transfer Matrix is Symmetric Matrix', '\n')
        eigenvalues, Gaps = Transfer.Get_alpha_upper_bound_theory()

        test_model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)

        "Initialization"
        for n in range(CLIENTS):
            model = Model(random_seed=seed, learning_rate=LEARNING_RATE, model_name=model_name, device=device, flatten_weight=True, pretrained_model_file=load_model_file)

            # initial_weights = model.get_weights().to(device)
            # initial_neighbor_models = [model.get_weights().to(device) for i in range(len(Transfer.neighbors[n]))]
            # initial_zeros = torch.zeros_like(model.get_weights()).to(device)
            # initial_neighbor_zeros = [torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))]

            models.append(model)
            self_weights.append(model.get_weights().to(device))
            data_loader.append(DataLoader(client_data[n], batch_size=BATCH_SIZE, shuffle=True))

            "DEFEAT / DEFEAT_Ada"
            residual_errors.append(torch.zeros_like(model.get_weights()).to(device))
            neighbor_weights.append([model.get_weights().to(device) for i in range(len(Transfer.neighbors[n]))])

            "DCD"

            "CHOCO"
            self_accumulate_update.append(torch.zeros_like(model.get_weights()).to(device))
            neighbor_accumulate_update.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])

            "BEER"
            self_H.append(torch.zeros_like(model.get_weights()).to(device))
            neighbor_H.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
            self_G.append(torch.zeros_like(model.get_weights()).to(device))
            neighbor_G.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])

            "MOTEF"

            "DeepSqueeze"
            self_update.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])
            neighbor_update.append([torch.zeros_like(model.get_weights()).to(device) for i in range(len(Transfer.neighbors[n]))])

            "Compression initialization"
            normalization = 1
            if COMPRESSION == 'quantization':
                if ADAPTIVE:
                    model_size = len(client_weights[n])
                    normalization = model_size
                if FIRST is True:
                    compressors.append(Quantization_I(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value, device=device))
                else:
                    compressors.append(Quantization_U_1(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value, device=device))  # Unbiased
            elif COMPRESSION == 'topk':
                compressors.append(Top_k(ratio=RATIO, device=device))
            elif COMPRESSION == 'randk':
                compressors.append(Rand_k(ratio=RATIO, device=device))
            else:
                raise Exception('Unknown compression method, please write the compression method first')

        neighbors = list(Transfer.neighbors)

        Algorithm = Algorithms(algorithm=ALGORITHM, compression=COMPRESSION, num_nodes=CLIENTS, neighbors=neighbors, models=models, data_transform=data_transform,
                 device=device, self_weights=self_weights, neighbor_weights=neighbor_weights, data_loader=data_loader, learning_rate=LEARNING_RATE,
                 compressors=compressors, gamma=DISCOUNT, residual_errors=residual_errors, self_accumulate_update=self_accumulate_update, neighbor_accumulate_update=neighbor_accumulate_update,
                 self_H=self_H, neighbor_H=neighbor_H, self_G=self_G, neighbor_G=neighbor_G, lamda=BETA, self_update=self_update,
                 neighbor_update=neighbor_update, average_rate=BETA, normalization=normalization)

        global_loss = []
        Test_acc = []
        Train_acc = []
        iter_num = 0
        print('ALGORITHM: ', ALGORITHM, 'CONSENSUS/GAMMA: ', DISCOUNT, 'MOMENTUM: ', BETA, 'FIRST_TIME: ', FIRST)

        "Training start"
        while True:
            if ALGORITHM == 'DEFEAT':
                Algorithm.DEFEAT(iter_num=iter_num)
            elif ALGORITHM == 'DEFEAT_C':
                Algorithm.DEFEAT_ada(iter_num=iter_num)
            elif ALGORITHM == 'DCD':
                Algorithm.DCD(iter_num=iter_num)
            elif ALGORITHM == 'CHOCO':
                Algorithm.CHOCO(iter_num=iter_num)  # consensus = gamma
            elif ALGORITHM == 'BEER':  # 1
                Algorithm.BEER(iter_num=iter_num)  # consensus = gamma
            elif ALGORITHM == 'MoTEF':  # 1
                Algorithm.MoTEF(iter_num=iter_num)  # consensus = gamma, Lambda = beta
            elif ALGORITHM == 'DeepSqueeze':
                Algorithm.DeepsSqueeze(iter_num=iter_num)
            else:
                raise Exception('Unknown algorithm, please update the algorithm codes')

            iter_num += 1

            "Need to change the testing model to local model rather than global averaged model"
            if TEST == 'average':
                test_weights = average_weights([Algorithm.models[i].get_weights() for i in range(CLIENTS)])  # test with global averaged model
            elif TEST == 'local':
                test_weights = average_weights([Algorithm.models[i].get_weights() for i in neighbors[0]])  # test with local model

            train_loss, train_acc = test_model.accuracy(weights=test_weights, test_loader=train_loader, device=device)
            test_loss, test_acc = test_model.accuracy(weights=test_weights, test_loader=test_loader, device=device)

            global_loss.append(train_loss)
            Test_acc.append(test_acc)
            Train_acc.append(train_loss)
            print('SEED |', seed, '| iteration |', iter_num, '| Global Loss', round(train_loss, 6), '| Training Accuracy |',
                  round(train_acc, 4), '| Test Accuracy |', round(test_acc, 4), '\n')

            if iter_num >= AGGREGATION:
                ACC += Test_acc
                LOSS += global_loss
                ACC_T += Train_acc
#                 if dataset == "CIFAR10" or "CIFAR100":
#                     txt_list = [ACC, '\n', LOSS]
#                     # print([compressors[i].discount_parameter for i in range(CLIENTS)])
#                     if COMPRESSION == 'quantization':
#                         f = open(
#                             '{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|.txt'.format(ALGORITHM, ALPHA, QUANTIZE_LEVEL, DISCOUNT,
#                                                                               TEST, dataset, LEARNING_RATE, BETA, CLIENTS,
#                                                                               NEIGHBORS, date.today(),
#                                                                               time.strftime("%H:%M:%S", time.localtime())),
#                             'w')
#                     elif COMPRESSION == 'topk' or 'randk':
#                         f = open('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|.txt'.format(ALGORITHM, ALPHA, RATIO, DISCOUNT, TEST,
#                                                                                    dataset, LEARNING_RATE, BETA, CLIENTS,
#                                                                                    NEIGHBORS, date.today(),
#                                                                                    time.strftime("%H:%M:%S",
#                                                                                                  time.localtime())), 'w')
#                     else:
#                         raise Exception('Unknown compression method')

#                     for item in txt_list:
#                         f.write("%s\n" % item)
                break
        del models
        del self_weights

        torch.cuda.empty_cache()  # Clean the memory cache

    if STORE == 1:
        if FIRST is True:
            Maxes = []
            Mines = []
            for i in range(CLIENTS):
                Maxes.append(max(compressors[i].max))
                Mines.append(min(compressors[i].min))
            txt_list = [Maxes, '\n', Mines, '\n', ACC, '\n', LOSS]
            print(max(Maxes), min(Maxes), max(Mines), min(Mines))
        else:
            # txt_list = [ACC, '\n', LOSS, '\n', ACC_T]
            txt_list = [Algorithm.vectors]

        if COMPRESSION == 'quantization':
            f = open('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|final.txt'.format(ALGORITHM, ALPHA, QUANTIZE_LEVEL, DISCOUNT, TEST, dataset, LEARNING_RATE, BETA, CLIENTS, NEIGHBORS, date.today(), time.strftime("%H:%M:%S", time.localtime())), 'w')
        elif COMPRESSION == 'topk' or 'randk':
            f = open('{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|final.txt'.format(ALGORITHM, ALPHA, RATIO, DISCOUNT, TEST, dataset, LEARNING_RATE, BETA, CLIENTS, NEIGHBORS, date.today(), time.strftime("%H:%M:%S", time.localtime())), 'w')
        else:
            raise Exception('Unknown compression method')

        for item in txt_list:
            f.write("%s\n" % item)
    else:
        print('NOT STORE THE RESULTS THIS TIME')
