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
import math

if device != 'cpu':
    current_device = torch.cuda.current_device()
    torch.cuda.set_device(current_device)
    device = 'cuda:{}'.format(CUDA_ID)

if __name__ == '__main__':
    COMM = []
    ALPHAS = []
    MAXES = []

    Learning_rates = [0.001, 0.01, 0.0316, 0.056, 0.1, 0.2]
    # Learning_rates = [0.001]
    if ALGORITHM == 'BEER':
        BETAS = [1]
        Gamma = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        # Gamma = [0.05]
    elif ALGORITHM == 'DeepSqueeze':
        BETAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Average step
        Gamma = [1.0]  # gamma
        # BETAS = [0.1]  # Average step
        # Gamma = [0.1]  # gamma
    elif ALGORITHM == 'MoTEF':
        BETAS = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]  # Lambda
        Gamma = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # gamma
        # BETAS = [0.005]  # Lambda
        # Gamma = [0.05]  # gamma
    # elif ALGORITHM == 'DEFEAT':
    #     BETAS = [1]  # Lambda
    #     Gamma = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    elif ALGORITHM == 'DEFEAT_C':
        BETAS = [1]  # Lambda
        Gamma = [1]
    elif ALGORITHM == 'CHOCO':
        BETAS = [1]
        Gamma = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        # Gamma = [0.1]
    elif ALGORITHM == 'DCD':
        BETAS = [1]
        Gamma = [1]

    print('Algorithm: ', ALGORITHM)
    print('Learning rate: ', Learning_rates)
    print('Lambda range: ', BETAS)
    print('Gamma range: ', Gamma)

    if ALGORITHM == 'DEFEAT':
        max_value = 0.4066
        min_value = -0.2881
    elif ALGORITHM == 'DCD':
        max_value = 0.4038
        min_value = -0.2891
    elif ALGORITHM == 'CHOCO':
        max_value = 0.30123514
        min_value = -0.21583036
    elif ALGORITHM == 'BEER':
        max_value = 3.6578
        min_value = -3.3810
    elif ALGORITHM == 'DeepSqueeze':
        max_value = 0.4066
        min_value = -0.2881
    elif ALGORITHM == 'CEDAS':
        max_value = 0.0525
        min_value = -0.0233
    elif ALGORITHM == 'MoTEF':
        max_value = 1.9098
        min_value = -2.7054

    Seed_set = [24]

    beta_loss = []
    beta_lr = []
    beta_cons = []

    start_time = time.time()
    for beta in BETAS:
        gamma_lr = []
        gamma_loss = []
        for cons in range(len(Gamma)):
            lr_loss = []
            for lr in range(len(Learning_rates)):
                ACC = []
                LOSS = []
                for seed in Seed_set:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                    if dataset == 'FashionMNIST' or 'CIFAR10' or 'MNIST' or 'KMNIST':
                        train_data, test_data = loading(dataset_name=dataset, data_path=dataset_path, device=device)
                        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
                        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)
                    else:
                        raise Exception('Unrecognized dataset !!!')

                    # train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
                    # test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=0)

                    print("......DATA LOADING COMPLETE......")
                    Sample = Sampling(num_client=CLIENTS, num_class=10, train_data=train_data, method='uniform',
                                      seed=seed, name=dataset)
                    if DISTRIBUTION == 'Dirichlet':
                        if ALPHA == 0:
                            client_data = Sample.DL_sampling_single()
                        elif ALPHA > 0:
                            client_data = Sample.Synthesize_sampling(alpha=ALPHA)
                    else:
                        raise Exception('This data distribution method has not been embedded')

                    print('gamma / consensus: ', Gamma[cons], 'lr: ', Learning_rates[lr], 'beta: ', beta)

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
                        # print(NETWORK, 'Transfer Matrix is Symmetric Matrix', '\n')
                        pass
                    eigenvalues, Gaps = Transfer.Get_alpha_upper_bound_theory()
                    test_model = Model(random_seed=seed, learning_rate=Learning_rates[lr], model_name=model_name,
                                       device=device, flatten_weight=True, pretrained_model_file=load_model_file)

                    "Initialization"
                    for n in range(CLIENTS):
                        model = Model(random_seed=seed, learning_rate=Learning_rates[lr], model_name=model_name,
                                      device=device, flatten_weight=True, pretrained_model_file=load_model_file)

                        # initial_weights = model.get_weights()
                        # initial_neighbor_models = [model.get_weights() for i in range(len(Transfer.neighbors[n]))]
                        # initial_zeros = torch.zeros_like(model.get_weights()).to(device)
                        # initial_neighbor_zeros = [torch.zeros_like(model.get_weights()) for i in range(len(Transfer.neighbors[n]))]

                        models.append(model)
                        self_weights.append(model.get_weights().to(device))
                        data_loader.append(DataLoader(client_data[n], batch_size=BATCH_SIZE, shuffle=True))

                        "DEFEAT / DEFEAT_Ada"
                        residual_errors.append(torch.zeros_like(model.get_weights()).to(device))
                        neighbor_weights.append(
                            [model.get_weights().to(device) for i in range(len(Transfer.neighbors[n]))])

                        "DCD"

                        "CHOCO"
                        self_accumulate_update.append(torch.zeros_like(model.get_weights()).to(device))
                        neighbor_accumulate_update.append([torch.zeros_like(model.get_weights()).to(device) for i in
                                                           range(len(Transfer.neighbors[n]))])

                        "BEER"
                        self_H.append(torch.zeros_like(model.get_weights()).to(device))
                        neighbor_H.append([torch.zeros_like(model.get_weights()).to(device) for i in
                                           range(len(Transfer.neighbors[n]))])
                        self_G.append(torch.zeros_like(model.get_weights()).to(device))
                        neighbor_G.append([torch.zeros_like(model.get_weights()).to(device) for i in
                                           range(len(Transfer.neighbors[n]))])

                        "MOTEF"

                        "DeepSqueeze"
                        self_update.append([torch.zeros_like(model.get_weights()).to(device) for i in
                                            range(len(Transfer.neighbors[n]))])
                        neighbor_update.append([torch.zeros_like(model.get_weights()).to(device) for i in
                                                range(len(Transfer.neighbors[n]))])

                        "Compression initialization"
                        normalization = 1
                        if COMPRESSION == 'quantization':
                            if ADAPTIVE:
                                model_size = len(client_weights[n])
                                normalization = model_size
                            if FIRST is True:
                                compressors.append(
                                    Quantization_I(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value,
                                                   device=device))
                            else:
                                compressors.append(
                                    Quantization_U_1(num_bits=QUANTIZE_LEVEL, max_value=max_value, min_value=min_value,
                                                     device=device))  # Unbiased
                        elif COMPRESSION == 'topk':
                            compressors.append(Top_k(ratio=RATIO, device=device))
                        elif COMPRESSION == 'randk':
                            compressors.append(Rand_k(ratio=RATIO, device=device))
                        else:
                            raise Exception('Unknown compression method, please write the compression method first')

                    neighbors = list(Transfer.neighbors)
                    Algorithm = Algorithms(algorithm=ALGORITHM, compression=COMPRESSION, num_nodes=CLIENTS,
                                           neighbors=neighbors, models=models, data_transform=data_transform,
                                           device=device, self_weights=self_weights, neighbor_weights=neighbor_weights,
                                           data_loader=data_loader, learning_rate=Learning_rates[lr],
                                           compressors=compressors, gamma=Gamma[cons], residual_errors=residual_errors,
                                           self_accumulate_update=self_accumulate_update,
                                           neighbor_accumulate_update=neighbor_accumulate_update,
                                           self_H=self_H, neighbor_H=neighbor_H, self_G=self_G, neighbor_G=neighbor_G,
                                           lamda=beta, self_update=self_update,
                                           neighbor_update=neighbor_update, average_rate=beta,
                                           normalization=normalization)

                    global_loss = []
                    Test_acc = []
                    iter_num = 0
                    # print(ALGORITHM, DISCOUNT, BETA, FIRST)

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
                            test_weights = average_weights([Algorithm.models[i].get_weights() for i in
                                                            range(CLIENTS)])  # test with global averaged model
                        elif TEST == 'local':
                            test_weights = Algorithm.models[1].get_weights()  # test with local model

                        train_loss, train_acc = test_model.accuracy(weights=test_weights, test_loader=train_loader,
                                                                    device=device)
                        test_loss, test_acc = test_model.accuracy(weights=test_weights, test_loader=test_loader,
                                                                  device=device)

                        # if math.isnan(train_loss):  # Skip the training if the loss becomes NaN
                        #     print("NaN loss detected, stopping training.")
                        #     break

                        global_loss.append(train_loss)
                        Test_acc.append(test_acc)
                        print('SEED |', seed, '| iteration |', iter_num, '| Global Loss', round(train_loss, 6),
                              '| Training Accuracy |',
                              round(train_acc, 4), '| Test Accuracy |', round(test_acc, 4), '\n')

                        if iter_num >= AGGREGATION:
                            ACC += Test_acc
                            LOSS += global_loss
                            # print([compressors[i].discount_parameter for i in range(CLIENTS)])
                            break
                    del models
                    del self_weights

                    torch.cuda.empty_cache()  # Clean the memory cache

                folder = './{}_{}_{}_{}_{}_{}'.format(ALGORITHM, dataset, NETWORK, CLIENTS, NEIGHBORS, ALPHA)
                os.makedirs(folder, exist_ok=True)
                if len(LOSS) == 0 or len(ACC) == 0:
                    print("[WARNING] There is no useful data for this setup")
                else:
                    txt_list_lr = [LOSS, '\n', ACC]

                    # if COMPRESSION == 'topk':
                    #     np.savetxt('./{}/{}_{}_{}_{}_{}_{}.txt'.format(folder, ALGORITHM, COMPRESSION, RATIO, beta, Gamma[cons], Learning_rates[lr]), txt_list_lr, fmt='%s')
                    # elif COMPRESSION == 'quantization':
                    #     np.savetxt('./{}/{}_{}_{}_{}_{}_{}.txt'.format(folder, ALGORITHM, COMPRESSION, QUANTIZE_LEVEL, beta, Gamma[cons], Learning_rates[lr]), txt_list_lr, fmt='%s')

                    with open('./{}/{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(folder, ALGORITHM, dataset, COMPRESSION, RATIO,
                                                                        beta, Gamma[cons], Learning_rates[lr], ALPHA),
                              "w") as f:
                        for item in txt_list_lr:
                            if isinstance(item, str):
                                # write formatting directly (e.g., '\n')
                                f.write(item)
                            elif isinstance(item, (list, tuple, np.ndarray)):
                                # write vector
                                f.write(" ".join(map(str, item)) + "\n")
                            else:
                                # fallback for scalars
                                f.write(str(item) + "\n")

                    lr_loss.append(sum(LOSS[-5:]) / len(LOSS[-5:]))
                    print(Learning_rates[lr], lr_loss)

            "Missing the learning rate selection here, need to rewrite, start from here"
            if len(lr_loss) == 0:
                print("[WARNING] lr_loss is empty, skipping LR selection")
                continue  # or break, or set default
            else:
                best_index_lr = lr_loss.index(min(lr_loss))
                gamma_lr.append(Learning_rates[best_index_lr])
                gamma_loss.append(lr_loss[best_index_lr])

            # print(ALGORITHM, beta,  Gamma[:cons+1], gamma_lr, gamma_loss)

        if len(gamma_loss) == 0:
            print("[WARNING] gamma_loss is empty, skipping LR selection")
            continue  # or break, or set default
        else:
            best_index_gamma = gamma_loss.index(min(gamma_loss))
            # print(Gamma[cons], best_index_gamma)
            best_gamma = Gamma[best_index_gamma]
            best_lr = gamma_lr[best_index_gamma]

            beta_loss.append(gamma_loss[best_index_gamma])
            beta_lr.append(best_lr)
            beta_cons.append(best_gamma)

    best_lr = 0
    best_beta = 0
    best_cons = 0

    if len(beta_loss) == 0:
        print("[WARNING] beta_loss is empty, skipping LR selection")
    else:
        best_index_beta = beta_loss.index(min(beta_loss))
        # print(beta, best_index_beta)

        best_beta = BETAS[best_index_beta]
        best_lr = beta_lr[best_index_beta]
        best_cons = beta_cons[best_index_beta]
        print(best_beta, best_lr, best_cons, beta_loss)
        print(ALGORITHM, 'Best pair of parameters: learning rate = {}, gamma = {}, beta = {}'.format(best_lr, best_cons,
                                                                                                     best_beta))

    time_consumed = time.time() - start_time

    if STORE == 1:
        txt_list = [ALGORITHM, 'loss_list: ', beta_loss, '\n', 'best beta: ', best_beta, '\n', 'best lr: ', beta_lr,
                    '\n', 'best gamma: ', beta_cons, '\n', 'time consumed: ', time_consumed]
        if COMPRESSION == 'quantization':
            f = open('{}|{}|{}|{}|{}|{}|{}|{}|{}|final.txt'.format(ALGORITHM, dataset, QUANTIZE_LEVEL, ALPHA, NETWORK,
                                                                   CLIENTS, NEIGHBORS, date.today(),
                                                                   time.strftime("%H:%M:%S", time.localtime())), 'w')
        elif COMPRESSION == 'topk':
            f = open('{}|{}|{}|{}|{}|{}|{}|{}|{}|final.txt'.format(ALGORITHM, dataset, RATIO, ALPHA, NETWORK, CLIENTS,
                                                                   NEIGHBORS, date.today(),
                                                                   time.strftime("%H:%M:%S", time.localtime())), 'w')
        else:
            raise Exception('Unknown compression method')

        for item in txt_list:
            f.write("%s\n" % item)
    else:
        print('NOT STORE THE RESULTS THIS TIME')
