import matplotlib.pyplot as plot
import numpy as np
import glob
import pandas as pd
import time

plot.rcParams['pdf.fonttype'] = 42
plot.rcParams['ps.fonttype'] = 42
plot.rcParams['font.family'] = 'arial'

def smooth(input):

    x1 = (input[0] + input[1] + input[2]) / 3
    x2 = (input[0] + input[1] + input[3]) / 3
    x3 = (input[0] + input[2] + input[3]) / 3
    x4 = (input[1] + input[2] + input[3]) / 3
    output = np.array([x1, x3, x2, x4])

    return output

def moving_average(input_data, window_size):
    moving_average = [[] for i in range(len(input_data))]
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if j < window_size - 1:
                if type(input_data[i][j + 1]) == str:
                    input_data[i][j + 1] = float(input_data[i][j + 1])
                # print(i, j, input_data[i][:j + 1])
                moving_average[i].append(sum(input_data[i][:j + 1]) / len(input_data[i][:j + 1]))
            else:
                # print(input_data[i][j - window_size + 1:j + 1])
                moving_average[i].append(sum(input_data[i][j - window_size + 1:j + 1]) / len(input_data[i][j - window_size + 1:j + 1]))
    moving_average_means = []
    for i in range(len(moving_average[0])):
        sum_data = []
        for j in range(len(moving_average)):
            sum_data.append(moving_average[j][i])
        moving_average_means.append(sum(sum_data) / len(sum_data))
    # print(len(moving_average_means))
    return np.array(moving_average), moving_average_means


"----------------------------------------------------NEW DATA DISTRIBUTION-------------------------------------------------------------------"
"Ring network 10 nodes alpha = 0.05"

# plot_list = ['DEFEAT|0.05|topk|0.05|0.25|0.056|10|2|.txt', 'DCD|0.05|topk|0.05|0.031610|2|.txt',
#              'CHOCO|0.05|topk|0.05|0.2|0.056|10|2|.txt', 'DeepSqueeze|0.05|topk|0.05|0.3|0.0316|0.4|10|2|.txt', 'BEER|0.05|topk|0.05|0.25|0.0316|10|2|.txt',
#              'MoTEF|0.05|topk|0.05|0.3|0.01|0.1|10|2|.txt', 'DEFEAT_C|0.05|topk|0.05|0.056|10|2|.txt']  # 'DEFEAT_C|0.05|topk|0.05|0.056|10|2|.txt'

"Ring network 10 nodes alpha = 5.0"  # MoTEF could be better?

# plot_list = ['DEFEAT|5.0|topk|0.05|0.25|0.1|10|2|.txt', 'DCD|5.0|topk|0.05|0.056|10|2|.txt',
#              'CHOCO|5.0|topk|0.05|0.1|0.1|10|2|.txt', 'DeepSqueeze|topk|0.05|0.25|0.1|0.25|10|2|.txt', 'BEER|5.0|topk|0.05|0.25|0.1|10|2|.txt',
#              'MoTEF|5.0|topk|0.05|0.4|0.1|0.1|10|2|.txt', 'DEFEAT_C|5.0|topk|0.05|0.1|10|2|.txt']  # 'DEFEAT_C|5.0|topk|0.05|0.1|10|2|.txt'
"--------------------------------------------------------------------------------------------------------------------------------------------"
"Ring network 20 nodes alpha = 0.05"  # DEFEAT could be better?

# plot_list = ['DEFEAT|0.05|topk|0.05|0.25|0.056|20|2|.txt', 'DCD|0.05|topk|0.05|0.25|0.056|20|2|.txt',
#              'DeepSqueeze|0.05|topk|0.05|0.35|0.056|0.3|20|2|.txt', 'CHOCO|0.05|topk|0.05|0.25|0.056|20|2|.txt',
#              'BEER|0.05|topk|0.05|0.25|0.056|20|2|.txt', 'MoTEF|0.05|topk|0.05|0.4|0.01|0.08|20|2|.txt',
#              'DEFEAT_C|0.05|topk|0.05|0.0316|20|2|.txt']

"Ring network 20 nodes alpha = 5.0"  # CHOCO diverges, need to make gamma smaller

# plot_list = ['DEFEAT|5.0|topk|0.05|0.35|0.1|20|2|.txt', 'DCD|5.0|topk|0.05|0.25|0.1|0.9|20|2|.txt',
#              'DeepSqueeze|5.0|topk|0.05|0.3|0.1|0.4|20|2|.txt', 'CHOCO|5.0|topk|0.05|0.2|0.1|20|2|.txt',
#              'BEER|5.0|topk|0.05|0.25|0.1|20|2|.txt', 'MoTEF|5.0|topk|0.05|0.3|0.1|0.1|20|2|.txt', 'DEFEAT_C|5.0|topk|0.05|0.056|20|2|.txt']

"--------------------------------------------------------------------------------------------------------------------------------------------"
"Ring network 40 nodes alpha = 0.05"
# plot_list = ['DEFEAT|0.05|topk|0.05|0.3|0.1|40|2|.txt', 'DEFEAT_C|0.05|topk|0.05|0.1|40|2|.txt',
#              'DCD|0.05|topk|0.05|0.0316|40|2|.txt', 'CHOCO|0.05|topk|0.05|0.25|0.056|40|2|.txt',
#              'DeepSqueeze|0.05|topk|0.05|0.3|0.1|0.4|40|2|.txt',
#              'BEER|0.05|topk|0.05|0.25|0.056|40|2|.txt', 'MoTEF|0.05|topk|0.05|0.4|0.1|0.1|40|2|.txt']

"Ring network 40 nodes alpha = 5.0"
# plot_list = ['DEFEAT|5.0|topk|0.05|0.35|0.1|40|2|.txt', 'DEFEAT_C|5.0|topk|0.05|0.1|40|2|.txt',
#              'DCD|5.0|topk|0.05|0.056|40|2|.txt', 'CHOCO|5.0|topk|0.05|0.25|0.1|40|2|.txt',
#              'DeepSqueeze|5.0|topk|0.05|0.25|0.1|0.4|40|2|.txt',
#              'BEER|5.0|topk|0.05|0.25|0.1|40|2|.txt', 'MoTEF|5.0|topk|0.05|0.4|0.1|0.1|40|2|.txt']
"--------------------------------------------------------------------------------------------------------------------------------------------"
"Compare DEFEAT / DEFEAT_C across different network topologies"
# plot_list = ['DEFEAT|0.05|topk|0.05|0.25|0.056|10|2|.txt', 'DEFEAT|0.05|topk|0.05|0.25|0.056|20|2|.txt', 'DEFEAT|0.05|topk|0.05|0.3|0.1|40|2|.txt']
#
# plot_list = ['DEFEAT_C|0.05|topk|0.05|0.056|10|2|.txt', 'DEFEAT_C|0.05|topk|0.05|0.0316|20|2|.txt', 'DEFEAT_C|0.05|topk|0.05|0.1|40|2|.txt']

# plot_list = ['DEFEAT|5.0|topk|0.05|0.25|0.1|10|2|.txt', 'DEFEAT|5.0|topk|0.05|0.35|0.1|20|2|.txt', 'DEFEAT|5.0|topk|0.05|0.35|0.1|40|2|.txt']

# plot_list = ['DEFEAT_C|5.0|topk|0.05|0.1|10|2|.txt', 'DEFEAT_C|5.0|topk|0.05|0.056|20|2|.txt', 'DEFEAT_C|5.0|topk|0.05|0.1|40|2|.txt']
"---------------------------------------------------------------------------------------------------------------------------------------------"
"Compare DEFEAT and DEFEAT_C"
plot_list = ['DEFEAT|0.05|topk|0.05|0.25|0.056|10|2|.txt', 'DEFEAT_C|0.05|topk|0.05|0.056|10|2|.txt',
             'DEFEAT_C|0.05|0.05|Ada|0.056|0.9|10|2|0.3_1.7|.txt', 'DEFEAT_C|0.05|0.05|Ada|0.056|0.9|10|2|0.3_1.0|.txt']
# plot_list = ['DEFEAT_C|0.05|0.05|Ada|0.056|0.9|10|2|0.3_1.7|.txt', 'DEFEAT_C|0.05|0.05|Ada|0.056|0.9|10|2|0.3_1.0|.txt']
# plot_list = ['DEFEAT|0.05|topk|0.05|0.25|0.056|20|2|.txt', 'DEFEAT_C|0.05|topk|0.05|0.0316|20|2|.txt']
# plot_list = ['DEFEAT|0.05|topk|0.05|0.3|0.1|40|2|.txt', 'DEFEAT_C|0.05|topk|0.05|0.1|40|2|.txt']
#
# plot_list = ['DEFEAT|5.0|topk|0.05|0.25|0.1|10|2|.txt', 'DEFEAT_C|5.0|topk|0.05|0.1|10|2|.txt']
# plot_list = ['DEFEAT|5.0|topk|0.05|0.35|0.1|20|2|.txt', 'DEFEAT_C|5.0|topk|0.05|0.056|20|2|.txt']
# plot_list = ['DEFEAT|5.0|topk|0.05|0.35|0.1|40|2|.txt', 'DEFEAT_C|5.0|topk|0.05|0.1|40|2|.txt']

# 11:00 mala2003

save = True
# save = False

color_list = ['blue', 'orange', 'green', 'red', 'gray', 'purple', 'brown', 'pink', 'yellow', 'cyan', 'olive', 'black']

# times = 4
times = 3

method = 'Top-k'

# compare = True
compare = False

# Ada_vs_fixed = True
Ada_vs_fixed = False
#
# Fair = 1  # Fair comparison
Fair = 0  # comparison in terms of number of aggregations

alpha1 = 0.05

# agg = 500
agg = 1000
# agg = 20000
# agg = 15000

iteration = range(agg)
if agg == 1000:
    dataset = "fashion"
else:
    dataset = "CIFAR"

missions = ['acc', 'loss']
if agg == 1000:
    window_size = 20
else:
    window_size = 250

# print(agg, dataset, window_size)
# plt.subplots(figsize=(10, 4))plt.plot(iteration, x_means, color=color_list[i], label='{}: dc={}'.format(name, dc))
fig, (ax1, ax2) = plot.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

for mission in missions:
    index = int(missions.index(mission)) + 1
    x_means_max = 0
    if mission == 'acc':
        plt = ax1
    else:
        plt = ax2
    for i in range(len(plot_list)):
        # print(plot_list[i])
        name = plot_list[i].split('|')[0]
        alpha = plot_list[i].split('|')[1]
        dc = plot_list[i].split('|')[-2]
        compression = plot_list[i].split('|')[2]
        num_nodes = plot_list[i].split('|')[-3]
        num_neighbors = plot_list[i].split('|')[-2]

        print(i, name)
        if compression == '4':
            method = 'quantization'
        elif compression == '6':
            method = 'quantization'
        elif compression == '8':
            method = 'quantization'
        elif compression == '0.05':
            method = 'Top-k'
        elif compression == '0.1':
            method = 'Top-k'
        elif compression == '0.2':
            method = 'Top-k'

        x = pd.read_csv(plot_list[i], header=None)

        x_acc, x_loss = x.values

        x_acc, x_loss = [x_acc[i * agg: (i + 1) * agg] for i in range(times)], [x_loss[i * agg: (i + 1) * agg] for i in range(times)]

        if mission == 'acc':
            x_area = np.stack(x_acc)
        elif mission == 'loss':
            x_area = np.stack(x_loss)

        x_area, x_means = moving_average(input_data=x_area, window_size=window_size)

        x_means = x_area.mean(axis=0)
        x_stds = x_area.std(axis=0, ddof=1)

        if name == 'CHOCO':
            name = 'CHOCO'
            if compare:
                plt.plot(iteration, x_means, color=color_list[i], label=r"{}: $\gamma'={}$".format(name, dc))  # dc is consensus in CHOCO case
            else:
                plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
        elif name == 'DEFEAT_C':
            name = 'DEFEAT_C'
            plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
        elif name == 'DeepSqueeze':
            name = 'DeepSqueeze'
            plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
        elif name == 'DCD':
            name = 'DCD'
            plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
        elif name == 'CEDAS':
            name = 'CEDAS'
            plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
            plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
        elif name == 'BEER':
            if Fair == 1:
                name = 'BEER (500)'
                y = int(len(x_means))
                x_means = x_means[:int(y/2)]
                x_stds = x_stds[:int(y/2)]
                x_b = np.arange(0, y, 2)
                plt.plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
            elif Fair == 0:
                name = 'BEER'
                plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
        elif name == 'DeCoM':
            if Fair == 1:
                name = 'DeCoM (500)'
                x_means = x_means[:int(agg/2)]
                x_stds = x_stds[:int(agg/2)]
                x_b = np.arange(0, agg, 2)
                plt.plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
            elif Fair == 0:
                name = 'DeCoM'
                plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
        elif name == 'MoTEF':
            if Fair == 1:
                name = 'MoTEF (500)'
                x_means = x_means[:int(agg/2)]
                x_stds = x_stds[:int(agg/2)]
                x_b = np.arange(0, agg, 2)
                plt.plot(x_b, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(x_b, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
            elif Fair == 0:
                name = 'MoTEF'
                plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
        elif name == 'DEFEAT':
            name = 'DEFEAT'
            if Ada_vs_fixed:
                if i == 0:
                    if compare:
                        plt.plot(iteration, x_means, color=color_list[i], label=r'{}: $\gamma$={}'.format(name, dc))
                    else:
                        plt.plot(iteration, x_means, color=color_list[i], label='{}: Fixed'.format(name))
                    plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
                if i == 1:
                    if compare:
                        plt.plot(iteration, x_means, color=color_list[i], label=r'{}: $\gamma$={}'.format(name, dc))
                    else:
                        plt.plot(iteration, x_means, color=color_list[i], label='{}: Adaptive'.format(name))
                    plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])
            else:
                if compare:
                    if i == 2:
                        compare = False
                        # name = 'DCD + Direct Error Feedback'
                    plt.plot(iteration, x_means, color=color_list[i], label=r'{}: $\gamma$={}'.format(name, dc))
                else:
                    if i == 2:
                        compare = False
                        # name = 'DCD + Direct Error Feedback'
                    plt.plot(iteration, x_means, color=color_list[i], label='{}'.format(name))
                plt.fill_between(iteration, x_means + x_stds, x_means - x_stds, alpha=alpha1, color=color_list[i])

        # plt.title('Top-k {} \u03B1 = {}'.format(compression, alpha))
    # if method == 'Top-k':
    #     plt.title('{} (k={}) \u03B1 = {}'.format(method, compression, alpha))
    # elif method == 'quantization':
    #     plt.title('{} ({}bits) \u03B1 = {}'.format(method, compression, alpha))

    plt.set_xlabel('Aggregations', fontsize=14)
    # plt.tick_params(axis='x', which='major', labelsize=14)
    # plt.tick_params(axis='y', which='major', labelsize=12)
    plt.set_title('{} (k={}) | \u03B1 = {} | Ring {}'.format(method, compression, alpha, num_nodes))
    if mission == 'acc':
        plt.set_ylabel('Test Accuracy', fontsize=14)
        plt.grid()
        if dataset == 'CIFAR':
            pass
            # plt.ylim([0.3, 0.72])
            # plt.ylim([0.4, 0.80])
        else:
            # pass
            if alpha == '0.05':
                plt.set_ylim([0.65, 0.825])
            elif alpha == '5.0':
                plt.set_ylim([0.75, 0.86])
            # plt.ylim([0.8, 0.95])  # MNIST
        # plt.legend(fontsize=14)
        # plt.grid()
        # if agg == 1000:
        #     plt.savefig('{}_{}_{}_{}_{}_{}.pdf'.format(mission, alpha, compression, dc, agg, time.strftime("%H:%M:%S", time.localtime())), bbox_inches='tight')
        #     pass
        # else:
        #     # plt.savefig('{}_{}_{}_{}_{}_{}.png'.format(mission, alpha, compression, dc, agg, time.strftime("%H:%M:%S", time.localtime())))
        #     # plt.savefig('{}_{}_{}_{}_{}_{}.pdf'.format(mission, alpha, compression, dc, agg, time.strftime("%H:%M:%S", time.localtime())), bbox_inches='tight')
        #     pass
        # plt.show()

    elif mission == 'loss':
        plt.set_ylabel('Global Loss', fontsize=14)
        plt.grid()
        if dataset == 'CIFAR':
            pass
            # plt.ylim([0.005, 0.016])
            # plt.ylim([0.001, 0.010])
        else:
            # pass
            # plt.ylim([0.8, 1.3])
            # plt.ylim([0.0045, 0.0065])
            if alpha == '0.05':
                plt.set_ylim([0.003, 0.008])  # FashionMNIST
            elif alpha == '5.0':
                plt.set_ylim([0.002, 0.006])
            # plt.ylim([0.004, 0.011])
            # plt.ylim([0.001, 0.006])  # MNIST
        # plt.legend(fontsize=14)
        # plt.grid()
        # if agg == 1000:
        #     fig = plt.gcf()
        #     fig.set_size_inches(6, 4)
        #     fig.savefig('{}_{}_{}_{}_{}_{}.pdf'.format(mission, alpha, compression, dc, agg, time.strftime("%H:%M:%S", time.localtime())), bbox_inches='tight')
        # else:
        #     # plt.savefig('{}_{}_{}_{}_{}_{}.png'.format(mission, alpha, compression, dc, agg, time.strftime("%H:%M:%S", time.localtime())))
        #     # plt.savefig('{}_{}_{}_{}_{}_{}.pdf'.format(mission, alpha, compression, dc, agg, time.strftime("%H:%M:%S", time.localtime())), bbox_inches='tight')
        #     pass
        # plt.show()

plot.legend(fontsize=14)
if save:
    # plot.savefig('Ring|{}|{}|.pdf'.format(num_nodes, alpha))
    pass
else:
    pass
plot.show()
