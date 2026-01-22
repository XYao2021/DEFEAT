import matplotlib.pyplot as plot
import numpy as np
import glob
import pandas as pd
import os

# =========================
# Global plotting settings
# =========================
plot.rcParams['pdf.fonttype'] = 42
plot.rcParams['ps.fonttype'] = 42
plot.rcParams['font.family'] = 'arial'

# =========================
# Utility functions
# =========================
def moving_average(input_data, window_size):
    moving_average = [[] for _ in range(len(input_data))]
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if j < window_size - 1:
                moving_average[i].append(np.mean(input_data[i][:j + 1]))
            else:
                moving_average[i].append(
                    np.mean(input_data[i][j - window_size + 1:j + 1])
                )
    moving_average_means = np.mean(moving_average, axis=0)
    return np.array(moving_average), moving_average_means


# =========================
# Configuration
# =========================
# folder_paths = [
#     # "./Ring-10-FashionMNIST-0.05",
#     "./Ring-10-FashionMNIST-0.2",
#     # "./Ring-10-FashionMNIST-0.5",
#     # "./Ring-10-FashionMNIST-5.0",
#     # "./Ring-20-FashionMNIST-0.05",
#     # "./Ring-20-FashionMNIST-5.0",
#     # "./Ring-40-FashionMNIST-0.05",
#     # "./Ring-40-FashionMNIST-5.0",
#     # "./Ring-10-FashionMNIST-0.05-Ada_vs_Fix",
# ]

# folder_paths = [
#     # "./Ring-10-KMNIST-0.05",
#     "./Ring-10-KMNIST-0.2",
#     # "./Ring-20-KMNIST-0.05",
# ]
#
# window_size = 20
# agg = 1000
#
folder_paths = [
    # "./Ring-10-CIFAR10-0.2",
    "./Ring-10-CIFAR100-0.2",
]

window_size = 250
agg = 10000

name_list = ['DEFEAT_C', 'DCD', 'CHOCO', 'DeepSqueeze', 'BEER', 'MoTEF', 'DEFEAT']
color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
              'olive', 'cyan', 'magenta', 'teal', 'navy', 'gold', 'darkgreen', 'black']
alpha1 = 0.15
times = 1
iteration = np.arange(agg)

# save = False
save = True

normal_algos = {"DCD", "DEFEAT_C", "CHOCO", "DeepSqueeze"}
double_comm_algos = {"BEER", "MoTEF"}

fair = False
# fair = True


# ============================================================================
# PART A: ONE FIGURE PER FOLDER (Accuracy + Loss)
# ============================================================================
print("\nGenerating individual figures...\n")

for folder in folder_paths:
    if not os.path.exists(folder):
        print(f"[Skip] {folder} not found")
        continue

    parts = folder.split('-')

    num_nodes = parts[1]  # "10"
    dataset_name = parts[2]  # "KMNIST"
    alpha_val = parts[3]

    print(f"\n Processing {folder}", alpha_val, dataset_name)
    fig, axes = plot.subplots(1, 2, figsize=(13, 6))
    ax_acc, ax_loss = axes

    files = glob.glob(f"{folder}/*.txt")
    files = [f for name in name_list for f in files if os.path.basename(f).split('|')[0] == name]

    handles, labels = [], []

    for i, file in enumerate(files):
        algo = os.path.basename(file).split('|')[0]
        x = pd.read_csv(file, header=None)
        acc_raw, loss_raw = x.values

        # print(algo, len(acc_raw), len(loss_raw))

        acc_runs = [acc_raw[j * agg:(j + 1) * agg] for j in range(times)]
        loss_runs = [loss_raw[j * agg:(j + 1) * agg] for j in range(times)]

        acc_runs = np.stack(acc_runs)
        loss_runs = np.stack(loss_runs)

        acc_smoothed, _ = moving_average(acc_runs, window_size)
        loss_smoothed, _ = moving_average(loss_runs, window_size)

        acc_mean = acc_smoothed.mean(axis=0)
        acc_std = acc_smoothed.std(axis=0, ddof=1)

        loss_mean = loss_smoothed.mean(axis=0)
        loss_std = loss_smoothed.std(axis=0, ddof=1)

        # ------------------------------------------------------------
        # Print averaged accuracy values (debug / report)
        # ------------------------------------------------------------
        if algo in normal_algos:
            vals = acc_mean[-10:]
            var = acc_std[-10:]
            print(f"[{folder}] {algo} last 10 avg acc:", np.round(sum(vals)/len(vals), 4), "+-", np.round(sum(var)/len(var), 4))

        elif algo in double_comm_algos:
            mid = len(acc_mean) // 2
            vals = acc_mean[mid - 10: mid]
            var = acc_std[mid - 10: mid]
            print(f"[{folder}] {algo} middle 10 avg acc:", np.round(sum(vals)/len(vals), 4), "+-", np.round(sum(var)/len(var), 4))
            vals = acc_mean[-10:]
            var = acc_std[-10:]
            print(f"[{folder}] {algo} last 10 avg acc:", np.round(sum(vals) / len(vals), 4), "+-", np.round(sum(var)/len(var), 4))
            if fair:
                acc_mean = np.repeat(acc_mean[:mid], 2)
                acc_std = np.repeat(acc_std[:mid], 2)

                loss_mean = np.repeat(loss_mean[:mid], 2)
                loss_std = np.repeat(loss_std[:mid], 2)

        line, = ax_acc.plot(iteration, acc_mean,
                            color=color_list[i],
                            linewidth=2.5,
                            label=algo)
        ax_acc.fill_between(iteration,
                            acc_mean + acc_std,
                            acc_mean - acc_std,
                            alpha=alpha1,
                            color=color_list[i])

        ax_loss.plot(iteration, loss_mean,
                     color=color_list[i],
                     linewidth=2.5)
        ax_loss.fill_between(iteration,
                             loss_mean + loss_std,
                             loss_mean - loss_std,
                             alpha=alpha1,
                             color=color_list[i])

        handles.append(line)

        if algo == "DEFEAT_C":
            algo = "A-DEFEAT"
        elif algo == "DEFEAT":
            gamma = os.path.basename(file).split('|')[3]
            algo = r"Fixed $\gamma={}$".format(gamma)
        labels.append(algo)

    ax_acc.set_ylabel("Test Accuracy", fontsize=16, fontweight='bold')
    ax_loss.set_ylabel("Global Loss", fontsize=16, fontweight='bold')

    if fair:
        ax_acc.set_xlabel("Communication Rounds", fontsize=16, fontweight='bold')
        ax_loss.set_xlabel("Communication Rounds", fontsize=16, fontweight='bold')
    else:
        ax_acc.set_xlabel("Aggregations", fontsize=16, fontweight='bold')
        ax_loss.set_xlabel("Aggregations", fontsize=16, fontweight='bold')

    if dataset_name == "FashionMNIST":
        if alpha_val == "0.05":
            # Accuracy limits
            ax_acc.set_ylim(0.55, 0.83)  # adjust if needed
            # Loss limits
            ax_loss.set_ylim(0.3, 1.5)  # adjust if needed
        elif alpha_val == "0.5" or "0.2":
            # Accuracy limits
            ax_acc.set_ylim(0.7, 0.85)  # adjust if needed
            # Loss limits
            ax_loss.set_ylim(0.3, 1.0)  # adjust if needed
        elif alpha_val == "5.0":
            # Accuracy limits
            ax_acc.set_ylim(0.7, 0.86)  # adjust if needed
            # Loss limits
            ax_loss.set_ylim(0.3, 1.0)  # adjust if needed
    elif dataset_name == "KMNIST":
        ax_acc.set_ylim(0.7, 1.0)  # adjust if needed
        # Loss limits
        ax_loss.set_ylim(0.05, 1.4)  # adjust if needed
    elif dataset_name == "CIFAR10":
        ax_acc.set_ylim(0.5, 0.71)  # adjust if needed
        # Loss limits
        ax_loss.set_ylim(0.1, 1.4)  # adjust if needed
    elif dataset_name == "CIFAR100":
        ax_acc.set_ylim(0.0, 0.5)  # adjust if needed
        # Loss limits
        ax_loss.set_ylim(0.0, 1.3)  # adjust if needed

    ax_acc.grid(True, alpha=0.6)
    ax_loss.grid(True, alpha=0.6)

    # fig.suptitle(folder, fontsize=20, fontweight='bold')
    # fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=13)
    fig.suptitle(
        f'{dataset_name} | α={alpha_val} | {num_nodes} nodes',
        fontsize=16,
        fontweight='bold',
        y=0.935
    )

    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.915),  # lower than before
        ncol=len(labels),
        fontsize=14,  # slightly smaller
        frameon=True,
        fancybox=True,
        shadow=False,
        borderaxespad=0.3,
        columnspacing=1.2,
        handlelength=2.0
    )

    # fig.suptitle(
    #     f'{dataset_name} | α={alpha_val} | {num_nodes} nodes',
    #     fontsize=16,
    #     fontweight='bold',
    #     y=0.995
    # )
    #
    # max_cols = 6  # adjust: 3–5 usually works best for ICML figures
    #
    # fig.legend(
    #     handles,
    #     labels,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, 0.965),
    #     ncol=min(len(labels), max_cols),
    #     fontsize=14,
    #     frameon=True,
    #     fancybox=True,
    #     shadow=False,
    #     borderaxespad=0.3,
    #     columnspacing=1.2,
    #     handlelength=2.0
    # )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        if dataset_name == "CIFAR10":
            if fair:
                fig.savefig(f"{folder}_comm.png", dpi=300, bbox_inches='tight')
                print(f"Saved {folder}_comm.png")
            else:
                fig.savefig(f"{folder}_agg.png", dpi=300, bbox_inches='tight')
                print(f"Saved {folder}_agg.png")
        elif dataset_name == "CIFAR100":
            if fair:
                fig.savefig(f"{folder}_comm.png", dpi=300, bbox_inches='tight')
                print(f"Saved {folder}_comm.png")
            else:
                fig.savefig(f"{folder}_agg.png", dpi=300, bbox_inches='tight')
                print(f"Saved {folder}_agg.png")
        elif dataset_name == "FashionMNIST":
            if fair:
                fig.savefig(f"{folder}_comm.pdf", dpi=300, bbox_inches='tight')
                print(f"Saved {folder}_comm.pdf")
            else:
                fig.savefig(f"{folder}_agg.pdf", dpi=300, bbox_inches='tight')
                print(f"Saved {folder}_agg.pdf")
        elif dataset_name == "KMNIST":
            if fair:
                fig.savefig(f"{folder}_comm.pdf", dpi=300, bbox_inches='tight')
                print(f"Saved {folder}_comm.pdf")
            else:
                fig.savefig(f"{folder}_agg.pdf", dpi=300, bbox_inches='tight')
                print(f"Saved {folder}_agg.pdf")
        else:
            print("Unrecognized dataset")
    else:
        plot.show()
        pass

plot.close('all')


# # ============================================================================
# # PART B: ONE GLOBAL COMPARISON FIGURE
# # ============================================================================
# print("\nGenerating global comparison figure...\n")
#
# topology_map = {
#     "Ring-10": "./Ring-10-FashionMNIST",
#     "Ring-20": "./Ring-20-FashionMNIST",
#     "Ring-40": "./Ring-40-FashionMNIST"
# }
#
# algorithms = ['DEFEAT_C', 'MoTEF']
# algo_colors = {'DEFEAT_C': 'blue', 'MoTEF': 'orange'}
# linestyles = ['-', '--', ':']
# alphas = ['0.05', '5.0']
#
# fig, axes = plot.subplots(2, 2, figsize=(18, 14))
# handles, labels = [], []
#
# for col, alpha in enumerate(alphas):
#     for row, metric in enumerate(['acc', 'loss']):
#         ax = axes[row, col]
#
#         for algo in algorithms:
#             for topo_idx, (topo_name, base_path) in enumerate(topology_map.items()):
#                 folder = f"{base_path}-{alpha}"
#                 if not os.path.exists(folder):
#                     continue
#
#                 files = glob.glob(f"{folder}/*.txt")
#                 file = next((f for f in files if os.path.basename(f).split('|')[0] == algo), None)
#                 if file is None:
#                     continue
#
#                 x = pd.read_csv(file, header=None)
#                 acc_raw, loss_raw = x.values
#                 data = acc_raw if metric == 'acc' else loss_raw
#
#                 runs = [data[j * agg:(j + 1) * agg] for j in range(times)]
#                 runs = np.stack(runs)
#                 runs, _ = moving_average(runs, window_size)
#
#                 mean = runs.mean(axis=0)
#                 std = runs.std(axis=0, ddof=1)
#
#                 if algo in double_comm_algos:
#                     if fair:
#                         mid = len(mean) // 2
#                         mean = np.repeat(mean[:mid], 2)
#                         std = np.repeat(std[:mid], 2)
#
#                 label = f"{algo} ({topo_name})"
#                 line, = ax.plot(iteration, mean,
#                                 color=algo_colors[algo],
#                                 linestyle=linestyles[topo_idx],
#                                 linewidth=3)
#
#                 ax.fill_between(iteration,
#                                 mean + std,
#                                 mean - std,
#                                 alpha=0.15,
#                                 color=algo_colors[algo])
#
#                 if row == 0 and col == 0:
#                     handles.append(line)
#                     if algo == "DEFEAT_C":
#                         algo_display = "A-DEFEAT"
#                     else:
#                         algo_display = algo
#                     label = f"{algo_display} ({topo_name})"
#                     labels.append(label)
#
#         if metric == "acc":
#             # Accuracy limits
#             if alpha == "0.05":
#                 ax.set_ylim(0.55, 0.83)  # adjust if needed
#             else:
#                 ax.set_ylim(0.6, 0.86)  # adjust if needed
#         else:
#             # Loss limits
#             ax.set_ylim(0.3, 1.5)  # adjust if needed
#
#         ax.set_xlabel("Aggregations", fontsize=16, fontweight='bold')
#         ax.set_title(f"{'Accuracy' if metric == 'acc' else 'Loss'} (α={alpha})",
#                      fontsize=18, fontweight='bold')
#         ax.grid(True, alpha=0.6)
#         ax.tick_params(labelsize=14)
#
# fig.legend(
#         handles,
#         labels,
#         loc='upper center',
#         bbox_to_anchor=(0.5, 0.93),  # lower than before
#         ncol=len(labels),
#         fontsize=14,  # slightly smaller
#         frameon=True,
#         fancybox=True,
#         shadow=False,
#         borderaxespad=0.3,
#         columnspacing=1.2,
#         handlelength=2.0
#     )
#
# fig.suptitle("Global Comparison: DEFEAT_C vs DCD",
#              fontsize=16, fontweight='bold', y=0.95)
#
# fig.tight_layout(rect=[0, 0, 1, 0.93])
#
# if save:
#     fig.savefig("Global_Comparison.pdf", dpi=300, bbox_inches='tight')
#     print("Saved Global_Comparison.pdf")
# else:
#     plot.show()

print("\nAll figures generated successfully.")
