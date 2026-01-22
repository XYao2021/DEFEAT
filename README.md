# DEFEAT Codes
The official Pytorch code for Adaptively Discounted Error Feedback Enhanced Algorithm for decenTralized learning (A-DEFEAT). These codes only use for experiments in paper "A-DEFEAT: an Adaptively Discounted Error Feedback Enhanced Algorithm for Decentralized Learning" ICML 2026.

# Python Libraries
torch, torchversion, numpy, matplotlib, time, sys, os, datetime, random, copy and any libraries needed in the codes.

# Configures
The configuration parameters can be found in "config.py".

# Datasets
Datasets include FashionMNIST, KMNIST, CIFAR10, CIFAR100. Mainly focus on FashionMNIST, KMNIST, CIFAR10 in the paper.

# Models
The models are custimized model every datasets. The models can be found in files under "model" folder, and can be modified if needed.

# Running
To run the experiments, using "python main.py" command in terminal to run the file called "main.py", or directly running the "main.py" file in python platform. The configurations can be changed by specifying the configuration parameters in the command line after "main.py", the details can be find in file "config.py". The other useful functions can be find in "util.py" file under "util" folder.

# Additional folder
Need to create a folder named "data" to store the datasets.
