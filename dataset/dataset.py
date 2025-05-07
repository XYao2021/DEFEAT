import os
import sys
from torchvision.datasets import FashionMNIST, MNIST, EMNIST, QMNIST, KMNIST, SVHN
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
# from pytorch_cinic.dataset import CINIC10
from PIL import Image
import torch
import torchvision

class FashionMNISTEnhanced(FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, device=None):
        super().__init__(root, train, transform, target_transform, download)
        self.data_transformed = []
        self.target_transformed = []

        for image in self.data:
            image = Image.fromarray(image.numpy(), mode='L')  # Mode L means (8-bit pixels, black and white)
            if self.transform is not None:
                image = self.transform(image)
            self.data_transformed.append(image)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.target_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.target_transformed = torch.stack(self.target_transformed)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.target_transformed = self.target_transformed.to(device)

    def __getitem__(self, index):
        image, label = self.data_transformed[index], self.target_transformed[index]
        return image, label

class EMNISTEnhanced(EMNIST):  # Need to reduce the number of inputs
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, device=None):
        super().__init__(root, train, transform, target_transform, download)
        self.data_transformed = []
        self.target_transformed = []

        for image in self.data:
            image = Image.fromarray(image.numpy(), mode='L')  # Mode L means (8-bit pixels, black and white)
            if self.transform is not None:
                image = self.transform(image)
            self.data_transformed.append(image)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.target_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.target_transformed = torch.stack(self.target_transformed)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.target_transformed = self.target_transformed.to(device)

    def __getitem__(self, index):
        image, label = self.data_transformed[index], self.target_transformed[index]
        return image, label

class QMNISTEnhanced(QMNIST):  # Need to reduce the number of inputs
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, device=None):
        super().__init__(root, train, transform, target_transform, download)
        self.data_transformed = []
        self.target_transformed = []

        for image in self.data:
            image = Image.fromarray(image.numpy(), mode='L')  # Mode L means (8-bit pixels, black and white)
            if self.transform is not None:
                image = self.transform(image)
            self.data_transformed.append(image)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.target_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.target_transformed = torch.stack(self.target_transformed)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.target_transformed = self.target_transformed.to(device)

    def __getitem__(self, index):
        image, label = self.data_transformed[index], self.target_transformed[index]
        return image, label

class KMNISTEnhanced(KMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, device=None):
        super().__init__(root, train, transform, target_transform, download)
        self.data_transformed = []
        self.target_transformed = []

        for image in self.data:
            image = Image.fromarray(image.numpy(), mode='L')  # Mode L means (8-bit pixels, black and white)
            if self.transform is not None:
                image = self.transform(image)
            self.data_transformed.append(image)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.target_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.target_transformed = torch.stack(self.target_transformed)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.target_transformed = self.target_transformed.to(device)

    def __getitem__(self, index):
        image, label = self.data_transformed[index], self.target_transformed[index]
        return image, label

class MNISTEnhanced(MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, device=None):
        super().__init__(root, train, transform, target_transform, download)
        self.data_transformed = []
        self.target_transformed = []

        for image in self.data:
            image = Image.fromarray(image.numpy(), mode='L')  # Mode L means (8-bit pixels, black and white)
            if self.transform is not None:
                image = self.transform(image)
            self.data_transformed.append(image)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.target_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.target_transformed = torch.stack(self.target_transformed)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.target_transformed = self.target_transformed.to(device)

    def __getitem__(self, index):
        image, label = self.data_transformed[index], self.target_transformed[index]
        return image, label

class SVHNEnhanced(SVHN):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, device=None):
        super().__init__(root, train, transform, target_transform, download)
        self.data_transformed = []
        self.target_transformed = []

        for image in self.data:
            image = Image.fromarray(image.numpy(), mode='L')  # Mode L means (8-bit pixels, black and white)
            if self.transform is not None:
                image = self.transform(image)
            self.data_transformed.append(image)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.target_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.target_transformed = torch.stack(self.target_transformed)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.target_transformed = self.target_transformed.to(device)

    def __getitem__(self, index):
        image, label = self.data_transformed[index], self.target_transformed[index]
        return image, label

class CIFAR10Enhanced(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=None, device=None):
        super().__init__(root, train, transform, target_transform, download)
        self.data_transformed = []
        self.target_transformed = []

        for image in self.data:
            image = Image.fromarray(image)
            if self.transform is not None:
                image = self.transform(image)
            self.data_transformed.append(image)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.target_transformed.append(torch.tensor(target))

        self.data_transformed = torch.stack(self.data_transformed)
        self.target_transformed = torch.stack(self.target_transformed)
        # self.target_transformed = torch.stack(self.target_transformed, dtype=torch.int64)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.target_transformed = self.target_transformed.to(device)

    def __getitem__(self, index):
        image, label = self.data_transformed[index], self.target_transformed[index]
        return image, label

def loading(dataset_name, data_path, device):
    # print(dataset_name, data_path)
    # FashionMNIST, MNIST, EMNIST, QMNIST, KMNIST
    if dataset_name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_data = FashionMNISTEnhanced(data_path, transform=transform, download=True, device=device)
        test_data = FashionMNISTEnhanced(data_path, train=False, transform=transform, device=device)

    elif dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_data = MNISTEnhanced(data_path, transform=transform, download=True, device=device)
        test_data = MNISTEnhanced(data_path, train=False, transform=transform, device=device)

    elif dataset_name == 'EMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_data = EMNISTEnhanced(data_path, transform=transform, download=True, device=device)
        test_data = EMNISTEnhanced(data_path, train=False, transform=transform, device=device)

    elif dataset_name == 'QMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_data = QMNISTEnhanced(data_path, transform=transform, download=True, device=device)
        test_data = QMNISTEnhanced(data_path, train=False, transform=transform, device=device)

    elif dataset_name == 'KMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_data = KMNISTEnhanced(data_path, transform=transform, download=True, device=device)
        test_data = KMNISTEnhanced(data_path, train=False, transform=transform, device=device)

    elif dataset_name == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])])
        # train_data = SVHNEnhanced(data_path, transform=transform, download=True, device=device)
        # test_data = SVHNEnhanced(data_path, train=False, transform=transform, device=device)
        train_data = torchvision.datasets.SVHN(root='./data', split='train', transform=transforms.ToTensor(), download=True)
        test_data = torchvision.datasets.SVHN('./data', split='test', transform=transform, download=True)
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_data = CIFAR10Enhanced(data_path, transform=transform, download=True, device=device)
        test_data = CIFAR10Enhanced(data_path, transform=transform, train=False, device=device)
    else:
        raise Exception('Unknown dataset')

    return train_data, test_data

# def loading_CINIC(data_path, device):
#     cinic_mean = [0.47889522, 0.47227842, 0.43047404]
#     cinic_std = [0.24205776, 0.23828046, 0.25874835]
#     train_data = torchvision.datasets.ImageFolder(dataset_path + '/train',
#                                                   transform=transforms.Compose([transforms.ToTensor(),
#                                                                                 transforms.Normalize(mean=cinic_mean,
#                                                                                                      std=cinic_std)]))
#     test_data = torchvision.datasets.ImageFolder(dataset_path + '/test',
#                                                  transform=transforms.Compose([transforms.ToTensor(),
#                                                                                transforms.Normalize(mean=cinic_mean,
#                                                                                                     std=cinic_std)]))
#     return train_data, test_data
