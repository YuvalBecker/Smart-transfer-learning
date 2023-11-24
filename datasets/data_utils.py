import torch
import numpy as np
from torchvision import transforms, datasets
from torchvision.datasets import KMNIST, MNIST
#CIFAR10 = datasets.CIFAR10(root='.', train=True, download=True)

from PIL import Image
class cifar_part(datasets.cifar.CIFAR10):
    def __init__(self, transform, train=True,middle_range=5 ,upper= True):
        super(cifar_part, self).__init__(root='.', train=train)
        self.transform = transform
        if upper:
            indexes_dataset = np.where(np.array(self.targets) >= middle_range)[0]
            self.targets = [self.targets[ind]- middle_range for ind in indexes_dataset]
        else:
            indexes_dataset = np.where(np.array(self.targets) < middle_range)[0]
            self.targets = [self.targets[ind] for ind in indexes_dataset]
        self.data= self.data[indexes_dataset]
# size data cifar10 (25000, 32, 32, 3)
class kmnist_part(KMNIST):
    def __init__(self, transform, train=True,middle_range=5 ,upper= True ,shuffle=False):
        super(kmnist_part, self).__init__(root='.', train=train,download=True)
        self.transform = transform

        if upper: # Take upper classes 5 to 10
            indexes_dataset = np.where(np.array(self.targets) >= middle_range)[0]
            if shuffle:
                rand_indexes = np.random.permutation(len(indexes_dataset))
                self.targets = [self.targets[ind].numpy() - middle_range for ind in indexes_dataset[rand_indexes]]
            else:
                self.targets = [self.targets[ind].numpy() - middle_range for ind in indexes_dataset]

        else:
            indexes_dataset = np.where(np.array(self.targets) < middle_range)[0]
            if shuffle:
                rand_indexes = np.random.permutation(len(indexes_dataset))
                self.targets = [self.targets[ind].numpy() for ind in  indexes_dataset[rand_indexes]]
            else:
                self.targets = [self.targets[ind].numpy() for ind in  indexes_dataset]

        self.data= self.data[indexes_dataset]
        temp_data = torch.unsqueeze(self.data, dim=3)
        self.data = torch.repeat_interleave(temp_data,3,dim=3).numpy()
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class mnist_part(MNIST):
    def __init__(self, transform, train=True,middle_range=5 ,upper= True):
        super(mnist_part, self).__init__(root='.', train=train,download=True)
        self.transform = transform
        if upper:
            indexes_dataset = np.where(np.array(self.targets) >= middle_range)[0]
            self.targets = [self.targets[ind].numpy()- middle_range for ind in indexes_dataset]
        else:
            indexes_dataset = np.where(np.array(self.targets) < middle_range)[0]
            self.targets = [self.targets[ind].numpy() for ind in indexes_dataset]

        self.data= self.data[indexes_dataset]
        temp_data = torch.unsqueeze(self.data, dim=3)
        self.data = torch.repeat_interleave(temp_data,3,dim=3).numpy()
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class Fmnist_part(MNIST):
    def __init__(self, transform, train=True,middle_range=5 ,upper= True):
        super(Fmnist_part, self).__init__(root='.', train=train,download=True)
        self.transform = transform
        if upper:
            indexes_dataset = np.where(np.array(self.targets) >= middle_range)[0]
            self.targets = [self.targets[ind].numpy()- middle_range for ind in indexes_dataset]
        else:
            indexes_dataset = np.where(np.array(self.targets) < middle_range)[0]
            self.targets = [self.targets[ind].numpy() for ind in indexes_dataset]

        self.data= self.data[indexes_dataset]
        temp_data = torch.unsqueeze(self.data, dim=3)
        self.data = torch.repeat_interleave(temp_data,3,dim=3).numpy()
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
