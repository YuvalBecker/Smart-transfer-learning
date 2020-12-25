from torchvision import models
import numpy as np
from torchvision import transforms, datasets
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn
from distribution_net import CustomRequireGrad

## This script was taken from pytorch tutrial ... only for example
## how to run use the class
if __name__ == "__main__":
    image_net_path = r'/home/yuval/imageNet/'
    transform = transforms.Compose([transforms.Resize((122, 122)),
                                    transforms.ToTensor()])
    data_imagenet = datasets.ImageFolder(image_net_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(data_imagenet, batch_size=4,
                                             shuffle=True)
    transform_rgb = transforms.Lambda(lambda image: image.convert('RGB'))

    transform = transforms.Compose([transform_rgb, transforms.Resize((122, 122)),
                                    transforms.ToTensor()])
    dataset_train = datasets.FashionMNIST(root='.', train=True, download=True,
                                     transform=transform)
    dataloader2 = torch.utils.data.DataLoader(dataset_train, batch_size=4,
                                              shuffle=True)

    network = models.vgg19(pretrained=True).cuda()
    num_ftrs = network.classifier[6].in_features
    # Adjusting output to 10 classes

    rg = CustomRequireGrad(network, dataloader, dataloader2)

    rg.run()
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    network.classifier[6] = nn.Linear(num_ftrs, 10).cuda()

    model = network
    batch_size= 16
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)
    dataset_train = datasets.CIFAR10(root='.',train=True,
                                     download=False,transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=16,
                                             shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = network
    criterion = nn.CrossEntropyLoss()
    # Changing specific layers learning rates

    optimizer = optim.SGD([{'params': rg.layers_list_to_stay, 'lr':1e-2},
                           {'params': rg.layers_list_to_change, 'lr': 1e-2}]
                          ,lr=1e-2, momentum=0.9)

    optimizer = optim.SGD(net.parameters(),lr=1e-3, momentum=0.9)
    accuracy = []
    loss = []
    for epoch in range(45):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i >= 20/batch_size:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            #if epoch < 8:
            #    rg.update_grads(net)
            optimizer.step()
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 400))
        running_loss = 0.0
        correct = 0
        total = 0
        count = 0
        with torch.no_grad():
            for data in testloader:
                count += 1
                images, labels = data
                outputs = net(images.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += np.array((predicted.detach().cpu() == labels)).sum()
                if count > 100:
                    break
        print(
            'Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))
        accuracy.append((epoch,100 * correct / total ))
    print('Finished Training')

    plt.plot( np.array(accuracy)[:,0], np.array(accuracy)[:,1] ,'-o')

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()


