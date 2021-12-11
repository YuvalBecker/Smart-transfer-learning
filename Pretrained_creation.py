import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn.functional as F

from torchvision import models

import torch
transform = transforms.Compose([transforms.Resize((64, 64)),
                                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                            (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((64, 64)),
                                transforms.Normalize((0.1307,), (0.3081,))])
def replicate_channels(im):
    return torch.stack([im, im, im]).squeeze()

transformMnist= transforms.Compose([transforms.ToTensor(),transforms.Resize((64, 64)),replicate_channels,
                                transforms.Normalize((0.1307,), (0.3081,))])

class Simple_Net(nn.Module):
    def __init__(self):
        super(Simple_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 20, 3)
        self.fc1 = nn.Linear(20 * 5 * 5, 5)

    def forward(self, x):
        x = self.pool(self.batch_norm1(F.relu(self.conv1(x))))
        x = self.pool(self.batch_norm2(F.relu(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 20 * 5 * 5)
        x = F.softmax(self.fc1(x))
        return x

class Large_Simple_Net(nn.Module):
    def __init__(self):
        super(Large_Simple_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.conv5 = nn.Conv2d(32, 20, 3)
        self.fc1 = nn.Linear(20 * 8 * 8, 5)

    def forward(self, x):
        x = self.pool(self.batch_norm1(F.relu(self.conv1(x))))
        x = self.batch_norm2(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 20 * 8 * 8)
        #x = torch.nn.Flatten(x)
        x = F.softmax((self.fc1(x)))
        return x

class diff_net(Simple_Net):
    def __init__(self):
        super(diff_net, self).__init__()
        self.group_n_0 = torch.nn.GroupNorm(1, 3, eps=1e-05, affine=True)
        self.group_n_1 = torch.nn.GroupNorm(4, 100, eps=1e-05, affine=True)
        self.group_n_1 = torch.nn.GroupNorm(4, 100, eps=1e-05, affine=True)
        self.group_n_2 = torch.nn.GroupNorm(4, 100, eps=1e-05, affine=True)

    def forward(self, x):

        x = self.pool(self.batch_norm1(F.relu(self.conv1(self.group_n_0(x)))))
        x = self.pool(self.batch_norm2(F.relu(self.conv2(self.group_n_1(x)))))
        x = F.relu(self.conv3(self.group_n_2(x)))
        x = x.view(-1, 100 * 11 * 11)
        x = F.sigmoid(self.fc1(x))
        return x


if __name__ == '__main__':
    batch_size  = 16
    #dataset_first = datasets.CIFAR10(root='.', train=True, download=True,
     #                                transform=transform)
    #dataset_first = datasets.CIFAR10(root='.', train=True, download=True,
    #                                 transform=transformMnist)
#
    #trainloader = torch.utils.data.DataLoader(dataset_first, batch_size=batch_size,
    #                                          shuffle=True)
    #dataset_test = datasets.CIFAR10(root='.', train=False, download=True,
     #                                transform=transform)

    #testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
     #                                     shuffle=False)

    dataset_first = datasets.KMNIST(root='.', train=True, download=True,transform=transformMnist)
    trainloader = torch.utils.data.DataLoader(dataset_first, batch_size=batch_size, shuffle=True)

    dataset_test = datasets.KMNIST(root='.', train=False, download=True,transform=transformMnist)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                              shuffle=False)


    #network = Simple_Net().cuda()
    #network = Large_Simple_Net().cuda()
    network = Simple_Net().cuda()

#network = models.vgg19(pretrained=True).cuda()
    #network = models.densenet121(pretrained=True).cuda()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(network.parameters(), lr=5e-3, momentum=0.9)
    optimizer = optim.Adam(network.parameters(), lr=5e-4)
    PATH = r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\simple\_net_new_MNIST4'
    network.load_state_dict(torch.load(PATH), strict=True)

#cycle_opt = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-3, 5e-3,
    #                                  step_size_up=100)

    accuracy = []
    loss = []
    for epoch in range(80):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            optimizer.zero_grad()
            inputs, labels = data
            ind_to_train = np.where(labels >= 5 )
            if len(ind_to_train[0]) == 0:
                continue
            labels = labels[ind_to_train[0]] - 5
            inputs = inputs[ind_to_train[0]]
            #plt.imshow(inputs.detach().cpu().numpy()[0].transpose())
            outputs = network(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
                #if epoch > 25:
                  #  optimizer.param_groups[0]['lr'] = 1e-5
            optimizer.step()
            #cycle_opt.step()
            running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / i))
        correct = 0
        total = 0
        count = 0
        with torch.no_grad():
            for data in testloader:
                count += 1
                images, labels = data
                ind_to_train = np.where(labels >= 5)
                if len(ind_to_train[0]) == 0:
                    continue
                labels = labels[ind_to_train[0]] - 5
                images = images[ind_to_train[0]]

                outputs = network(images.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += np.array((predicted.detach().cpu() == labels)).sum()
                if count > 250:
                    break
        torch.save(network.state_dict(),r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\simple\_KMNIST'+str(epoch))

        print(
            'Accuracy of the network on the 10000 test images: %f %%' % (
                    100 * correct / total))
        accuracy.append((epoch,100 * correct / total ))
        running_loss = 0.0