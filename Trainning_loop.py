from torchvision import models
import numpy as np
from torchvision import transforms, datasets
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from distribution_net import CustomRequireGrad

amount_data = 30
use_rg = True


writer = SummaryWriter('./runds/change_grads_HHyp1er_diff111'
                       + str(use_rg) + '_smaller_data_more_'
                       + str(amount_data)+'_samp')

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    dataset_first = datasets.CIFAR10(root='.', train=True, download=True,
                                     transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset_first, batch_size=4,
                                              shuffle=False)

    transform_rgb = transforms.Lambda(lambda image: image.convert('RGB'))

    transform = transforms.Compose([transform_rgb, transforms.Resize((64, 64)),
                                    transforms.ToTensor(),transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.FashionMNIST(root='.', train=True, download=True,
                                          transform=transform)
    dataloader2 = torch.utils.data.DataLoader(dataset_train, batch_size=4,
                                              shuffle=False)

    network = models.vgg19(pretrained=False).cuda()
    num_ftrs = network.classifier[6].in_features
    network.classifier[6] = nn.Linear(num_ftrs, 10).cuda()
    vgg_cifar = torch.load('./model_cifar10')
    network.load_state_dict(vgg_cifar)
    # Adjusting output to 10 classes
    if use_rg:
        rg = CustomRequireGrad(network, dataloader, dataloader2)
        rg.run()

    transform_rgb = transforms.Lambda(lambda image: image.convert('RGB'))

    transform_train = transforms.Compose([transform_rgb, transforms.Resize((64, 64)),
                                    transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    model = network
    batch_size= 8
    testset = datasets.FashionMNIST(root='./data', train=False,
                                           download=True, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    dataset_train = datasets.FashionMNIST(root='.',train=True,
                                     download=False,transform=transform_train)
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                             shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = network
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    accuracy = []
    loss = []
    for epoch in range(40):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i >= amount_data/batch_size:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            if use_rg:
                if epoch < 20:
                    rg.update_grads(net)
                if epoch > 20:
                    optimizer.param_groups[0]['lr'] = 1e-4
            optimizer.step()
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

                outputs = net(images.cuda())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += np.array((predicted.detach().cpu() == labels)).sum()
                if count > 1000:
                    break
        print(
            'Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))
        writer.add_scalar('Test - Accuracy',
                          (100 * correct / total),
                           epoch)
        writer.add_scalar('Train - Loss',
                          (running_loss / i),
                           epoch)

        accuracy.append((epoch,100 * correct / total ))
        running_loss = 0.0

    print('Finished Training')

    plt.plot(np.array(accuracy)[:,0], np.array(accuracy)[:, 1], '-o')

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()


