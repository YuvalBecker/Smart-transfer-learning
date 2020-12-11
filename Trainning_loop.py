from torchvision import models
import numpy as np
from torchvision import transforms, datasets
import torch


import torch.optim as optim
import torch.nn as nn
from net_investigate import CustomRequireGrad

if __name__ == "__main__":
    image_net_path = r'/home/yuval/imageNet/'
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    data_imagenet = datasets.ImageFolder(image_net_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(data_imagenet, batch_size=9,
                                             shuffle=True)
    dataset_train = datasets.CIFAR10(root='.', train=True, download=True,
                                     transform=transform)
    dataloader2 = torch.utils.data.DataLoader(dataset_train, batch_size=8,
                                              shuffle=True)

    network = models.vgg19(pretrained=True).cuda()

    num_ftrs = network.classifier[6].in_features
    network.classifier[6] = nn.Linear(num_ftrs, 10).cuda()

    rg = CustomRequireGrad(network, dataloader, dataloader2)
    rg.run(0.7)
    # Adjusting output to 10 classes

    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    model = network
    batch_size=16
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                             shuffle=False, num_workers=2)
    dataset_train = datasets.CIFAR10(root='.',train=True,download=False,transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=16,
                                             shuffle=False)
    import matplotlib.pyplot as plt
    #plt.plot(rg.mean_var)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    import torch.nn.functional as F

    net = network
    criterion = nn.CrossEntropyLoss()
    # Changing specific layers learning rates
    optimizer = optim.SGD([  {'params': rg.layers_list_to_stay},
         {'params': rg.layers_list_to_change, 'lr': 1e-5},
     ], lr=1e-3, momentum=0.9)
    accuracy = []
    loss = []
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i >= 50e3/batch_size:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            #if i % 1200 == 0 :
                #rg.list_grads = [True] * len(rg.list_grads)
                #rg._update_require_grads_params()
                #print('Updated grads')

            # print statistics
            running_loss += loss.item()
            if i % 400 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 400))
                running_loss = 0.0
                correct = 0
                total = 0
                count = 0
                with torch.no_grad():
                    for data in testloader:
                        count +=1
                        images, labels = data
                        outputs = net(images.cuda())
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += np.array( (predicted.detach().cpu() == labels)).sum()
                        if count > 100:
                            break

                print(
                    'Accuracy of the network on the 10000 test images: %d %%' % (
                            100 * correct / total))
                accuracy.append((i,100 * correct / total ))

    print('Finished Training')
