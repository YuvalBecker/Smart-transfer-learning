from torchvision import models
import numpy as np
from torchvision import transforms, datasets
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from trainning_net import  Simple_Net
from distribution_net import CustomRequireGrad
batch_size = 20
num_b = 2
amount_data = batch_size*num_b
use_rg = True
backbone_no_Grad = False

writer = SummaryWriter('./runds/Fmnsit/usee1_grads1DD12113'
                       + str(use_rg) + '_'
                       + str(amount_data)+'_samples')

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

    dataset_first = datasets.CIFAR10(root='.', train=True, download=True,
                                     transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset_first, batch_size=batch_size,
                                              shuffle=True)

    transform_rgb = transforms.Lambda(lambda image: image.convert('RGB'))

    #transform2 = transforms.Compose([transform_rgb, transforms.Resize((64, 64)),
    #                                transforms.ToTensor(),transforms.Normalize(
    #        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #dataset_train = datasets.FashionMNIST(root='.', train=True, download=True,
    #                                      transform=transform2)
    #transform2 = transforms.Compose([transform_rgb, transforms.Resize((64, 64)),
         #                          transforms.ToTensor(),transforms.Normalize(
        #   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10(root='.', train=True, download=True
                                     , transform=transform )

    dataloader2 = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                              shuffle=False)

    #network = models.vgg19(pretrained=True).cuda()
    network =Simple_Net().cuda()
    PATH = r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\model39'
    network.load_state_dict(torch.load(PATH), strict=False)

    #num_ftrs = network.classifier[6].in_features
    #network.classifier[6] = nn.Linear(num_ftrs, 10).cuda()
    if use_rg:
        rg = CustomRequireGrad(network, dataloader, dataloader2,
                               dist_processing_method='fft', batches_num=num_b*3,
                               percent=80, deepest_layer=7,
                               save_folder=r'C:\\Users\\yuval\\PycharmProjects\\smart_pretrained\\Statistics-pretrained\\save_stats')
        rg.run()

    if backbone_no_Grad:
        for param in network.features.parameters():
            param.requires_grad = False
    #network.classifier[6] = nn.Linear(num_ftrs, 10).cuda()

    #transform_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
#
    #transform_train = transforms.Compose([transform_rgb, transforms.Resize((64, 64)),
    #                                transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),
    #                                                     (0.5, 0.5, 0.5))])
    #model = network
    #testset = datasets.FashionMNIST(root='./data', train=False,
    #                                       download=True, transform=transform_train)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=0)
    #dataset_train = datasets.FashionMNIST(root='.',train=True,
    #                                 download=True,transform=transform_train)
    #trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                            #                 shuffle=True)
    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    dataset_train = datasets.CIFAR10(root='.',train=True,
                                     download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                             shuffle=False)
    #testset = datasets.CIFAR100(root='./data', train=False,
    #                                       download=True, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=2)
    #dataset_train = datasets.CIFAR100(root='.',train=True,
    #                                 download=False,transform=transform)
    #trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
    #                                         shuffle=False)

    net = network
    net.load_state_dict(torch.load(PATH), strict=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=9e-2, momentum=0.9)
    cycle_opt = torch.optim.lr_scheduler.CyclicLR(optimizer, 9e-2 , 5e-1,
                                      step_size_up=100)

    accuracy = []
    loss = []
    for epoch in range(40):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i >= amount_data/batch_size:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #plt.imshow(inputs.detach().cpu().numpy()[0].transpose())
            optimizer.zero_grad()
            outputs = net(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            if use_rg:
                if epoch < 35:
                    rg.update_grads(net)
                #if epoch > 25:
                  #  optimizer.param_groups[0]['lr'] = 1e-5
            optimizer.step()
            cycle_opt.step()
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
        #writer.add_scalar('Test - Accuracy',
        #                  (100 * correct / total),
        #                   epoch)
        #writer.add_scalar('Train - Loss',
        #                  (running_loss / i),
        #                   epoch)
        #writer.add_scalar('LR',
        #                  (optimizer.param_groups[0]['lr']),
        #                   epoch)
#
        accuracy.append((epoch,100 * correct / total ))
        running_loss = 0.0

    print('Finished Training')

    plt.plot(np.array(accuracy)[:,0], np.array(accuracy)[:, 1], '-o')

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()


