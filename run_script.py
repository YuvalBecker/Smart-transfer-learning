import numpy as np
from torchvision import transforms, datasets
import torch
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
from trainning_net import  Simple_Net, diff_net
from distribution_net import CustomRequireGrad
import argparse

def main(args):
    torch.cuda.manual_seed_all(args.seed)
    batch_size = args.batch_size
    num_b = args.num_batch
    amount_data =num_b*batch_size
    ############# statistics for custom :
    if args.pre_dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                                    (0.5, 0.5, 0.5))])
        dataset_pre = datasets.CIFAR10(root='.', train=True, download=True,transform=transform)
        dataloader_pre = torch.utils.data.DataLoader(dataset_pre, batch_size=batch_size,shuffle=False)

    if args.pre_dataset == 'KMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_pre = datasets.KMNIST(root='.', train=True, download=True, transform=transform)
        dataloader_pre = torch.utils.data.DataLoader(dataset_pre, batch_size=batch_size,
                                                  shuffle=True)

    if args.test_dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                                    (0.5, 0.5, 0.5))])
        dataset_new = datasets.CIFAR10(root='.', train=True, download=True,transform=transform)
        dataloader_new = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size, shuffle=True)

        testset = datasets.CIFAR10(root='./data', train=False,
                                   download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        dataset_train = datasets.CIFAR10(root='.', train=True,
                                         download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  shuffle=True)

    if args.test_dataset == 'FMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
                                        transforms.Normalize((0.1307,), (0.3081,))])

        dataset_new = datasets.FashionMNIST(root='.', train=True, download=True,
                                                 transform=transform)
        dataloader_new = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size,
                                                  shuffle=False)
        ## for training later
        testset = datasets.FashionMNIST(root='./data', train=False,
                                   download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        dataset_train = datasets.FashionMNIST(root='.', train=True,
                                         download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  shuffle=False)

    if args.test_dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_new = datasets.MNIST(root='.', train=True, download=True,
                                                 transform=transform)
        dataloader_new = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size,
                                                  shuffle=False)
        ## for training later
        testset = datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        dataset_train = datasets.MNIST(root='.', train=True,
                                         download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  shuffle=False)

    if args.pre_model == 'simple':
        network = Simple_Net().to(args.device)
        network.load_state_dict(torch.load(args.pre_model_path), strict=False)
    if args.pre_model == 'vgg':
        network = models.vgg19(pretrained=True).to(args.device)
    if args.pre_model == 'diff_simple':
        network = diff_net().to(args.device)
        network.load_state_dict(torch.load(args.pre_model_path), strict=False)

    if args.with_custom_grad:
        rg = CustomRequireGrad(net=network,pretrained_data_set= dataloader_pre, input_test= dataloader_new,
                               dist_processing_method='fft', batches_num=args.num_batch_analysis,
                               percent=args.percent, deepest_layer=args.deepest_layer,
                               save_folder=args.folder_save_stats + str(args.num_run),
                               process_method=args.process_method, similarity=args.similarity_func)
        rg.run()
        ############# running training session :

    net = network
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    cycle_opt = torch.optim.lr_scheduler.CyclicLR(optimizer, args.lr, args.lr*100,
                                                  step_size_up=10)

    writer = SummaryWriter(args.folder_save_stats+str(args.num_run)+'/'+ str(args.with_custom_grad) + '_' + str(amount_data) + '_samples')
    accuracy = []

    dict_a =args.__dict__
    with open(args.folder_save_stats+str(args.num_run)+'/'+ str(args.with_custom_grad) + '_' + str(amount_data) + '_samples'+ '_config_run.csv', 'w') as f:
        for key in dict_a.keys():
            f.write("%s,%s\n" % (key, dict_a[key]))

    for epoch in range(args.num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, args.seed):
            if i-args.seed >= amount_data / batch_size:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # plt.imshow(inputs.detach().cpu().numpy()[0].transpose())
            optimizer.zero_grad()
            outputs = net(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            if args.with_custom_grad:
                if epoch < 35:
                    rg.update_grads(net)
            optimizer.step()
            if args.cycle_opt:
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
                if count > 4000:
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
        writer.add_scalar('LR',
                         (optimizer.param_groups[0]['lr']),
                          epoch)

        accuracy.append((epoch, 100 * correct / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_run', type=int, default=15)
    parser.add_argument('--seed', type=int, default=10)

    #model:
    parser.add_argument('--pre_model', type=str, default='simple')
    parser.add_argument('--pre_model_path', type=str, default=r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\model9')
    parser.add_argument('--pre_dataset', type=str, default='KMNIST')
    parser.add_argument('--test_dataset', type=str, default='FMNIST')

    #custom gradient:
    parser.add_argument('--with_custom_grad', type=bool, default=True)
    parser.add_argument('--percent', type=int, default=75)
    parser.add_argument('--num_batch_analysis', type=int, default=40)
    parser.add_argument('--folder_save_stats', type=str, default=r'./all_runs/')
    parser.add_argument('--process_method', type=str, default='fft')
    parser.add_argument('--deepest_layer', type=int, default=8)
    parser.add_argument('--similarity_func', type=str, default='ws')

    # Training
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_batch', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=8e-3)
    parser.add_argument('--cycle_opt', type=bool, default=True)

    main(parser.parse_args())
