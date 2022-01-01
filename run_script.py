import numpy as np
from torchvision import transforms
import torch
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary
import torch.optim as optim
from Pretrained_creation import  Simple_Net, diff_net, Large_Simple_Net
from distribution_net import CustomRequireGrad
import argparse
from data_utils import cifar_part, kmnist_part, mnist_part, Fmnist_part
import os
def replicate_channels(im):
    return torch.stack([im, im, im]).squeeze()
def main(args):
    torch.cuda.manual_seed_all(args.seed)
    batch_size = args.batch_size
    num_b = args.num_batch
    amount_data =num_b*batch_size
    if not os.path.exists(args.folder_save_stats):
        os.mkdir(args.folder_save_stats)
    ############# statistics for custom :
    if args.pre_dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                                    (0.5, 0.5, 0.5))])
        #dataset_pre = datasets.CIFAR10(root='.', train=True, download=True,transform=transform)
        dataset_pre = cifar_part(transform, train=True, middle_range=5, upper= True)
        dataloader_pre = torch.utils.data.DataLoader(dataset_pre, batch_size=batch_size,shuffle=False)

    if args.pre_dataset == 'KMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_pre = kmnist_part(transform, train=True, middle_range=5, upper= True)
        dataloader_pre = torch.utils.data.DataLoader(dataset_pre, batch_size=batch_size,
                                                     shuffle=True)

    if args.pre_dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_pre = mnist_part(transform, train=True, middle_range=5, upper= True)
        dataloader_pre = torch.utils.data.DataLoader(dataset_pre, batch_size=batch_size,
                                                     shuffle=True)

    if args.pre_dataset == 'FMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        dataset_pre = Fmnist_part(transform, train=True, middle_range=5, upper= True)
        dataloader_pre = torch.utils.data.DataLoader(dataset_pre, batch_size=batch_size,
                                                     shuffle=True)

    if args.test_dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                                    (0.5, 0.5, 0.5))])
        #dataset_new = datasets.CIFAR10(root='.', train=True, download=True,transform=transform)
        dataset_new = cifar_part(transform, train=True, middle_range=5, upper= False)
        dataloader_new = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size, shuffle=True)

        #testset = datasets.CIFAR10(root='./data', train=False,
        #                           download=True, transform=transform)
        testset = cifar_part(transform, train=False, middle_range=5, upper= False)

        testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                                 shuffle=False, num_workers=0)

        dataset_train = dataset_new

        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  shuffle=False)

    if args.test_dataset == 'FMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        #dataset_new = datasets.MNIST(root='.', train=True, download=True,
        #                             transform=transform)
        #
        dataset_new = Fmnist_part(transform, train=True, middle_range=5, upper=False)
        dataloader_new = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size,
                                                     shuffle=False)
        ## for training later
        testset = Fmnist_part(transform, train=False, middle_range=5, upper=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        dataset_train = dataset_new
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  shuffle=False)

    if args.test_dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        #dataset_new = datasets.MNIST(root='.', train=True, download=True,
        #                             transform=transform)
        #
        dataset_new = mnist_part(transform, train=True, middle_range=5, upper=False)
        dataloader_new = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size,
                                                     shuffle=False)
        ## for training later
        testset = mnist_part(transform, train=False, middle_range=5, upper=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        dataset_train = dataset_new
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  shuffle=False)

    if args.test_dataset == 'KMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64)),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        #dataset_new = datasets.MNIST(root='.', train=True, download=True,
        #                             transform=transform)
        #
        dataset_new = kmnist_part(transform, train=True, middle_range=5, upper=True)
        dataloader_new = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size,
                                                     shuffle=False)
        ## for training later
        testset = kmnist_part(transform, train=False, middle_range=5, upper=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        dataset_train = dataset_new
        trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                  shuffle=False)

    if args.pre_model == 'simple':
        network = Simple_Net().to(args.device)
        network.load_state_dict(torch.load(args.pre_model_path), strict=True)
    if args.pre_model == 'vgg':
        network = models.vgg19(pretrained=True).to(args.device)
        network.load_state_dict(torch.load(args.pre_model_path), strict=True)
        # To adapt the 5 outputs
        network.classifier[6] = torch.nn.Linear(in_features=4096, out_features=5).cuda()
    #path = r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\models33'
        #network.load_state_dict(torch.load(path),strict=True)
    if args.pre_model == 'diff_net':
        network =   Large_Simple_Net().to(args.device)
        network.load_state_dict(torch.load(args.pre_model_path), strict=True)
    if args.pre_model == 'densenet121':
        network = models.densenet121(pretrained=True).to(args.device)

        #network.load_state_dict(torch.load(args.pre_model_path), strict=True)

    if args.with_custom_grad:
        freeze_all_str = ''
        rg = CustomRequireGrad(net=network,pretrained_data_set= dataloader_pre, input_test= dataloader_new,
                               dist_processing_method=args.process_method, batches_num=args.num_batch_analysis,
                               percent=args.percent, deepest_layer=args.deepest_layer,
                               save_folder=args.folder_save_stats + str(args.num_run),
                               process_method=args.process_method, similarity=args.similarity_func)
        rg.run(mode=args.run_mode)
        net = network
    ############# running training session :
    else: # Freezing everything except the last layer
        freeze_all_str = '_freeze_all_layers_' + str(args.freeze_all)
        net = network
        for ind, param in enumerate(net.parameters()):
            #print(ind)
            if ind < -2:
                param.requires_grad = not args.freeze_all


    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr)

    cycle_opt = torch.optim.lr_scheduler.CyclicLR(optimizer, args.lr/10, args.lr * 20,
                                                  step_size_up=10)
    if args.with_custom_grad == False:
        freeze_mode = 'None'
    else:
        freeze_mode = args.freezing_mode
    writer = SummaryWriter(args.folder_save_stats+str(args.num_run)+'/'+ str(args.with_custom_grad) + '_' + str(amount_data) + '_samples' + 'freezing_mode_' + freeze_mode + freeze_all_str)
    accuracy = []

    dict_a =args.__dict__
    with open(args.folder_save_stats+str(args.num_run)+'/'+ str(args.with_custom_grad) + '_' + str(amount_data) + '_samples'+ '_config_run.csv', 'w') as f:
        for key in dict_a.keys():
            f.write("%s,%s\n" % (key, dict_a[key]))
    count = 0
    running_loss_PREV = -1
    count_time_same_loss = 0
    count_larger_th_loss=0
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
                    rg.update_grads(net=net, epoch=epoch, mode=args.freezing_mode)
            optimizer.step()
            if args.cycle_opt:
                cycle_opt.step()
            running_loss += loss.item()
        if (np.abs(running_loss - running_loss_PREV)) < 1e-4:
            count_time_same_loss+=1
            if count_time_same_loss > 11:
                break
            optimizer.param_groups[0]['lr'] *= 2
            optimizer.param_groups[0]['lr'] = np.min([optimizer.param_groups[0]['lr'], 0.007])
            print('lr is:' + str(optimizer.param_groups[0]['lr'] ))
        else:
            count_larger_th_loss += 1
            if count_larger_th_loss > 8:
                optimizer.param_groups[0]['lr'] = args.lr
                count_larger_th_loss = 0
            count_time_same_loss = 0
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / (i - args.seed)))

        running_loss_PREV= running_loss

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
                if count > 10000:
                    break
        print(
            'Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))
        writer.add_scalar('Test - Accuracy',
                          (100 * correct / total),
                          epoch)
        writer.add_scalar('Train - Loss',
                          (running_loss / (i - args.seed)),
                          epoch)
        writer.add_scalar('LR',
                          (optimizer.param_groups[0]['lr']),
                          epoch)

        accuracy.append((epoch, 100 * correct / total))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_run', type=int, default=1231231),
    parser.add_argument('--seed', type=int, default=3)

    #model:
    parser.add_argument('--pre_model', type=str, default='densenet121')
    parser.add_argument('--pre_model_path', type=str, default=r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\densenet121\models_dense_net_MNIST5')
    parser.add_argument('--pre_dataset', type=str, default='MNIST')
    parser.add_argument('--test_dataset', type=str, default='FMNIST')

    #custom gradient:
    parser.add_argument('--with_custom_grad', type=bool, default=True)
    parser.add_argument('--freeze_all', type=bool, default=True, help='only for custom_grad is false, if True,'
                                                                       'freeze everything except the classification'
                                                                       'layer')

    parser.add_argument('--percent', type=int, default=25)
    parser.add_argument('--num_batch_analysis', type=int, default=40)
    parser.add_argument('--folder_save_stats', type=str, default=r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\different_data_domain\\')
    parser.add_argument('--process_method', type=str, default='linear')
    parser.add_argument('--deepest_layer', type=int, default=40)
    parser.add_argument('--run_mode', type=str, default='per_layer')
    parser.add_argument('--freezing_mode', type=str, default='per_layer')
    parser.add_argument('--similarity_func', type=str, default='KS')

    # Training
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_batch', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=35)
    parser.add_argument('--lr', type=float, default=1E-5)
    parser.add_argument('--cycle_opt', type=bool, default=True)
    # Script to run over all params
    num_batch = [  10, 20, 40, 80]
    num_seed  = [ 120, 19, 255]
    args = parser.parse_args()
    for id, batch_size in enumerate(num_batch):
        #print(id)
        args.num_run = id
        for n_seed in num_seed:
            args.num_batch = batch_size
            args.seed = n_seed
            main(args)
