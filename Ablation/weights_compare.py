import pickle
import numpy as np
from torchvision import transforms
import torch
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
from Pretrained_creation import  Simple_Net, Large_Simple_Net
from CustomStatisticGrad import CustomStatisticGrad
import argparse
from datasets.data_utils import cifar_part, kmnist_part, mnist_part, Fmnist_part
import os
import matplotlib.pyplot as plt

with open(r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\Ablation\weight_zeroing_idx.pickle', 'rb') as handle:
    weight_dict_change = pickle.load(handle)

network_original_path = r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\diff_net\_KMNIST19'
network_original =   Large_Simple_Net().to('cuda')
network_original.load_state_dict(torch.load(network_original_path), strict=True)

network_alg_path = r"C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\diff_net\accuracy_24.54_C_samples_8_base_KMNIST_ablation21"
network_alg =   Large_Simple_Net().to('cuda')
network_alg.load_state_dict(torch.load(network_alg_path), strict=True)

network_alg_path_more_samples =r'C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\diff_net\accuracy_78.46_C_samples_8_KMNIST_ablation30'
network_alg_more_samples =   Large_Simple_Net().to('cuda')
network_alg_more_samples.load_state_dict(torch.load(network_alg_path_more_samples), strict=True)

network_alg_path_middle_samples = r"C:\Users\yuval\PycharmProjects\smart_pretrained\Statistics-pretrained\saved_models\diff_net\accuracy_58.62_C_samples_8_KMNIST_ablation42"
network_alg_middle_samples =   Large_Simple_Net().to('cuda')
network_alg_middle_samples.load_state_dict(torch.load(network_alg_path_middle_samples), strict=True)

layer_dict_error_smaller = {}
layer_dict_error_larger = {}
layer_dict_error_middle = {}

for k,v in weight_dict_change.items():

    print(len(v))
    size_2_average = 0
    error_weight = 0
    error_weight_more_samples = 0
    error_weight_middle_samples = 0
    weights_original = network_original.state_dict()[k][v]
    num, depth, w,h = np.shape(weights_original)
    weights_original = weights_original.view(-1,depth*w,h).cpu().detach()
    weights_alg = network_alg.state_dict()[k][v]
    weights_alg = weights_alg.view(-1,depth*w,h).cpu().detach()

    weights_alg_more_samples = network_alg_more_samples.state_dict()[k][v]
    weights_alg_more_samples  = weights_alg_more_samples.view(-1,depth*w,h).cpu().detach()


    weights_alg_middle_samples = network_alg_middle_samples.state_dict()[k][v]
    weights_alg_middle_samples  = weights_alg_middle_samples.view(-1,depth*w,h).cpu().detach()

#fig, axs = plt.subplots(1, num)
    #fig.suptitle(f'layer: {k}')
    plt.figure()
    cat_weights = []
    for i in range(num):
        catted_weight = (torch.cat([weights_alg[i], torch.ones((depth*w,1)) * torch.nan,weights_original[i], torch.ones((depth*w,1)) * torch.nan, weights_alg_more_samples[i] ],dim=1)  )
        error_weight += torch.sum(torch.abs(weights_alg[i] -weights_original[i]) )
        error_weight_more_samples += torch.sum(torch.abs(weights_alg_more_samples[i] -weights_original[i]) )
        error_weight_middle_samples += torch.sum(torch.abs(weights_alg_middle_samples[i] -weights_original[i]) )

        size_2_average +=  len(torch.flatten(weights_original[i]) )

        if len(cat_weights) >0:
            cat_weights = torch.cat([ cat_weights, torch.ones((depth*w,3)) * torch.nan, catted_weight], dim=1)
        else:
            cat_weights = catted_weight
        plt.imshow(cat_weights.numpy().transpose())
        #avg_error = np.round((error_weight/size_2_average).numpy(),7)
        #avg_error_larger  = np.round((error_weight_more_samples/size_2_average).numpy(),7)
        #plt.title((f'layer: {k}' f'  Avg error 8 samples' + str(avg_error) + f'  Avg error larger samples' + str(avg_error_larger)))

    avg_error = np.round((error_weight/size_2_average).numpy(),7)
    avg_error_larger  = np.round((error_weight_more_samples/size_2_average).numpy(),7)
    avg_error_middle  = np.round((error_weight_middle_samples/size_2_average).numpy(),7)

    layer_dict_error_smaller[k] = avg_error
    layer_dict_error_larger[k] = avg_error_larger
    layer_dict_error_middle[k] = avg_error_middle

fig, ax = plt.subplots(1,1)
plt.scatter(np.arange(0,len(layer_dict_error_smaller.values())),layer_dict_error_smaller.values(), label='50 samples')
plt.scatter(np.arange(0,len(layer_dict_error_larger.values())),layer_dict_error_larger.values(),label='200 samples')
plt.scatter(np.arange(0,len(layer_dict_error_middle.values())),layer_dict_error_middle.values(),label='150 samples')

plt.legend()
ax.set_xticks(np.arange(0,len(layer_dict_error_smaller.values())))

ax.set_xticklabels(layer_dict_error_smaller.keys())
plt.ylabel('Average L1 weight difference')
plt.title('Original weights vs new weights')
