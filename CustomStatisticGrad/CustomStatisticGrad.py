import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import os
import torch
import scipy
import torchvision
from  CustomStatisticGrad.PreProcess import PriorPreprocess
def kl(p, q):
    # Kl divergence metric
    p = np.abs(np.asarray(p, dtype=np.float) + 1e-15)
    q = np.abs(np.asarray(q, dtype=np.float) + 1e-15)
    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def smoothed_hist_kl_distance(a, b, nbins=20):
    ahist, bhist = (np.histogram(a, bins=nbins)[0],
                    np.histogram(b, bins=nbins)[0])
    return kl(ahist, bhist)

class CustomStatisticGrad:
    '''
    This class collects statistics for each kernel/layer given a pre-trained model and two datasets.
    Performs a similarity test over the collected distributions kernel wise.
    Performs gradient modification of the chosen kernels/layers during backpropagation.

    Required inputs:
    net -> Network architecture.
    pretrained_data_set -> Dataloader points to the dataset the pre-trained network was trained.
    input_test -> Dataloader points to the new task dataset .
    '''
    def __init__(self, net: torchvision.models , pretrained_data_set: torch.utils.data.DataLoader, input_test:torch.utils.data.DataLoader,
                 dist_processing_method: str='fft', batches_num: int=10, percent: int=70,
                 deepest_layer: int=11,similarity: str='ws', save_folder: str='./',
                 process_method: str='fft'):
        self.process_method=process_method
        self.pretrained_data_set = pretrained_data_set
        self.input_test = input_test
        self.network = net
        self.max_layer = deepest_layer
        self.dist_processing_method = dist_processing_method
        self.num_batches = batches_num
        self.threshold_percent = percent
        self.activation = {}
        self.batch_size = self.pretrained_data_set.batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.similarity = similarity
        self.save_folder = save_folder
        self.ablation_mode = True
        self.PriorPreprocess =PriorPreprocess
        self.modules_name_list = []

    def _plot_distribution(self, ind_layer, layer_pretrained, layer_test,
                           stats_val=0, method='gram', kernel_num=0, pvalue=0, num_plots=20, ax_sub=None, layer_name=''):

        if method != 'gram':
            # Assuming log normal dist due to relu :
            plt.subplot(np.sqrt(num_plots), np.sqrt(num_plots),
                        ind_layer % num_plots + 1)
            values_pre, axis_val_pre = np.histogram(np.log(layer_pretrained),100)
            plt.plot(axis_val_pre[10:], values_pre[9:] / np.max(values_pre[10:]),
                     linewidth=4, alpha=0.7, label='D')
            values, axis_val = np.histogram(np.log(layer_test), 100)
            plt.plot(axis_val[10:], values[9:] / np.max(values[10:]), linewidth=4,
                     alpha=0.7, label='D2')
            #plt.legend()
            plt.xlim([-5, 3])
            plt.ylim([0, 1 + 0.1])
            plt.title('Layer : ' + str(ind_layer) + 'p: ' + str(np.round(
                stats_val, 2)),fontsize=7)
        else:

            values, axis_val = np.histogram(layer_test, 100)
            ax_sub[self.plot_counter ].plot(axis_val[10:], values[9:] / np.max(values[10:]),
                                            linewidth=4,
                                            alpha=0.7, label='Dtest')
            minx_1 = np.min(axis_val[10:])
            maxx_1 = np.max(axis_val[10:])

            values, axis_val = np.histogram(layer_pretrained, 100)
            ax_sub[self.plot_counter].plot(axis_val[10:],
                                           values[9:] / np.max(values[10:]),
                                           linewidth=4,
                                           alpha=0.7, label='Dpre')
            minx_2 = np.min(axis_val[10:])
            maxx_2 = np.max(axis_val[10:])

            min_x = int(np.min([minx_2, minx_1])) - 0.5
            max_x = int(np.max([maxx_1, maxx_2])) + 0.5

            ax_sub[self.plot_counter].legend()
            plt.xlim([min_x, max_x])
            plt.ylim([0, 1 + 0.1])
            plt.title(' Kernel num : ' + str(
                kernel_num) +' p:'+str(pvalue))
            plt.suptitle('Layer number:' + layer_name)

            self.plot_counter +=1

    def plot_activation(self, name_layer, indexes, im_batch=1,
                        save_path=None):
        num_kernels = np.size(indexes)
        num_per_axis = int(np.ceil(np.sqrt(num_kernels)))
        for i, index in enumerate(indexes):
            fig_pre = plt.figure(1)
            plt.subplot(num_per_axis, num_per_axis,i + 1)
            plt.title('kernel index:' + str(index))
            plt.imshow(self.activations_input_pre[name_layer][im_batch][index])

            fig_new = plt.figure(2)
            plt.subplot(num_per_axis, num_per_axis,i + 1)
            plt.title('kernel index:' + str(index))
            plt.imshow(self.activations_input_test[name_layer][im_batch][index])

        fig_bpm = plt.figure(3)
        plt.title('kernel index:' + str(index))
        #plt.imshow(np.squeeze( 1 * (self.bpm.cpu().numpy() > 0)[im_batch]))

        fig_bpm_test = plt.figure(4)
        plt.title('kernel index:' + str(index))
        #plt.imshow(np.squeeze( 1 * (self.bpm_test.cpu().numpy() > 0)[im_batch]))

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig_pre.savefig(save_path+'/'+name_layer+"_pre.jpg", dpi=900)
        fig_new.savefig(save_path+'/'+name_layer+"_new.jpg", dpi=900)
        #fig_bpm.savefig(save_path+'/'+'bpm'+"_new.jpg", dpi=900)
        #fig_bpm_test.savefig(save_path+'/'+'bpm_pre'+"_pre.jpg", dpi=900)

        fig_new.clf()
        fig_pre.clf()
        plt.close('all')

    def update_grads(self, net, mode='normal',epoch = 1):
        # Mode freeze but bias, freezes all, execept weights that are have different distribution and in those layers
        #only the bias is unfreeze.
        dict_model = dict(net.named_modules())
        for name in self.modules_name_list:

            module = dict_model[name]
            if  'weight' in module._parameters and 'Conv'  in module._get_name() :  # Skip module modules

                if len(list(module.parameters())) > 0:  # weights
                    if len(self.layers_grad_mult[name]) > 0:
                        if mode == 'freeze_except_bias':
                            module.weight.grad *= np.min([1e-2, (2**epoch) * torch.tensor(np.min(self.layers_grad_mult[name]['weights'])
                                                                    ).to(self.device)])
                        else:
                            module.weight.grad *= torch.FloatTensor(
                                self.layers_grad_mult[name]['weights']).to(self.device)
                            if module.bias != None:
                                module.bias.grad *= torch.FloatTensor(
                                    np.squeeze(self.layers_grad_mult[name]['bias'])).to(self.device)

    def get_activation(self, name):
        def hook(_, __, output):
            try:
                self.activation[name] = output.detach().cpu().numpy().copy()
                if (np.sum(np.abs(output.detach().cpu().numpy()) <=1e-8  ) > 1000):
                    print('errr')
            except:
                self.activation[name] = None
        return hook

    def _prepare_input_tensor(self):
        self.pretrained_iter = map(lambda v: v[0].to(self.device),
                                   self.pretrained_data_set)
        self.input_test_iter = map(lambda v: v[0].to(self.device), self.input_test)

    def _prepare_input_tensor(self):
        self.pretrained_iter = self.pretrained_data_set
        self.input_test_iter =self.input_test

    def _hook_assign_module(self):
        self.modules_name_list = []
        hooks = {}
        for ind, (name, module) in enumerate(self.network.named_modules()):
            #if ind > self.max_layer:
            #    break
            # body.2.conv2
            if ind > self.max_layer:
                break
            if len(list(module._modules)) < 2 and \
                    'weight' in module._parameters and 'Conv'  in module._get_name() :  # Skip module modules
                self.modules_name_list.append(name)
                hooks[name] = module.register_forward_hook(
                    self.get_activation(name))


    def _calc_layers_outputs(self, batches_num=10, mode='normal'):
        ## Hook to each relavant module
        self._hook_assign_module()

        for ind_batch, (input_model, input_test) \
                in enumerate(zip(self.pretrained_iter,
                                 self.input_test_iter)):
            if ind_batch > batches_num:
                break
            self.activation = {} # clear all activations every batch

            bp = input_model[0].to(self.device, dtype=torch.float)
            bp_test = input_test[0].to(self.device, dtype=torch.float)

            self.network(bp)
            self.activations_input_pre = self.activation.copy()
            self.activation = {}
            self.network(bp_test)
            self.activations_input_test = self.activation.copy()
            values_post_test = 0
            values_post_pre = 0

            for name in self.modules_name_list:
                if self.activations_input_test[name] is not None:
                    dist_new = self.activations_input_test[name].copy()
                    values_pre = self.activations_input_pre[name].copy()
                    if mode=='per_layer':
                        out_new = self.gram_layer(dist_new)
                        out_pre = self.gram_layer(values_pre)

                        self.statistic_test[name].append(np.ravel(out_new))
                        self.statistic_pretrained[name].append(np.ravel(out_pre))
                    else:
                        dist_new_tot_size_per_ch = None
                        dist_pre_tot_size_per_ch = None
                        if len(np.shape(dist_new)) > 2:
                            ## seperating distribution per kernel
                            # -> [#channels, #BatchSIze,#activation size (#,#) ]
                            dist_new_channel_first = np.transpose(dist_new, [1, 0, 2, 3])
                            dist_new_tot_size_per_ch = np.zeros(((np.shape(dist_new_channel_first)[0],
                                                                  np.prod(np.shape(dist_new_channel_first)
                                                                          [1:]))))

                            values_pre_channel_first = np.transpose(values_pre, [1, 0, 2, 3])
                            dist_pre_tot_size_per_ch = np.zeros(((np.shape(values_pre_channel_first)[0],
                                                                  np.prod(np.shape(values_pre_channel_first)
                                                                          [1:]))))

                            # Required shape per channel:
                            transform_prior = self.PriorPreprocess(method=self.process_method, shape_act=np.shape(dist_new), **self.__dict__)
                            values_post_test, values_post_pre = transform_prior.initialize_list()


                            ## Aggragating along in a dict for each  channel:

                            for ll in range(np.shape(dist_new_channel_first)[0]):
                                dist_new_tot_size_per_ch[ll] = \
                                    np.ravel(dist_new_channel_first[ll])
                                dist_pre_tot_size_per_ch[ll] = \
                                    np.ravel(values_pre_channel_first[ll])
                                values_post_test[ll] = transform_prior.run_prior_transformation(
                                    dist_new_channel_first[ll])
                                values_post_pre[ll] = transform_prior.run_prior_transformation(
                                    values_pre_channel_first[ll])
                            if len(dist_pre_tot_size_per_ch[0]) > 200e3:
                                dist_new_tot_size_per_ch = dist_new_tot_size_per_ch[
                                                           :, np.random.randint(
                                    0, len(dist_pre_tot_size_per_ch[0]), size=5000)]
                                dist_pre_tot_size_per_ch = dist_pre_tot_size_per_ch[:, np.random.randint(
                                    0, len(dist_pre_tot_size_per_ch[0]), size=5000)]
                        # Concatanating the data along the different batches :
                        if len(np.shape(self.stats_test[name])) == 0:
                            self.stats_test[name] = values_post_test
                            self.pre_trained_outputs[name] = values_post_pre
                        else:
                            clipped_log_gram = np.clip(
                                (np.abs(values_post_test)), -2e6, 2e6)
                            self.stats_test[name] = np.concatenate(
                                [self.stats_test[name], clipped_log_gram], axis=1)
                            self.pre_trained_outputs[name] = np.concatenate(
                                [self.pre_trained_outputs[name], np.clip((np.abs(values_post_pre)), -2e6, 2e6)], axis=1)
                        self.statistic_test[name].append(dist_new_tot_size_per_ch)
                        self.statistic_pretrained[name].append(dist_pre_tot_size_per_ch)

            #self.bpm = bpm
            #self.bpm_test = bpm_test

    def _calc_layers_outputs(self, batches_num=10, mode='normal'):
        output_pre = []
        output_test = []
        #### Runs over the first layer.
        for ind_batch, (input_model, input_test) \
                in enumerate(zip(self.pretrained_iter,
                                 self.input_test_iter)):
            if ind_batch > batches_num:
                break
            bp = input_model[0].to(self.device, dtype=torch.float)
            bp_test = input_test[0].to(self.device, dtype=torch.float)
            module_f = list(self.network.children())[0]
            name_module = list(self.network.named_modules())[1][0]

            if len(output_pre) > 0:
                output_pre = torch.cat([output_pre , module_f.forward(bp)] ,dim=0)
                output_test = torch.cat([output_test , module_f.forward(bp_test)] ,dim=0)
            else:
                output_pre = module_f.forward(bp)
                output_test = module_f.forward(bp_test)

        output_test_nump = np.transpose(output_test.cpu().detach().numpy(), [1,0,2,3])
        output_pre_nump = np.transpose(output_pre.cpu().detach().numpy(), [1,0,2,3])
        sim_ch =[]
        self.kernel_mean = defaultdict(list)
        self.kernel_std = defaultdict(list)
        if len(list(module_f._modules)) < 2 and \
                'weight' in module_f._parameters and 'Conv'  in module_f._get_name() :  # Skip module modules
            self.modules_name_list.append(name_module)

        for out_test_ch,output_pre_ch  in zip(output_test_nump,output_pre_nump):
            """
            Compansate forwarding distribution from non similar kernels to later similar kernels.
            There is a possiblitly that a generalized kernel in the later layers wont be identified as a generalized due to out of distribution input from previous layers.
            We want to match the distribution of the source in order to eliminate/ reduce such cases.
            By doing the following:
            1.Calculating addition value for the mean of each kernel in order to distribute the same as the source.
            2.Calculating the multiplication needed for the std in order to distribute the same.
            Than we forward our input layer by layer, before we forward pass to the next layer, we modify it's mean and std values of the kernels that identified as non-generalized kernels.
            """
            # Calculate the similarity:
            transform_prior = self.PriorPreprocess(method=self.process_method, shape_act=np.shape(out_test_ch), **self.__dict__)
            values_post_test, values_post_pre = transform_prior.initialize_list()
            out_test_ch_transform = transform_prior.run_prior_transformation(
                out_test_ch)
            output_pre_ch_transform = transform_prior.run_prior_transformation(
                output_pre_ch)
            sim_ch.append(kl(np.ravel(out_test_ch_transform), np.ravel(output_pre_ch_transform)) )
            # Calculate the required mean and std:
            self.kernel_mean[name_module].append(np.mean(-1*np.ravel(out_test_ch)) + np.mean(np.ravel(output_pre_ch)))
            self.kernel_std[name_module].append( np.std(np.ravel(output_pre_ch)) / np.std(np.ravel(out_test_ch)))

        # Here we chose a hard threshold. - Should be parametrized by the user.
        bad_sim = sim_ch  > np.mean(sim_ch) * 0.9
        L = len(list(self.network.children()))
        name_module = list(self.network.named_modules())[1][0]

        self.stats_value_per_layer[name_module] = sim_ch
        for index_module, module_f in enumerate(list(self.network.children())[1: L - 1 ]):
            output_pre_new = []
            output_test_new = []
            inds_replace = np.where(bad_sim)[0]
            for indr in inds_replace:
                indr = int(indr)
                output_test[:,torch.tensor(indr, device='cuda')]  = (output_test[:, torch.tensor(indr, device='cuda')] +  self.kernel_mean[name_module][indr] ) * self.kernel_std[name_module][indr]

            for ind in range(len(output_pre) - 1):
                #print(ind)## Continue later.

                if len(output_pre_new) > 0:
                        output_pre_new = torch.cat([output_pre_new , module_f.forward(output_pre[ind].unsqueeze(0))] ,dim=0)
                        output_test_new = torch.cat([output_test_new , module_f.forward(output_test[ind].unsqueeze(0))] ,dim=0)
                else:
                        output_pre_new = module_f.forward(output_pre[ind].unsqueeze(0))
                        output_test_new = module_f.forward(output_test[ind].unsqueeze(0) )


            output_test_nump = np.transpose(output_test_new.cpu().detach().numpy(), [1,0,2,3])
            output_pre_nump = np.transpose(output_pre_new.cpu().detach().numpy(), [1,0,2,3])
            sim_ch =[]
            name_module = list(self.network.named_modules())[index_module + 2][0]

            for out_test_ch,output_pre_ch  in zip(output_test_nump,output_pre_nump):
                #sim_ch.append(stats.ks_2samp(np.ravel(out_test_ch), np.ravel(output_pre_ch))[1] )
                values_post_test, values_post_pre = transform_prior.initialize_list()
                out_test_ch_transform = transform_prior.run_prior_transformation(
                    out_test_ch)
                output_pre_ch_transform = transform_prior.run_prior_transformation(
                    output_pre_ch)
                sim_ch.append(kl(np.ravel(out_test_ch_transform), np.ravel(output_pre_ch_transform)) )
                self.kernel_mean[name_module].append(np.mean(-1*np.ravel(out_test_ch)) + np.mean(np.ravel(output_pre_ch)))
                self.kernel_std[name_module].append( np.std(np.ravel(output_pre_ch)) / np.std(np.ravel(out_test_ch)))
            self.modules_name_list.append(name_module)


            self.stats_value_per_layer[name_module] = sim_ch


            #if len(list(module_f._modules)) < 2 and \
            #        'weight' in module_f._parameters and 'Conv'  in module_f._get_name() :  # Skip module modules
            #    self.modules_name_list.append(name_module)


            bad_sim = sim_ch  > np.mean(sim_ch) * 0.9
            output_pre = output_pre_new.clone()
            output_test = output_test_new.clone()

    def gram_layer(self, dist_new):
        b_size, num_filters, w, h = np.shape(dist_new)
        gram_prepare = np.reshape(dist_new, (b_size, num_filters, w * h))
        gram_output = [gr @ gr.T for gr in gram_prepare]
        return gram_output

    def _metric_compare(self):
        for ind_layer, name in enumerate(self.modules_name_list):
            if ind_layer > self.max_layer:
                break
            num_plots = np.shape(self.pre_trained_outputs[name])[0]
            fig = plt.figure(ind_layer, figsize=(20, 20))
            ax_sub = fig.subplots(int(np.ceil(np.sqrt(num_plots))), int(np.ceil(np.sqrt(num_plots))))
            ax_sub = ax_sub.ravel()
            stats_value = []
            self.plot_counter = 0
            if np.size(self.stats_test[name]) > 1:  # check if has values
                for ind_inside_layer, (test, pre) in enumerate(zip(
                        self.stats_test[name], self.pre_trained_outputs[name])):
                    if np.size(self.activations_input_pre[name][0][ind_inside_layer]) > 20:
                        ## Convert from log normal to normal distribution. (assumtion been made)
                        test_in = np.log(np.abs(test[np.abs(test) > 1e-7]))
                        pre_in =  np.log(np.abs(pre[np.abs(pre)  > 1e-7]))
                        ## Similarity units! regardless of the test
                        if len(test_in) > 20 and len(pre_in) > 20: # Chekck there are enought values for statistics
                            if self.similarity == 'KS':
                                sim = 1/(1e-8 + stats.ks_2samp(test_in, pre_in)[0])
                            if self.similarity == 'kl':
                                sim = kl(test_in, pre_in)
                            if self.similarity =='ws':
                                sim = 1 / ( 1e-8 + scipy.stats.wasserstein_distance(test_in, pre_in))
                            if self.similarity == 'euclidian':
                                sim = np.sum(np.abs(test) + np.abs(pre)) / (np.sum(np.abs(test - pre)) + 1e-8)

                        else: # non sufficient points mark as non similarity
                            sim = -1
                        self._plot_distribution(ind_layer=int(ind_layer),
                                                layer_pretrained=pre_in,
                                                layer_test=test_in,
                                                kernel_num=ind_inside_layer,
                                                method='gram', pvalue=sim, num_plots=num_plots,ax_sub=ax_sub,layer_name=name)
                    else:
                        sim = [-1]
                    stats_value.append(sim)
            else:
                stats_value = [-1]
                self.stats_value_per_layer[name] = stats_value.copy()
            self.stats_value_per_layer[name] = stats_value.copy()
            ### Finished layer loop over kernels :
            if not os.path.exists(self.save_folder + '//dist/'):
                os.makedirs(self.save_folder + '//dist/')
            plt.savefig(self.save_folder + '//dist/' + '//Layer_name_' +
                        name + '.jpg', dpi=400)
            plt.close()

    def _metric_compare_full_layer(self):
        num_plots = len(self.modules_name_list)
        fig = plt.figure(1, figsize=(20, 20))
        ax_sub = fig.subplots(int(np.ceil(np.sqrt(num_plots))), int(np.ceil(np.sqrt(num_plots))))
        self.plot_counter = 0
        ax_sub = ax_sub.ravel()
        for ind_layer, name in enumerate(self.modules_name_list):
            if ind_layer > self.max_layer:
                break
            stats_value = []
            if np.size(self.statistic_test[name]) > 1:  # check if has values
                test = np.ravel(self.statistic_test[name])
                pre  = np.ravel(self.statistic_pretrained[name])
                if np.size(self.activations_input_pre[name][0]) > 20:
                    ## Convert from log normal to normal distribution. (assumtion been made)

                    test_in = np.log(np.abs(test[np.abs(test) > 1e-7]))
                    pre_in =  np.log(np.abs(pre[np.abs(pre)  > 1e-7]))
                    test_in = test
                    pre_in =  pre

                ## Similarity units! regardless of the test
                    if len(test_in) > 20 and len(pre_in) > 20: # Chekck there are enought values for statistics
                        if self.similarity == 'KS':
                            sim = 1/(1e-8 + stats.ks_2samp(test_in, pre_in)[0])
                        if self.similarity == 'kl':
                            sim = kl(test_in, pre_in)
                        if self.similarity =='ws':
                            sim = 1 / ( 1e-8 + scipy.stats.wasserstein_distance(test_in, pre_in))
                        if self.similarity == 'euclidian':
                            sim = np.sum(np.abs(test) + np.abs(pre)) / (np.sum(np.abs(test - pre)) + 1e-8)

                    else: # non sufficient points mark as non similarity
                        sim = -1
                    self._plot_distribution(ind_layer=int(ind_layer),
                                            layer_pretrained=pre_in,
                                            layer_test=test_in,
                                            kernel_num=ind_layer,
                                            method='gram', pvalue=sim, num_plots=num_plots,ax_sub=ax_sub,layer_name=name)
                    if not os.path.exists(self.save_folder + '//dist/'):
                        os.makedirs(self.save_folder + '//dist/')
                else:
                    sim = [-1]
                stats_value.append(sim)
            else:
                stats_value = [-1]
            self.stats_value_per_layer[name] = stats_value.copy()
        plt.savefig(self.save_folder + '//dist/' + '//Layer_name_' +
                    name + '.jpg', dpi=400)
        plt.close()

        #self.stats_value_per_layer[name] = stats_value.copy()
        ### Finished layer loop over kernels :

    def _require_grad_search(self, percent=50, mult_grad_value=1e-3):
        th_value = defaultdict(dict)
        for k, v in self.stats_value_per_layer.items():
            th_value[k] = np.percentile(v, percent)
        dict_model = dict(self.network.named_modules())
        weights_zeroing_name = defaultdict(list)
        for ind , name in enumerate(self.modules_name_list):
            if ind > self.max_layer:
                break
            module = dict_model[name]

            if (len(list(module.children()))) < 2 and np.size(
                    self.stats_value_per_layer[name]) > 1:
                change_activations = np.ones(np.shape(
                    self.stats_value_per_layer[name]))
                if th_value[name] > 0:
                    change_inds = np.where((np.array(self.stats_value_per_layer[
                                                         name]) > th_value[name]) *
                                           (np.array(self.stats_value_per_layer[
                                                         name]) < np.inf))[0]
                    #### Used only for ablation study, where I initialized randomally the weights that aren't similar.
                    change_weights = np.where((np.array(self.stats_value_per_layer[
                                                         name]) < th_value[name]) *
                                           (np.array(self.stats_value_per_layer[
                                                         name]) < np.inf))[0]
                else:
                    change_inds = []
                if len(change_inds) > 0:
                    if hasattr(self,'activations_input_pre' ):
                        self.plot_activation(name_layer=name,
                                             indexes=change_inds,
                                             im_batch=0, save_path=self.save_folder + '/activations/' )
                print('layer: ' + name +
                      '  Similar distributions in activation '
                      'num: ' + str(change_inds))
                change_activations[change_inds] *= mult_grad_value
                for weight in module.parameters():
                    new_shape = np.shape(weight)
                    change_activations = np.reshape(
                        change_activations,
                        (len(change_activations), 1, 1, 1))
                    if len(new_shape) > 2:
                        self.layers_grad_mult[name]['weights'] = {}
                        change_activations = np.reshape(
                            change_activations,
                            (len(change_activations), 1, 1, 1))
                        self.layers_grad_mult[name]['weights'] = np.tile(
                            change_activations, (1, new_shape[1], new_shape[2], new_shape[3]))
                        # Replace the chosen layer with random initialization -> for the ablation study:
                        self.ablation_mode = False
                        if self.ablation_mode:
                            temp_weight = weight.clone()
                            temp_weight = torch.tensor(self.kernel_std[name], device='cuda') .unsqueeze(1).unsqueeze(2).unsqueeze(3) * temp_weight
                            #temp_weight = torch.tensor(self.kernel_mean[name], device='cuda') .unsqueeze(1).unsqueeze(2).unsqueeze(3) + temp_weight

                            #initi_values = torch.nn.init.kaiming_normal(temp_weight.clone())
                            #ones_weights = torch.ones_like(temp_weight.clone())
                            #temp_weight[change_weights] =  initi_values.clone()[change_weights]
                            self.network.state_dict()[name+'.weight'].copy_(temp_weight)
                            #weights_zeroing_name[name+'.weight'] = change_weights

                    else:
                        self.layers_grad_mult[name]['bias'] = {}
                        self.layers_grad_mult[name][
                            'bias'] = change_activations
        import pickle
        with open('weight_zeroing_idx.pickle', 'wb') as handle:
            pickle.dump(weights_zeroing_name, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _require_grad_search_layer(self, percent=25, mult_grad_value=1e-7):
        th_value = np.percentile(list(self.stats_value_per_layer.values()),percent)
        dict_model = dict(self.network.named_modules())
        for ind , name in enumerate(self.modules_name_list):
            if ind > self.max_layer:
                break
            module = dict_model[name]
            if (len(list(module.children()))) < 2 and np.size(
                    self.stats_value_per_layer[name]) > 0:
                change_activations = np.ones(np.shape(
                    self.stats_value_per_layer[name]))
                if th_value > 0:
                    change_inds = np.where((np.array(self.stats_value_per_layer[
                                                         name]) > th_value) *
                                           (np.array(self.stats_value_per_layer[
                                                         name]) < np.inf))[0]
                else:
                    change_inds = []
                if len(change_inds) > 0:
                    self.plot_activation(name_layer=name,
                                         indexes=change_inds,
                                         im_batch=0, save_path=self.save_folder + '/activations/' )
                print('layer: ' + name +
                      '  Similar distributions in activation '
                      'num: ' + str(change_inds))
                change_activations[change_inds] *= mult_grad_value
                for weight in module.parameters():
                    new_shape = np.shape(weight)
                    if len(new_shape) > 2:
                        self.layers_grad_mult[name]['weights'] = {}
                        self.layers_grad_mult[name]['weights'] = mult_grad_value * np.ones_like(weight.cpu()
                                                                                                .detach().numpy())
                    else:
                        self.layers_grad_mult[name]['bias'] = {}
                        self.layers_grad_mult[name][
                            'bias'] = mult_grad_value * np.ones_like(weight.cpu()
                                                                     .detach().numpy())

    def _initialize_parameters(self):
        self.outputs_list = defaultdict(int)
        self.layers_grad_mult = defaultdict(dict)
        self.input_list = defaultdict(int)
        self.stats_value_per_layer = defaultdict(int)
        self.statistic_test = defaultdict(list)
        self.statistic_pretrained = defaultdict(list)
        self.stats_test = defaultdict(int)
        self.pre_trained_outputs = defaultdict(int)
        self.list_grads = []
        self.layers_list_to_change = []
        self.layers_list_to_stay = []
        self.mean_var_tested = []
        self.mean_var_pretrained_data = []
        self.stats_value = []

    def run(self, mode='per_layer'):
        self._initialize_parameters()
        self._prepare_input_tensor()
        self._calc_layers_outputs(batches_num=self.num_batches,mode=mode)
        if mode == 'per_layer':
            self._metric_compare_full_layer()
            self._require_grad_search_layer(percent=self.threshold_percent)
        else:
            #self._metric_compare()
            self._require_grad_search(percent=self.threshold_percent)

