import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt

import torch
from scipy.ndimage.filters import gaussian_filter
import scipy

def kl(p, q):
    p = np.abs(np.asarray(p, dtype=np.float) + 1e-9)
    q = np.abs(np.asarray(q, dtype=np.float) + 1e-9)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def smoothed_hist_kl_distance(a, b, nbins=40, sigma=1):
    a = np.clip(a,0, 2000000)
    b = np.clip(b,0, 2000000)

    ahist, bhist = (np.histogram(a, bins=nbins)[0],
                    np.histogram(b, bins=nbins)[0])
    asmooth, bsmooth = (gaussian_filter(ahist, sigma),
                        gaussian_filter(bhist, sigma))
    return kl(asmooth, bsmooth)


class CustomRequireGrad:

    @staticmethod
    def gram_matrix(layer):
        fft_size = int( np.shape(layer)[2]/2)

        fft_out = np.abs(np.fft.fft2(layer))[:,0:fft_size,0:fft_size]
        return fft_out.ravel()


    @staticmethod
    def gram_matrix1(layer):
        val = np.zeros
        for ll in layer:
            val += (ll @ ll.T).ravel()
            val += ll .ravel()
        return val/np.shape(layer)[0]

    @staticmethod
    def _prepare_mean_std_layer(layer):
        normal_dist_pre = np.log(layer)
        vals_pre, axis_val_pre = np.histogram(normal_dist_pre, 100)
        normal_dist_pre[normal_dist_pre < -4] = \
            axis_val_pre[10 + np.argmax(vals_pre[10:])]
        mu = np.mean(normal_dist_pre)
        std = np.std(normal_dist_pre)
        return mu, std

    @staticmethod
    def _concat_func(list_arr):
        init_vec = []
        for vec in list_arr:
            if len(init_vec) == 0:
                init_vec = vec
            else:
                init_vec = np.concatenate([init_vec, vec], axis=1)
        return init_vec


    def _plot_distribution(self, ind_layer, layer_pretrained, layer_test, stats_val=0,
                           method='gram', kernel_num=0,
                           folder_path=
                           '/mnt/dota/dota/Temp/dist/'):
        if method != 'gram':
            num_plots = 9
            # Assuming log normal dist due to relu :
            plt.subplot(np.sqrt(num_plots), np.sqrt(num_plots),
                        ind_layer % num_plots + 1)
            values_pre, axis_val_pre = np.histogram(np.log(layer_pretrained),100)
            plt.plot(axis_val_pre[10:], values_pre[9:] / np.max(values_pre[10:]),
                     linewidth=4, alpha=0.7, label='D')
            values, axis_val = np.histogram(np.log(layer_test), 100)
            plt.plot(axis_val[10:], values[9:] / np.max(values[10:]), linewidth=4,
                     alpha=0.7, label='D2')
            plt.legend()
            plt.xlim([-5, 3])
            plt.ylim([0, 1 + 0.1])
            plt.title('Layer : ' + str(ind_layer) + 'p: ' + str(np.round(
                stats_val, 2)))
        else:
            if self.plot_counter < 24:
                plt.figure(ind_layer, figsize = (20,20))
                plt.subplot(8, 3, self.plot_counter + 1)
                values, axis_val = np.histogram(layer_test, 100)
                plt.plot(axis_val[10:], values[9:] / np.max(values[10:]),
                         linewidth=4,
                         alpha=0.7, label='Dtest')
                minx_1 = np.min(axis_val[10:])
                maxx_1 = np.max(axis_val[10:])

                values, axis_val = np.histogram(layer_pretrained, 100)
                plt.plot(axis_val[10:],
                         values[9:] / np.max(values[10:]),
                         linewidth=4,
                         alpha=0.7, label='Dpre')
                minx_2 = np.min(axis_val[10:])
                maxx_2 = np.max(axis_val[10:])

                min_x = int(np.min([minx_2, minx_1])) - 0.5
                max_x = int(np.max([maxx_1, maxx_2])) + 0.5

                plt.legend()
                plt.xlim([min_x, max_x])
                plt.ylim([0, 1 + 0.1])
                plt.title(' Kernel num : ' + str(
                    kernel_num))
                plt.suptitle('Layer number:' + str(ind_layer))
                self.plot_counter += 1
            else:
                if self.plot_counter == 24:
                    self.plot_counter += 1
                    plt.savefig(folder_path+'/Layer_number_' +
                                str(ind_layer)+'.jpg', dpi=900)
                    plt.close()

    def __init__(self, net, pretrained_data_set, input_test, max_layer=5):
        self.pretrained_data_set = pretrained_data_set
        self.input_test = input_test
        self.network = net
        self.max_layer = max_layer
        self.activation = {}    
        self.batch_size = self.pretrained_data_set.batch_size   

    def plot_activation(self, name_layer, indexes, im_batch=0, save_path='/mnt/dota/dota/Temp/dist/activations/'):
        self.activations_input_pre[name_layer][0][indexes[0]]
        num_kernels = np.size(indexes)
        num_per_axis = int(np.ceil(np.sqrt(num_kernels)))
        fig_pre = plt.figure(1)
        fig_new = plt.figure(2)
        for i, index in enumerate(indexes):
            fig_pre = plt.figure(1)
            plt.subplot(num_per_axis, num_per_axis,
                        i + 1)
            plt.imshow(self.activations_input_pre[name_layer][im_batch][i]
                       .cpu().numpy())
            fig_new = plt.figure(2)
            plt.subplot(num_per_axis, num_per_axis,
                        i + 1)
            plt.imshow(self.activations_input_test[name_layer][im_batch][i]
                       .cpu().numpy())
        fig_pre.savefig(save_path+'/'+name_layer+"_pre.jpg", dpi=900)
        fig_new.savefig(save_path+'/'+name_layer+"_new.jpg", dpi=900)
        fig_new.clf()
        fig_pre.clf()

    def update_grads(self, net):
        for ind, (name, module) in enumerate(net.named_modules()):
            if len(list( module._modules) ) > 2:
                continue
            if ind > self.max_layer:
                break
            if len(list(module.parameters())) > 0:  # weights
                if len(self.layers_grad_mult[name]) > 0:
                    module.weight.grad *= torch.FloatTensor(
                        self.layers_grad_mult[name]['weights']).cuda()
                    module.bias.grad *= torch.FloatTensor(
                        np.squeeze(self.layers_grad_mult[name]['bias'])).cuda()

    def get_activation(self, name):
        def hook(_, __, output):
            try:
                self.activation[name] = output.detach()
            except:
                self.activation[name] = None
        return hook

    def _prepare_input_tensor(self):
        self.pretrained_iter = map(lambda v: v[0].cuda(),
                                   self.pretrained_data_set)
        self.input_test_iter = map(lambda v: v[0].cuda(), self.input_test)

    def _per_kernel_distribution(self,dist_new, values_pre):
        import bm3d
        output_new = np.zeros_like(dist_new)
        output_pre = np.zeros_like(dist_new)
        for batch in range(np.shape(dist_new)[0]):
            output = bm3d.fft2(dist_new[batch]/np.max(dist_new[batch]))
            output2 = bm3d.fft2(values_pre[batch]/np.max(values_pre[batch]))

        output_1_axix2 = bm3d.fft(output2,axis=0)
        output_1_axix = bm3d.fft(output,axis=0)



        aa1 = np.sum(np.abs(output_1_axix[40:,0:10,0:10 ]))
        aa2 = np.sum(np.abs(output_1_axix2[40:,0:10,0:10 ]))
        print(aa1/aa2)


    def _calc_layers_outputs(self, batches_num=10):
        hooks = {}
        for ind, (name, module) in enumerate(self.network.named_modules()):
            if ind > self.max_layer:
                break
            if len(list( module._modules) ) < 2:
                print(ind)
                hooks[name] = module.register_forward_hook(
                    self.get_activation(name))
        for ind_batch, (input_model, input_test) \
                in enumerate(zip(self.pretrained_iter,
                                 self.input_test_iter)):
            if ind_batch > batches_num:
                break
            self.activation = {}
            self.network(input_model)
            self.activations_input_pre = self.activation.copy()
            self.activation = {}
            self.network(input_test)
            self.activations_input_test = self.activation.copy()
            values_gram_test = 0
            values_gram_pre = 0
            for ind, (name, module) in enumerate(self.network.named_modules()):
                if len(list( module._modules) ) > 2:
                    continue
                if ind > self.max_layer:
                    break
                if self.activations_input_test[name] is not None:
                    dist_new = np.abs(
                        self.activations_input_test[name].cpu().numpy() + 1e-4)
                    values_pre = np.abs(
                        self.activations_input_pre[name].cpu().numpy() + 1e-4)

                    dist_new_tot_size_per_channel = None
                    dist_pre_tot_size_per_channel = None
                    if len(np.shape(dist_new)) > 2:
                        
                        dist_new_channel_first = np.transpose(dist_new, [1, 0, 2, 3])
                        dist_new_tot_size_per_channel = np.zeros(((np.shape(dist_new_channel_first)[0],
                                                  np.prod(np.shape(dist_new_channel_first)
                                                          [1:]))))

                        # Required shape per channel:
                        fft_size = int(np.shape(dist_new)[2]/2)

                        values_gram_test = np.zeros((
                            np.shape(dist_new)[1],
                            self.batch_size*fft_size**2))

                        values_gram_pre = np.zeros((
                            np.shape(dist_new)[1],
                            self.batch_size*fft_size**2))
 
                        ## seperating distribution per kernel
                        # -> [#channels, #BatchSIze,#activation size (#,#) ] 
                        values_pre1 = np.transpose(values_pre, [1, 0, 2, 3])
                        dist_pre_tot_size_per_channel = np.zeros(((np.shape(values_pre1)[0],
                                                 np.prod(np.shape(values_pre1)
                                                         [1:]))))

                        for ll in range(np.shape(dist_new_channel_first)[0]):
                            dist_new_tot_size_per_channel[ll] = np.ravel(dist_new_channel_first[ll])
                            dist_pre_tot_size_per_channel[ll] = np.ravel(values_pre1[ll])
                            values_gram_test[ll] = self.gram_matrix(
                                dist_new_channel_first[ll])

                            values_gram_pre[ll] = self.gram_matrix(
                                values_pre1[ll])
                        if len(dist_pre_tot_size_per_channel[0]) > 200e3:
                            dist_new_tot_size_per_channel = dist_new_tot_size_per_channel[:, np.random.randint(
                                0, len(dist_pre_tot_size_per_channel[0]), size=1000)]
                            dist_pre_tot_size_per_channel = dist_pre_tot_size_per_channel[:, np.random.randint(
                                0, len(dist_pre_tot_size_per_channel[0]), size=1000)]
                    if len(np.shape(self.gram_test[name])) == 0:
                        self.gram_test[name] = values_gram_test
                        self.gram_pre[name] = values_gram_pre
                    else:
                        self.gram_test[name] = np.concatenate(
                            [self.gram_test[name], (np.abs(values_gram_test))], axis=1)
                        self.gram_pre[name] = np.concatenate(
                            [self.gram_pre[name],(np.abs(values_gram_pre))], axis=1)

                    #self.gram_pre[name] += values_gram_pre

                    self.statistic_test[name].append(dist_new_tot_size_per_channel)
                    self.statistic_pretrained[name].append(dist_pre_tot_size_per_channel)

    def _distribution_compare(self, test='kl', plot_dist=False):
        for layer_test, layer_pretrained in (
                zip(self.statistic_test.items(),
                    self.statistic_pretrained.items())):
            stats_value = []
            if not np.sum(layer_pretrained[1][0]) is None:  # has grads layers
                layer_test_concat = self._concat_func(layer_test[1])
                layer_pretrained_concat = self._concat_func(layer_pretrained[1])
                for layer_test_run, layer_pretrained_run in zip(
                        layer_test_concat, layer_pretrained_concat):
                    if test == 't':
                        mu_pre, std_pre = self._prepare_mean_std_layer(
                            layer_pretrained)
                        mu_test, std_test = self._prepare_mean_std_layer(
                            layer_test)
                        test_normal = stats.norm.rvs(
                            loc=mu_test, scale=std_test, size=200)
                        pretrained_normal = stats.norm.rvs(
                            loc=mu_pre, scale=std_pre, size=200)
                        stats_value = stats.ttest_ind(
                            pretrained_normal, test_normal, equal_var=False)[1]
                    if test == 'kl':
                        norm_test = np.log(layer_test_run)
                        norm_pre = np.log(layer_pretrained_run)
                        kl_value = 1 / (1e-9 + smoothed_hist_kl_distance(
                            norm_test, norm_pre, nbins=10, sigma=1))
                        stats_value.append(kl_value)
                self.stats_value_per_layer[layer_test[0]] = stats_value

                if plot_dist:
                    self._plot_distribution(method='kl',
                        ind_layer=1,
                        layer_pretrained=layer_pretrained[1],
                        layer_test=layer_test[1], stats_val=stats_value[-1])
            else:
                self.stats_value_per_layer[layer_test[0]] = 0

    def _metric_compare(self):
        for ind, (name, module) in enumerate(self.network.named_modules()):
            if len(list( module._modules) ) > 2:
                continue
            stats_value = []
            if ind > self.max_layer:
                break
            self.plot_counter = 0
            if np.size(self.gram_test[name]) > 1:  # check if has values
                for ind_inside_layer, (test, pre) in enumerate(zip(
                        self.gram_test[name], self.gram_pre[name])):
                    if np.size(self.activations_input_pre[name][0]
                               [ind_inside_layer].cpu().numpy()) > 1000:
                        act_power_pre = np.mean(np.abs(pre))
                        act_power_test = np.mean(np.abs(test))
                        act_power_pre = np.where(act_power_pre > 1,
                                                 act_power_pre, 0)
                        act_power_test = np.where(act_power_test > 1,
                                                  act_power_test, 0)
                        stats_value.append(
                            (act_power_test)
                            / (1+smoothed_hist_kl_distance(test, pre)))
                        plot = True
                        if plot:
                            self._plot_distribution(ind_layer=ind,
                                                    layer_pretrained=pre,
                                                    layer_test=test,
                                                    kernel_num=ind_inside_layer, method='gram')
                    else:
                        stats_value = [1e-14]


            else:
                stats_value = [1e-14]
                self.stats_value_per_layer[name] = stats_value.copy()
            self.stats_value_per_layer[name] = stats_value.copy()



    def _require_grad_search(self, percent=70):
        th_value = np.percentile([np.percentile(val, percent)
                              for key, val in
                              self.stats_value_per_layer.items()],percent)
        th_value = th_value*1
        for ind, (name, module) in enumerate(self.network.named_modules()):
            if len(list( module._modules) ) > 2:
                continue
            if ind > self.max_layer:
                break
            if (len(list(module.children()))) < 2 and np.size(
                    self.stats_value_per_layer[name]) > 1:
                if ind < len(self.stats_value_per_layer):
                    change_activations = np.ones(np.shape(
                        self.stats_value_per_layer[name]))
                    change_inds = np.where((np.array(self.stats_value_per_layer[
                                                         name]) > th_value) *
                                           (np.array(self.stats_value_per_layer[
                                                         name]) < np.inf))[0]
                    if len(change_inds) > 1 and ind < 14:
                        path_save = '/mnt/dota/dota/Temp/dist/activations/'
                        self.plot_activation(name_layer=name,
                                             indexes=change_inds,
                                             im_batch=0, save_path=path_save)
                    print('layer: ' + name +
                          '  Similar distributions in activation '
                          'num: ' + str(change_inds))
                    change_activations[change_inds] *= 0.31
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
                                change_activations, (
                                    1, new_shape[1], new_shape[2],
                                    new_shape[3]))
                        else:
                            self.layers_grad_mult[name]['bias'] = {}
                            self.layers_grad_mult[name][
                                'bias'] = change_activations
                else:
                    for _ in module.parameters():
                        self.layers_grad_mult[ind] = None

    def _initialize_parameters(self):
        self.outputs_list = defaultdict(int)
        self.layers_grad_mult = defaultdict(dict)
        self.input_list = defaultdict(int)
        self.stats_value_per_layer = defaultdict(int)
        self.statistic_test = defaultdict(list)
        self.statistic_pretrained = defaultdict(list)
        self.gram_test = defaultdict(int)
        self.gram_pre = defaultdict(int)
        self.list_grads = []
        self.layers_list_to_change = []
        self.layers_list_to_stay = []
        self.mean_var_tested = []
        self.mean_var_pretrained_data = []
        self.stats_value = []

    def run(self, layer_eval_method='gram'):
        self._initialize_parameters()
        self._prepare_input_tensor()
        self._calc_layers_outputs(batches_num=50)
        if layer_eval_method == 'gram':
            self._metric_compare()
        if layer_eval_method == 'distribution':
            self._distribution_compare()
        self._require_grad_search()
