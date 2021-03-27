import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import os
import torch
import scipy

def kl(p, q):
    p = np.abs(np.asarray(p, dtype=np.float) + 1e-15)
    q = np.abs(np.asarray(q, dtype=np.float) + 1e-15)
    p = p/np.sum(p)
    q = q/np.sum(q)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def smoothed_hist_kl_distance(a, b, nbins=20):
    ahist, bhist = (np.histogram(a, bins=nbins)[0],
                    np.histogram(b, bins=nbins)[0])
    return kl(ahist, bhist)

class CustomRequireGrad:

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

    class PriorPreprocess:
        def __init__(self,  method='fft', shape_act=None,**kwargs):
            self.__dict__.update(kwargs)
            self.method = method
            self.shape_act = shape_act

        def run_prior_transformation(self, layer):
            if self.method == 'fft':
                return self.fft_distribution(layer)
            if self.method == 'gram':
                return self.gram_matrix1(layer)
            if self.method == 'linear':
                return np.ravel(layer)

        def initialize_list(self):
            if self.method == 'fft':
                self.fft_size = int(self.shape_act[2]/2)
                values_post_test = np.zeros((
                    self.shape_act[1],
                    1 * self.fft_size ** 2))
                values_post_pre = np.zeros((
                    self.shape_act[1],
                    1 * self.fft_size ** 2))
                return values_post_test, values_post_pre
            if self.method == 'linear':
                values_post_pre = np.zeros(((self.shape_act[1], self.shape_act[0] * np.prod(self.shape_act[2:]))))
                values_post_test = np.zeros(((self.shape_act[1], self.shape_act[0] * np.prod(self.shape_act[2:]))))
                return values_post_test, values_post_pre

        def fft_distribution(self, layer):
            fft_out = np.abs(np.fft.fft2(layer))[:, 0:self.fft_size, 0:self.fft_size]
            mean_fft = np.mean(np.log(np.abs(fft_out+1e-8))  ,axis=0)
            return mean_fft.ravel()

        @staticmethod
        def gram_matrix1(layer):
            val = np.zeros
            for ll in layer:
                val += (ll @ ll.T).ravel()
                val += ll.ravel()
            return val / np.shape(layer)[0]

    def _plot_distribution(self, ind_layer, layer_pretrained, layer_test,
                           stats_val=0,method='gram', kernel_num=0,save_path=
                           './images_dist/', pvalue=0, num_plots = 20, ax_sub=None):

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
            plt.suptitle('Layer number:' + str(ind_layer))

            self.plot_counter +=1

    def __init__(self, net, pretrained_data_set, input_test,
                 dist_processing_method='fft', batches_num=10, percent=70,
                 deepest_layer=11,similarity = 'ws', save_folder='/home/yuvalbe/bpct2/bpcpt/Statistics_pretrained'):
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

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig_pre.savefig(save_path+'/'+name_layer+"_pre.jpg", dpi=900)
        fig_new.savefig(save_path+'/'+name_layer+"_new.jpg", dpi=900)
        fig_new.clf()
        fig_pre.clf()
        plt.close('all')

    def update_grads(self, net):
        dict_model = dict(net.named_modules())
        for name in self.modules_name_list:
            module = dict_model[name]
            if len(list(module.parameters())) > 0:  # weights
                if len(self.layers_grad_mult[name]) > 0:
                    module.weight.grad *= torch.FloatTensor(
                        self.layers_grad_mult[name]['weights']).to(self.device)
                    module.bias.grad *= torch.FloatTensor(
                        np.squeeze(self.layers_grad_mult[name]['bias'])).to(self.device)

    def get_activation(self, name):
        def hook(_, __, output):
            try:
                self.activation[name] = output.detach().cpu().numpy().copy()
                if (np.sum(np.abs(output.detach().cpu().numpy()) <=1e-8  ) > 10):
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
            if len(list(module._modules)) < 2 and\
                    'weight' in module._parameters and 'Conv'  in module._get_name() :  # Skip module modules
                self.modules_name_list.append(name)
                hooks[name] = module.register_forward_hook(
                    self.get_activation(name))

    def _calc_layers_outputs(self, batches_num=10):
        ## Hook to each relavant module
        self._hook_assign_module()

        for ind_batch, (input_model, input_test) \
                in enumerate(zip(self.pretrained_iter,
                                 self.input_test_iter)):
            if ind_batch > batches_num:
                break
            self.activation = {} # clear all activations every batch

            bp_flage =False # Only for bpc use :
            if bp_flage == True:
                bp = input_model[1]
                bp = bp.to(self.device, dtype=torch.float)
                bp_test = input_test[1]
                bp_test = bp_test.to(self.device, dtype=torch.float)
            else:
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
                    dist_new_tot_size_per_ch = None
                    dist_pre_tot_size_per_ch = None
                    if len(np.shape(dist_new)) > 2: 
                        dist_new_channel_first = np.transpose(dist_new, [1, 0, 2, 3])
                        dist_new_tot_size_per_ch = np.zeros(((np.shape(dist_new_channel_first)[0],
                                                  np.prod(np.shape(dist_new_channel_first)
                                                          [1:]))))
                        # Required shape per channel:
                        transform_prior = self.PriorPreprocess(method='linear', shape_act=np.shape(dist_new), **self.__dict__)
                        values_post_test, values_post_pre = transform_prior.initialize_list()

                        ## seperating distribution per kernel
                        # -> [#channels, #BatchSIze,#activation size (#,#) ]
                        values_pre1 = np.transpose(values_pre, [1, 0, 2, 3])
                        dist_pre_tot_size_per_ch = np.zeros(((np.shape(values_pre1)[0],
                                                 np.prod(np.shape(values_pre1)
                                                         [1:]))))

                        ## Aggragating along in a dict for each  channel:
                        for ll in range(np.shape(dist_new_channel_first)[0]):
                            dist_new_tot_size_per_ch[ll] =\
                                np.ravel(dist_new_channel_first[ll])
                            dist_pre_tot_size_per_ch[ll] = \
                                np.ravel(values_pre1[ll])
                            values_post_test[ll] = transform_prior.run_prior_transformation(
                                dist_new_channel_first[ll])
                            values_post_pre[ll] = transform_prior.run_prior_transformation(
                                values_pre1[ll])
                        if len(dist_pre_tot_size_per_ch[0]) > 200e3:
                            dist_new_tot_size_per_ch = dist_new_tot_size_per_ch[
                                                       :, np.random.randint(
                                0, len(dist_pre_tot_size_per_ch[0]), size=5000)]
                            dist_pre_tot_size_per_ch = dist_pre_tot_size_per_ch[:, np.random.randint(
                                0, len(dist_pre_tot_size_per_ch[0]), size=5000)]
                    # Concatanating the data along the different batches :
                    if len(np.shape(self.gram_test[name])) == 0:
                        self.gram_test[name] = values_post_test
                        self.gram_pre[name] = values_post_pre
                    else:
                        clipped_log_gram = np.clip(
                            (np.abs(values_post_test)), -2e6, 2e6)
                        self.gram_test[name] = np.concatenate(
                            [self.gram_test[name],clipped_log_gram], axis=1)
                        self.gram_pre[name] = np.concatenate(
                            [self.gram_pre[name], np.clip(  (np.abs(values_post_pre)), -2e6 ,2e6 )], axis=1)
                    self.statistic_test[name].append(dist_new_tot_size_per_ch)
                    self.statistic_pretrained[name].append(dist_pre_tot_size_per_ch)

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
                        layer_test=layer_test[1],save_path='/home/yuvalbe/bpct2/bpcpt/Statistics_pretrained/dist/', stats_val=stats_value[-1],ax_sub=ax_sub)
            else:
                self.stats_value_per_layer[layer_test[0]] = 0

    def _metric_compare(self):
        for ind_layer, name in enumerate(self.modules_name_list):
            if ind_layer > self.max_layer:
                break
            num_plots = np.shape(self.gram_pre[name])[0]
            fig = plt.figure(ind_layer, figsize=(20, 20))
            ax_sub = fig.subplots(int(np.ceil(np.sqrt(num_plots))), int(np.ceil(np.sqrt(num_plots))))
            ax_sub = ax_sub.ravel()
            stats_value = []
            self.plot_counter = 0
            if np.size(self.gram_test[name]) > 1:  # check if has values
                for ind_inside_layer, (test, pre) in enumerate(zip(
                        self.gram_test[name], self.gram_pre[name])):
                    if np.size(self.activations_input_pre[name][0][ind_inside_layer]) > 20:
                        ## Convert from log normal to normal distribution. (assumtion been made)
                        test_in = np.log(np.abs(test[test > 1e-7]))
                        pre_in =  np.log(np.abs(pre[pre > 1e-7]))
                        ## Similarity units! regardless of the test
                        if len(test_in) > 20 and len(pre_in) > 20: # Chekck there are enought values for statistics
                            if self.similarity == 'KS':
                                sim = 1/stats.ks_2samp(test_in, pre_in)[0]
                            if self.similarity == 'kl':
                                sim = kl(test_in, pre_in)
                            if self.similarity =='ws':
                                sim = 1 / scipy.stats.wasserstein_distance(test_in, pre_in)
                            if self.similarity == 'euclidian':
                                sim = np.sum(np.abs(test) + np.abs(pre)) / (np.sum(np.abs(test - pre)) + 1e-8)

                        else: # non sufficient points mark as non similarity
                            sim = -1
                        self._plot_distribution(ind_layer=int(ind_layer),
                                                layer_pretrained=pre_in,
                                                layer_test=test_in,
                                                kernel_num=ind_inside_layer,
                                                method='gram', save_path=self.save_folder + '/dist/', pvalue=sim, num_plots=num_plots,ax_sub=ax_sub)
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
            plt.savefig(self.save_folder + '//dist/' + '//Layer_number_' +
                        str(ind_layer) + '.jpg', dpi=400)
            plt.close()

    def _require_grad_search(self, percent=25, mult_grad_value=1e-3):
        th_value = [np.percentile(val, percent)for key, val in
                                  self.stats_value_per_layer.items()]
        dict_model = dict(self.network.named_modules())
        for ind , name in enumerate(self.modules_name_list):
            if ind > self.max_layer:
                break
            module = dict_model[name]
            if (len(list(module.children()))) < 2 and np.size(
                    self.stats_value_per_layer[name]) > 1:
                change_activations = np.ones(np.shape(
                    self.stats_value_per_layer[name]))
                if th_value[ind] > 0:
                    change_inds = np.where((np.array(self.stats_value_per_layer[
                                                         name]) > th_value[ind]) *
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
                    else:
                        self.layers_grad_mult[name]['bias'] = {}
                        self.layers_grad_mult[name][
                            'bias'] = change_activations

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

    def run(self):
        self._initialize_parameters()
        self._prepare_input_tensor()
        self._calc_layers_outputs(batches_num=self.num_batches)
        if self.dist_processing_method == 'fft':
            self._metric_compare()
        if self.dist_processing_method == 'distribution':
            self._distribution_compare()
        self._require_grad_search(percent=self.threshold_percent)
