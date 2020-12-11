import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt


class CustomRequireGrad:
    def __init__(self, net, pretrained_data_set, input_test,
                 change_grads=False):
        self.pretrained_data_set = pretrained_data_set
        self.input_test = input_test
        self.change_grads = change_grads
        self.network = net
        self.list_grads = []
        self.outputs_list = defaultdict(int)
        self.input_list = defaultdict(int)
        self.layers_list_to_change = []
        self.layers_list_to_stay = []
        self.mean_var_tested = []
        self.mean_var_pretrained_data = []
        self.statistic_test = defaultdict(list)
        self.statistic_pretrained = defaultdict(list)
        self.p_value = []
        self._prepare_input_tensor()
        self._calc_threshold()  # calculating threshold outputs

    def _prepare_input_tensor(self):
        self.pretrained_iter = map(lambda v: v[0].cuda(),
                                   self.pretrained_data_set)
        self.input_test_iter = map(lambda v: v[0].cuda(), self.input_test)

    def _calc_threshold(self):
        for ind_batch, (input_model, input_test) \
                in enumerate(zip(self.pretrained_iter,
                                 self.input_test_iter)):
            if ind_batch > 10:
                break
            for ind, feature in enumerate(self.network.features):
                output_pre_trained = feature(input_model)
                output_test = feature(input_test)
                self.outputs_list[ind] += \
                    feature(input_model).detach().cpu().numpy() \
                    // self.pretrained_data_set.batch_size
                self.input_list[ind] += \
                    feature(input_test).detach().cpu().numpy()\
                    // self.input_test.batch_size
                self.statistic_test[ind].append(np.abs(np.ravel(
                    (feature(input_test).detach().cpu().numpy())) + 1e-4))
                self.statistic_pretrained[ind].append(np.abs(np.ravel(
                    (feature(input_model).detach().cpu().numpy())) + 1e-4))
                input_model = output_pre_trained
                input_test = output_test

    def _prepare_mean_std_layer(self,layer):
        normal_dist_pre = np.log(layer)
        vals_pre, axis_val_pre = np.histogram(normal_dist_pre, 100)
        normal_dist_pre[normal_dist_pre < -4] = \
            axis_val_pre[10 + np.argmax(vals_pre[10:])]
        mu = np.mean(normal_dist_pre)
        std = np.std(normal_dist_pre)
        return mu, std

    def _distribution_compare(self, test='t'):
        for ind, (layer_test, layer_pretrained) in enumerate(
                zip(self.statistic_test.values(),
                    self.statistic_pretrained.values())):
            num_plots = 9
            if ind % num_plots == 0:
                plt.figure()
            # Assuming log normal dist due to relu :
            plt.subplot(np.sqrt(num_plots), np.sqrt(num_plots),  ind % num_plots +1)
            vals_pre, axis_val_pre = np.histogram(np.log(layer_pretrained), 100)
            plt.plot(axis_val_pre[10:], vals_pre[9:] / np.max(vals_pre[10:]), linewidth=4
                     ,alpha=0.7)
            vals, axis_val = np.histogram(np.log(layer_test), 100)
            plt.plot(axis_val[10:], vals[9:] / np.max(vals[10:]), linewidth=4
                     , alpha=0.7)
            plt.xlim([-5, 3])
            plt.ylim([0, 1+0.1])
            mu_pre, std_pre = self._prepare_mean_std_layer(layer_pretrained)
            #normal_dist_pre = np.log(layer_pretrained)
            #normal_dist_pre[normal_dist_pre < -4] = \
            #    axis_val_pre[10 + np.argmax(vals_pre[10:])]
            #normal_mu_pretrained = np.mean(normal_dist_pre)
            #normal_std_pretrained = np.std(normal_dist_pre)
            mu_test, std_test = self._prepare_mean_std_layer(layer_test)

            #normal_dist_test = np.log(layer_test)
            #normal_dist_test[normal_dist_test < -4] = \
            #    axis_val[10 + np.argmax(vals[10:])]
#
            #normal_mu_test = np.mean(normal_dist_test)
            #normal_std_test = np.std(normal_dist_test)

            test_normal = stats.norm.rvs(loc=mu_test,
                                         scale=std_test, size=1000)
            pretrained_normal = stats.norm.rvs(loc=mu_pre,
                                               scale=std_pre
                                               , size=1000)
            out = stats.ttest_ind(pretrained_normal, test_normal,
                                  equal_var=False)
            plt.title('Layer : ' + str(ind) + 'p: ' + str(np.round(out[1], 2)))
            self.p_value.append(out[1])

    def _require_grad_search(self, th_ratio, mode='stats'):
        for ind, feature in enumerate(self.network.parameters()):
            self.mean_var_tested.append((np.mean(np.abs(self.input_list[ind])),
                                  np.var(np.abs(self.input_list[ind]))))
            self.mean_var_pretrained_data.append((
                np.mean(np.abs(self.outputs_list[ind])),
                                  np.var(np.abs(self.outputs_list[ind]))))
            if (np.mean(np.abs(self.input_list[ind])) >
                    th_ratio * np.mean(np.abs(self.outputs_list[ind]))):
                print('layer: ' + str(ind) + ' change grads')
                self.list_grads.append(False)
                self.layers_list_to_change.append(feature)
            else:
                self.list_grads.append(True)
                self.layers_list_to_stay.append(feature)

    def _require_grad_search(self, th_ratio, mode='stats'):
        for ind, feature in enumerate(self.network.parameters()):
            try:
                if (self.p_value[ind] >
                        0.1):
                    print('layer: ' + str(ind) + ' change grads')
                    self.list_grads.append(False)
                    self.layers_list_to_change.append(feature)
                else:
                    self.list_grads.append(True)
                    self.layers_list_to_stay.append(feature)
            except:
                1

    def _update_require_grads_params(self):
        for param, require in zip(self.network.parameters(), self.list_grads):
            param.requires_grad = require

    def _check_change_grads(self):
        for param, require in zip(self.network.parameters(), self.list_grads):
            if not (param.requires_grad == require):
                print('Not Succeed')
                break
        print('succeed')

    def _zero_parameters(self):
        self.list_grads = []

    def run(self, th_ratio):
        self._distribution_compare()
        self._require_grad_search(th_ratio)
        if self.change_grads:
            self._update_require_grads_params()
            self._check_change_grads()
