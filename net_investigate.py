import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
from functools import partial
from torch.nn import Sequential
import itertools
activation = {}

def hook_fn(name, model, input, output):
    activation[name] = output.detach()

class CustomRequireGrad:
    def __init__(self, net, pretrained_data_set, input_test,
                 change_grads=False):
        self.pretrained_data_set = pretrained_data_set
        self.input_test = input_test
        self.change_grads = change_grads
        self.network = net

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
    def _plot_distribution(ind_layer, layer_pretrained, layer_test, p_val):
        num_plots = 9
        if ind_layer % num_plots == 0:
            plt.figure()
        # Assuming log normal dist due to relu :
        plt.subplot(np.sqrt(num_plots), np.sqrt(num_plots),
                    ind_layer % num_plots + 1)
        vals_pre, axis_val_pre = np.histogram(np.log(layer_pretrained), 100)
        plt.plot(axis_val_pre[10:], vals_pre[9:] / np.max(vals_pre[10:]),
                 linewidth=4, alpha=0.7, label='D')
        vals, axis_val = np.histogram(np.log(layer_test), 100)
        plt.plot(axis_val[10:], vals[9:] / np.max(vals[10:]), linewidth=4,
                 alpha=0.7, label='D2')
        plt.legend()
        plt.xlim([-5, 3])
        plt.ylim([0, 1 + 0.1])
        plt.title('Layer : ' + str(ind_layer) + 'p: ' + str(np.round(p_val, 2)))

    def _prepare_input_tensor(self):
        self.pretrained_iter = map(lambda v: v[0].cuda(),
                                   self.pretrained_data_set)
        self.input_test_iter = map(lambda v: v[0].cuda(), self.input_test)

    def _calc_layers_outputs(self, batches_num=10):

        def get_activation(name):
            def hook(model, input, output):
                try:
                    activation[name] = output.detach()
                except:
                    activation[name] = None
            return hook
        hooks = {}
        for name, module in self.network.named_modules():
            hooks[name] = module.register_forward_hook(get_activation(name))
        for ind_batch, (input_model, input_test) \
                in enumerate(zip(self.pretrained_iter,
                                 self.input_test_iter)):
            if ind_batch > batches_num:
                break
            activation ={}
            self.network(input_model)
            activations_input = activation.copy()
            activation ={}
            self.network(input_test)
            activations_input_test = activation.copy()
            for name, module in self.network.named_modules():
                 if activations_input_test[name] != None:
                    vals_test =np.abs(np.ravel(
                        activations_input_test[name].cpu()) + 1e-4)
                    vals_pre =np.abs(np.ravel(
                        activations_input[name].cpu()) + 1e-4)
                    if len(vals_pre) >20e3:
                        vals_test = vals_test[vals_test >3e-2]
                        vals_test =vals_test[np.random.randint(0, len(vals_test),
                                                               size=1000)]
                        vals_pre = vals_pre[vals_pre >3e-2]
                        vals_pre =vals_pre[np.random.randint(0, len(vals_pre),
                                                             size=1000)]
                    self.statistic_test[name].append(vals_test)
                    self.statistic_pretrained[name].append(vals_pre)

    def _distribution_compare(self, test='t', plot_dist=True):
        for ind, (layer_test, layer_pretrained) in enumerate(
                zip(self.statistic_test.values(),
                    self.statistic_pretrained.values())):
            layer_test = list(itertools.chain.from_iterable(layer_test))
            layer_pretrained = list(itertools.chain.from_iterable(
                layer_pretrained))
            mu_pre, std_pre = self._prepare_mean_std_layer(layer_pretrained)
            mu_test, std_test = self._prepare_mean_std_layer(layer_test)
            if test == 't':
                test_normal = stats.norm.rvs(loc=mu_test, scale=std_test,
                                             size=200)
                pretrained_normal = stats.norm.rvs(loc=mu_pre, scale=std_pre,
                                                   size=200)
                p_value = stats.ttest_ind(pretrained_normal, test_normal,
                                          equal_var=False)[1]
            self.p_value.append(p_value)
            if plot_dist:
                self._plot_distribution(ind_layer=ind,
                                        layer_pretrained=layer_pretrained,
                                        layer_test=layer_test, p_val=p_value)

    def _require_grad_search(self, p_value=0.1):
        for ind,(name, module) in enumerate( self.network.named_modules() ) :

            if (len( list( module.children()) ) ) < 2:
                print(name)
                if ind < len(self.p_value):
                    if self.p_value[ind] > p_value:

                        print('layer: ' + str(ind) + ' change grads')
                        self.list_grads.append(False)
                        for weight in module.parameters():
                            self.layers_list_to_change.append(weight)
                    else:
                        self.list_grads.append(True)
                        for weight in module.parameters():
                            self.layers_list_to_stay.append(weight)


    def _update_require_grads_params(self):
        for param, require in zip(self.network.parameters(), self.list_grads):
            param.requires_grad = require

    def _check_change_grads(self):
        for param, require in zip(self.network.parameters(), self.list_grads):
            if not (param.requires_grad == require):
                print('Not Succeed')
                break
        print('succeed')

    def _initialize_parameters(self):
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

    def run(self, p_value=0.1):
        self._initialize_parameters()
        self._prepare_input_tensor()
        self._calc_layers_outputs(batches_num=2)  # calculating threshold outputs
        self._distribution_compare()
        self._require_grad_search(p_value)
        if self.change_grads:
            self._update_require_grads_params()
            self._check_change_grads()