import numpy as np

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
