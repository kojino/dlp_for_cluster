from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from model.DeepLP import DeepLP

class DeepLP_RBF(DeepLP):
    '''
    Deep label propagation with scalor divisor as a parameter.
    See our paper for details.
    '''
    def __init__(self, num_nodes,
                       features,
                       graph,
                       sigma,
                       num_iter=100,
                       loss_type='mse',          # 'mse' or 'log'
                       lr=0.1,
                       regularize=0,       # add L1 regularization to loss
                       graph_sparse=False, # make the graph sparse
                       print_freq=10,      # print frequency when training
                       multi_class=False): # implementation for multiclass

        self.phi     = tf.constant(features, dtype=tf.float32)
        self.graph   = tf.constant(graph, dtype=tf.float32)
        self.sigma   = tf.Variable(sigma, dtype=tf.float32)
        self.weights = self._init_weights(self.phi, self.graph, self.sigma)

        self._build_graph(num_iter,
                          num_nodes,
                          loss_type,
                          lr,
                          regularize,
                          graph_sparse,
                          print_freq,
                          multi_class)

    def _save_params(self,epoch,data,n):
        sigmab = self._get_val(self.sigma)
        self.sigmas.append(sigmab)
        if epoch % 1 == 0:
            print("sigma:",sigmab)

    def _init_weights(self, phi, G, sigma):
        r = tf.reduce_sum(phi*phi, 1)
        r = tf.reshape(r, [-1, 1])
        D = tf.cast(r - 2*tf.matmul(phi, tf.transpose(phi)) + tf.transpose(r),tf.float32)
        W = G * tf.exp(-tf.divide(D, sigma ** 2))
        return W

    def train(self,data,full_data,epochs):
        self.sigmas = []
        super().train(data,full_data,epochs)

    def _plot_params(self):
        plt.plot(self.sigmas)
        plt.title("parameter")
        plt.show()
