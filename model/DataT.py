import sklearn
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from model.utils import indices_to_vec, array_to_one_hot

class DataT:
    '''
    Load datasets.
    '''
    def load_iris():
        # load iris data
        iris   = sklearn.datasets.load_iris()
        data   = iris["data"]
        labels = iris["target"]

        # get label 0 and 1, and corresponding features
        true_labels = labels[labels < 2]
        features = data[np.where(labels < 2)]

        return true_labels, features, []

    def load_cora(multiclass=False,rel_path=''):
        # load cora data
        if multiclass:
            nodes = pd.read_csv(rel_path+'cora/selected_contents_multiclass.csv',index_col=0,)
        else:
            nodes = pd.read_csv(rel_path+'cora/selected_contents.csv',index_col=0,)
        graph = np.loadtxt(rel_path+'cora/graph.csv',delimiter=',')
        id_    = np.array(nodes.index)

        # get label 0 and 1, and corresponding features
        true_labels = np.array(nodes['label'])
        features   = nodes.loc[:,'feature_0':].as_matrix()
        true_labels = array_to_one_hot(true_labels)

        return true_labels, features, graph

    def prepare(labels,labeled_indices,true_labels,k,num_classes,num_samples,num_nodes):
        num_nodes = len(labels)
        X_ = np.tile(labels.T,num_samples).reshape(num_classes,num_samples,num_nodes)
        y_ = np.tile(true_labels.T,1).reshape(num_classes,1,num_nodes)
        true_labeled_ = np.repeat(indices_to_vec(labeled_indices,num_nodes).reshape(1,num_nodes),num_classes,axis=0).reshape((num_classes,1,len(true_labels)))
        labeled_ = np.repeat(true_labeled_,num_samples,axis=1)
        masked_ = np.zeros((num_classes,num_samples,num_nodes))

        validation_data = {
            'X': labels.T.reshape(num_classes,1,num_nodes),
            'y': y_,
            'labeled': true_labeled_,
            'true_labeled': true_labeled_, # this will not be used
            'masked': masked_  # this will not be used
        }

        for i in range(num_samples):
            indices_to_mask = np.random.choice(labeled_indices, k)
            X_[:,i,indices_to_mask] = 1/num_classes
            labeled_[:,i,indices_to_mask] = 0
            masked_[:,i,indices_to_mask] = 1

        data = {
            'X': X_,
            'y': y_,
            'labeled': labeled_,
            'true_labeled': true_labeled_,
            'masked': masked_
        }

        return data, validation_data
