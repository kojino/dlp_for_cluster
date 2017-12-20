from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import argparse
import sys
sys.path.append('../')
from model import *

def main(args):
    arg_dict = vars(args)
    file_name = args.filename
    if not file_name:
        for key in arg_dict:
            if key != "filename":
                file_name += key + '_' + str(arg_dict[key]) + '--'
        file_name = file_name[:-2]

    if args.data == "cora":
        true_labels, features, graph = DataT.DataT.load_cora(rel_path='../',multiclass=False)
        if not args.graph:
            graph = []

    np.random.seed(args.seed)
    sigma = np.random.uniform(0.2,5)
    theta = np.random.uniform(0.2,5,(1,features.shape[1]))
    print(theta)

    labels, is_labeled, labeled_indices, unlabeled_indices \
    = utils.random_unlabel(true_labels,
                           unlabel_prob=1-float(args.labeled_percentage),
                           hard=False,multiclass=True,seed=args.seed)
    solution  = true_labels[unlabeled_indices]

    weights, graph = utils.rbf_kernel(features,
                                      s=sigma,
                                      G=graph,
                                      percentile=args.threshold_percentage)

    num_nodes = len(labels)
    num_classes = true_labels.shape[1]

    # crossval
    np.random.seed(None)
    num_labeled = len(labeled_indices)
    num_unlabeled = len(unlabeled_indices)
    p=float(args.crossval_p)
    val_num_labeled = int(num_labeled*p)
    val_num_unlabeled = int(num_unlabeled*p)
    m=int(args.crossval_m)
    for i in range(m):
        np.random.seed(args.seed+i)
        val_labeled_indices_original = sorted(np.random.choice(labeled_indices, val_num_labeled))
        val_unlabeled_indices_original = sorted(np.random.choice(unlabeled_indices, val_num_unlabeled))
        print(val_labeled_indices_original)
        val_indices_original = sorted(np.append(val_labeled_indices_original,val_unlabeled_indices_original))
        val_labeled_indices = np.where(np.isin(val_indices_original,val_labeled_indices_original))[0]
        val_unlabeled_indices = np.where(np.isin(val_indices_original,val_unlabeled_indices_original))[0]
        val_labels = labels[val_indices_original]
        val_is_labeled = is_labeled[val_indices_original]
        val_solution = true_labels[val_unlabeled_indices_original]
        val_features = features[val_indices_original]
        val_weights = weights[np.ix_(val_indices_original,val_indices_original)]
        val_graph = graph[np.ix_(val_indices_original,val_indices_original)]
        val_true_labels = true_labels[val_indices_original]
        val_num_nodes = len(val_indices_original)

        val_data, val_validation_data = DataT.DataT.prepare(val_labels,val_labeled_indices,val_true_labels,args.k,num_classes,args.num_samples,val_num_nodes)

        if args.link_function == "rbf":
            dlp_rbf = DeepLP_RBFT.DeepLP_RBFT(val_num_nodes,
                                              num_classes,
                                              val_features,
                                              val_graph,
                                              sigma,
                                              num_iter=args.num_iter,
                                              lr=args.lr,
                                              regularize=args.regularize,
                                              loss_type=args.loss,
                                              graph_sparse=True)
            dlp_rbf.train(val_data,val_validation_data,args.num_epoch)

            out = np.hstack([np.array(dlp_rbf.sigmas).reshape(-1,1),
                             np.array(dlp_rbf.labeled_losses).reshape(-1,1),
                             np.array(dlp_rbf.losses).reshape(-1,1),
                             np.array(dlp_rbf.accuracies).reshape(-1,1),
                             np.array(dlp_rbf.true_losses).reshape(-1,1),
                             np.array(dlp_rbf.true_accuracies).reshape(-1,1)])

            cols = ['sigma','labeled_loss','unlabeled_loss','accuracy','true_unlabeled_loss','true_accuracy']
            df = pd.DataFrame(out, columns=cols)
            df.to_csv(file_name + '--cv_' + str(i) + '.csv')


        if args.link_function == "wrbf":

            dlp_wrbf = DeepLP_WeightedRBFT.DeepLP_WeightedRBFT(val_num_nodes,
                                                             num_classes,
                                                             val_features,
                                                             val_graph,
                                                             sigma,
                                                             theta,
                                                             num_iter=args.num_iter,
                                                             lr=args.lr,
                                                             regularize=args.regularize,
                                                             loss_type=args.loss,
                                                             graph_sparse=True)
            dlp_wrbf.train(val_data,val_validation_data,args.num_epoch)

            out = np.hstack([np.array(dlp_wrbf.thetas),
                             np.array(dlp_wrbf.labeled_losses).reshape(-1,1),
                             np.array(dlp_wrbf.losses).reshape(-1,1),
                             np.array(dlp_wrbf.accuracies).reshape(-1,1),
                             np.array(dlp_wrbf.true_losses).reshape(-1,1),
                             np.array(dlp_wrbf.true_accuracies).reshape(-1,1)])
            cols = []
            for j in range(len(dlp_wrbf.thetas[0])):
                cols.append('theta_'+str(j))
            cols += ['labeled_loss','unlabeled_loss','accuracy','true_unlabeled_loss','true_accuracy']
            df = pd.DataFrame(out, columns=cols)
            df.to_csv(file_name + '--cv_' + str(i) + '.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename",             default='',                 help="name of the output file")

    parser.add_argument("--crossval_p",           default=1,      type=float, help="fraction to be sampled for cv, 1 means no split")
    parser.add_argument("--crossval_m",           default=1,      type=int,   help="number of cv folds")
    parser.add_argument("--data",                 default="cora",             help="dataset")
    parser.add_argument("--graph",                default=0,      type=int,   help="use a given graph if 1, construct a graph if 0")
    parser.add_argument("--k",                    default=1,      type=int,   help="leave-k-out loss")
    parser.add_argument("--labeled_percentage",   default=0.1,    type=float, help="percentage of labeled nodes")
    parser.add_argument("--link_function",        default="rbf",              help="link function g: rbf, wrbf")
    parser.add_argument("--loss",                 default="mse",              help="type of loss function: mse, log")
    parser.add_argument("--lr",                   default=0.1,    type=float, help="learning rate")
    parser.add_argument("--num_epoch",            default=1000,   type=int,   help=" number of epochs to train the network, early stopping is always applied") #
    parser.add_argument("--num_iter",             default=100,    type=int,   help="number of hidden layers, i.e. iterations of lp")
    parser.add_argument("--num_samples",          default=10,     type=int,   help="number of data points for training")
    parser.add_argument("--regularize",           default=0,      type=int,   help="0: no regularization")
    parser.add_argument("--seed",                 default=None,   type=int,   help="seed for random parameter initialization")
    parser.add_argument("--threshold_percentage", default=0,      type=float, help="if graph is constructed, dropping nodes under this percentile")

    args = parser.parse_args()
    main(args)
