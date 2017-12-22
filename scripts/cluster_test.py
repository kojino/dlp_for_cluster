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

    true_labels, features, labels, is_labeled, labeled_indices, unlabeled_indices = DataT.DataT.load_mnist(rel_path='../')
    graph = []

    sigma = args.sigma

    solution  = true_labels[unlabeled_indices]

    print('weights')
    num_nodes = len(labels)
    num_classes = true_labels.shape[1]


    num_labeled = len(labeled_indices)
    num_unlabeled = len(unlabeled_indices)
    val_num_labeled = 600
    val_num_unlabeled = 10000
    df = pd.DataFrame(columns=['accuracy','mse','log'])

    np.random.seed(1)
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

    val_weights, val_graph = utils.rbf_kernel(val_features,
                                          s=sigma,
                                          G=[],
                                          k=10)
    val_true_labels = true_labels[val_indices_original]
    val_num_nodes = len(val_indices_original)

    print('data prepared')

    lp = LPT.LPT()
    lp_iter_prediction = lp.iter(val_labels,
                                 val_weights,
                                 val_is_labeled,
                                 num_iter=args.num_iter,multiclass=True)
    lp_iter_onehot = utils.prob_to_one_hot(lp_iter_prediction)

    accuracy = utils.accuracy(val_solution, lp_iter_onehot[val_unlabeled_indices],multiclass=True)
    mse = utils.mse(val_solution, lp_iter_prediction[val_unlabeled_indices],num_classes)
    log = utils.log_loss(val_solution, lp_iter_prediction[val_unlabeled_indices],multiclass=True)

    out = np.hstack([sigma,args.num_iter,accuracy,mse,log])

    import csv
    with open('out.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--num_iter", type=int)
    args = parser.parse_args()
    main(args)
