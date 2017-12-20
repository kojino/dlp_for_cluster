import numpy as np
from numpy.linalg import inv

import sys
sys.path.append('../')
from model.utils import rbf_kernel

class LPT:
    '''
    Label propagation for predicting labels for unlabeled nodes.
    Closed form and iterated solutions.
    See mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf for details.
    '''
    def __init__(self):
        return

    def closed(self, labels,
                     weights,
                     labeled_indices,
                     unlabeled_indices):
        '''
        Closed solution of label propagation.
        '''
        # normalize T
        Tnorm = self._tnorm(weights)
        # sort Tnorm by unlabeled/labeld
        Tuu_norm = Tnorm[np.ix_(unlabeled_indices,unlabeled_indices)]
        Tul_norm = Tnorm[np.ix_(unlabeled_indices,labeled_indices)]
        # closed form prediction for unlabeled nodes
        lapliacian = (np.identity(len(Tuu_norm))-Tuu_norm)
        propagated = Tul_norm @ labels[labeled_indices]
        label_predictions = np.linalg.solve(lapliacian, propagated)
        return label_predictions

    def iter(self, X, # input labels
                   weights,
                   is_labeled,
                   num_iter, multiclass=False):
        '''
        Iterated solution of label propagation.
        '''
        # normalize T
        Tnorm = self._tnorm(weights)
        h = X.copy()

        for i in range(num_iter):
            # propagate labels
            if multiclass:
                h = np.dot(Tnorm,h)
            else:
                h = np.dot(h,Tnorm.T)

            # don't update labeled nodes
            h = h * (1-is_labeled) + X * is_labeled

        # only return label predictions
        return h


    def _tnorm(self,weights):
        '''
        Column normalize -> row normalize weights.
        '''
        # row normalize T
        Tnorm = weights / np.sum(weights, axis=1, keepdims=True)
        return Tnorm
