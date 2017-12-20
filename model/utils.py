from scipy.spatial.distance import pdist, squareform
import numpy as np
import scipy as sp
from sklearn import datasets

def random_unlabel(true_labels,unlabel_prob=0.1,hard=False,seed=None,multiclass=False):
    '''
    randomly unlabel nodes based on unlabel probability
    '''
    np.random.seed(seed)
    labels = true_labels.copy().astype(float)
    n = len(labels)
    is_labeled = np.zeros(n)
    is_labeled.fill(True)

    # unlabeled_indices = np.arange(int(n * unlabel_prob))
    unlabeled_indices = np.array(sorted(np.random.choice(n, int(n * unlabel_prob), replace=False)))
    labeled_indices = np.delete(np.arange(n),unlabeled_indices)
    is_labeled.ravel()[unlabeled_indices] = False

    if hard:
        print('hard initialization')
        labels[unlabeled_indices] = 1 - labels[unlabeled_indices]
    else:
        labels[unlabeled_indices] = 0.5

    if multiclass:
        k = labels.shape[1]
        labels[unlabeled_indices] = 1/k

    return labels, is_labeled.reshape(is_labeled.shape[0],1), labeled_indices, unlabeled_indices

def rbf_kernel(X,s=1,G=[],percentile=3):
    '''
    Use RBF kernel to calculate the weights of edges.
    If given a graph G, drop edges not in G.
    If not, drop edges that are not in the top percentile.
    '''
    # use rbf kernel to estimate weights
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    K = sp.exp(-pairwise_dists ** 2 / s ** 2)
    # K = features @ features.T/

    if G == []:
        print('graph constructed')
        threshold = np.percentile(K,percentile)
        np.fill_diagonal(K, 0)

        Knew = K * (K > threshold)
        argmax = np.argmax(K,axis=1)
        Knew[np.arange(len(K)), argmax] = K[np.arange(len(K)), argmax]
        Knew[argmax,np.arange(len(K))] = K[np.arange(len(K)), argmax]
        K = Knew
        G = K > 0
    else:
        print('graph given')
        K = K * G

    return K, G

def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None,
                     use_columns=False, xticks=None, colormap=None,
                     **kwds):
    '''
    Plot pararrel coordinates.
    This function is inherited from matplotlib but is
    modified to accept continuous values.
    '''
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    n = len(frame)
    class_col = frame[class_column]
    class_min = np.amin(class_col)
    class_max = np.amax(class_col)

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends = set([])

    ncols = len(df.columns)

    # determine values to use for xticks
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError('Columns must be numeric to be used as xticks')
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError('xticks specified must be numeric')
        elif len(xticks) != ncols:
            raise ValueError('Length of xticks must match number of columns')
        x = xticks
    else:
        x = range(ncols)

    fig = plt.figure()
    ax = plt.gca()

    Colorm = plt.get_cmap(colormap)

    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), **kwds)

    for i in x:
        ax.axvline(i, linewidth=1, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right')
    ax.grid()

    bounds = np.linspace(class_min,class_max,10)
    cax,_ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

    return fig

def accuracy(y,yhat,multiclass=False):
    if multiclass:
        return np.mean((yhat == y).all(axis=1))
    else:
        return np.mean((yhat == y))

def log_loss(y,yhat,multiclass=False,eps=0.0000000001):
    yhat[yhat < eps] = eps
    yhat[yhat > 1-eps] = 1-eps
    log_yhat = np.log(yhat)
    if multiclass:
        return np.sum((y * log_yhat)/np.count_nonzero(y)) * (-1)
    else:
        return np.mean((y * np.log(yhat) + (1-y) * np.log(1-yhat)))

def mse(y,yhat):
    return np.mean((yhat - y)**2)

def objective(Ly,Uy_lp,W):
    n = len(Ly) + len(Uy_lp)
    labels = np.hstack((Ly,Uy_lp)).reshape(n,1)
    row, col = np.diag_indices_from(W)
    D = np.identity(W.shape[0])
    D[row,col] = np.sum(W,axis=0)
    return (labels.T @ (D-W) @ labels)[0][0]

def prob_to_one_hot(prob):
    return (prob == prob.max(axis=1)[:,None]).astype(int)

def array_to_one_hot(vec):
    num_nodes = len(vec)
    num_classes = len(set(vec))
    res = np.zeros((num_nodes, num_classes))
    res[np.arange(num_nodes), vec.astype(int)] = 1
    return res

def indices_to_vec(labeled_indices, num_nodes):
    res = np.zeros(num_nodes)
    res[labeled_indices] = 1
    return res

def accuracy_mult(sol,pred):
    match = (sol == pred).all(axis=1)
    return np.sum(match) / len(match)
