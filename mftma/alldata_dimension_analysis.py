'''
Calculates total data dimension in two different ways (explained variance, participation ratio)

Based on original code by SueYeon Chung
'''

import torch
import numpy as np
from sklearn.decomposition import PCA


def alldata_dimension_analysis(XtotT, perc=0.90, device=None):
    '''
    Computes the total data dimension by explained variance and participation ratio

    Args:
        XtotT: Sequence of 2D arrays/tensors of shape (N, P_i) where N is the ambient dimension, and P_i is the number
            of samples for the i_th manifold
        perc: Percentage of explained variance to use.
        device: PyTorch device to use for computations

    Returns:
        Dsvd: Dimension (participation ratio)
        D_expvar: Dimension (explained variance)
        D_feature: Ambient feature dimension
    '''
    # Auto-detect device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors if needed and concatenate all the samples
    if isinstance(XtotT[0], np.ndarray):
        XtotT = [torch.tensor(x, dtype=torch.float32, device=device) for x in XtotT]
    
    X = torch.cat(XtotT, dim=1)
    M = X.shape[0]
    # Subtract the global mean
    X = X - torch.mean(X, dim=1, keepdim=True)
    # Compute the total dimension via participation ratio
    ll0, Dsvd, Usvd = compute_participation_ratio(X)
    # Compute the total dimension via explained variance (convert to numpy for sklearn)
    X_np = X.cpu().numpy()
    D_expvar = compute_dim_expvar(X_np, perc)
    return Dsvd, D_expvar, M
    

def compute_dim_expvar(X, perc):
    '''
    Computes the dimension needed to explain perc of the total variance

    Args:
        X: Input data of shape (N, P) where N is the ambient dimension and P is the total number of points
        perc: Percentage of variance to explain

    Returns:
        D_expvar: Dimension needed to explain perc of the total variance
    '''
    N, M = X.shape
    # Subtract the mean
    X_centered = X - X.mean(axis=1, keepdims=True)
    # Do PCA on the centered data
    pca = PCA()
    pca.fit(X)
    # Compute the number of dimensions required to explain perc of the variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    D_expvar = len([x for x in cumulative_variance if x <= perc])
    return D_expvar


def compute_participation_ratio(X):
    '''
    Computes the participation ratio of the total data.

    Args:
        X: Input tensor of shape (N, P) where N is the ambient dimension and P is the total number of points

    Returns:
        s: Singular values
        D_participation: Participation ratio
        U: U matrix from singular value decompisition
    '''
    # Convert to tensor if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    
    P = X.shape[1]
    # Subtract the mean of the data
    mean = torch.mean(X, dim=1, keepdim=True)
    X_centered = X - mean
    # find the SVD of the centered data
    U, S, V = torch.linalg.svd(X_centered)
    S = S[0:-1]
    # Compute the participation ratio
    ss = torch.square(S)
    square_sum = torch.square(torch.sum(ss))
    sum_square = torch.sum(torch.square(ss))
    D_participation = square_sum/sum_square
    return S.cpu().numpy(), D_participation.item(), U[:, 0:-1].cpu().numpy()
