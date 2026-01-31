# Adapted from MountainSort5 (https://github.com/flatironinstitute/mountainsort5)
# Licensed under Apache-2.0

import numpy as np
from sklearn import decomposition
from isosplit6 import isosplit6
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree
from numpy.typing import NDArray
from typing import Optional

def compute_pca_features(X: NDArray, npca: int) -> NDArray:
    L = X.shape[0]
    D = X.shape[1]
    npca_2 = np.minimum(np.minimum(npca, L), D)
    if L == 0 or D == 0:
        return np.zeros((0, npca_2), dtype=np.float32)
    pca = decomposition.PCA(n_components=npca_2)
    return pca.fit_transform(X)

def isosplit6_subdivision_method(X: NDArray, npca_per_subdivision: int, inds: Optional[NDArray] = None) -> NDArray:
    if inds is not None:
        X_sub = X[inds]
    else:
        X_sub = X
        
    L = X_sub.shape[0]
    if L == 0:
        return np.zeros((0,), dtype=np.int32)
        
    features = compute_pca_features(X_sub, npca=npca_per_subdivision)
    labels = isosplit6(features)
    
    K = int(np.max(labels)) if len(labels) > 0 else 0
    
    if K <= 1:
        return labels
        
    centroids = np.zeros((K, X.shape[1]), dtype=np.float32)
    for k in range(1, K + 1):
        centroids[k - 1] = np.median(X_sub[labels == k], axis=0)
        
    dists = np.sqrt(np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2))
    dists_condensed = squareform(dists)
    
    Z = linkage(dists_condensed, method='single', metric='euclidean')
    clusters0 = cut_tree(Z, n_clusters=2)
    
    cluster_inds_1 = np.where(clusters0 == 0)[0] + 1
    cluster_inds_2 = np.where(clusters0 == 1)[0] + 1
    
    inds1 = np.where(np.isin(labels, cluster_inds_1))[0]
    inds2 = np.where(np.isin(labels, cluster_inds_2))[0]
    
    if inds is not None:
        inds1_b = inds[inds1]
        inds2_b = inds[inds2]
    else:
        inds1_b = inds1
        inds2_b = inds2
        
    labels1 = isosplit6_subdivision_method(X, npca_per_subdivision=npca_per_subdivision, inds=inds1_b)
    labels2 = isosplit6_subdivision_method(X, npca_per_subdivision=npca_per_subdivision, inds=inds2_b)
    
    K1 = int(np.max(labels1))
    ret_labels = np.zeros(L, dtype=np.int32)
    ret_labels[inds1] = labels1
    ret_labels[inds2] = labels2 + K1
    return ret_labels


