import glob

import geopandas as gpd
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from libpysal.graph import read_parquet
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from scipy import stats

from sklearn.neighbors import KDTree

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score
from core.cluster_validation import get_linkage_matrix

from fast_hdbscan.cluster_trees import (
    cluster_tree_from_condensed_tree,
    condense_tree,
    extract_eom_clusters,
    extract_leaves,
    get_cluster_label_vector,
    mst_to_linkage_tree,
)
from fast_hdbscan.numba_kdtree import kdtree_to_numba
from sklearn.neighbors import KDTree


def preprocess_clustering_data(X_train, clip, to_drop):
    
    ## drop non-buildings
    X_train = X_train[X_train.index >= 0]

    # drop 'to_drop' columns and spatial lag
    all_drop = []
    for c in to_drop:
        all_drop += X_train.columns[X_train.columns.str.contains(c)].tolist()
    X_train = X_train.drop(all_drop, axis=1)

    # standardise data
    vals = StandardScaler().fit_transform(X_train)
    X_train = pd.DataFrame(vals, columns=X_train.columns, index=X_train.index)
    vals = np.nan_to_num(X_train)
    X_train = pd.DataFrame(vals, columns=X_train.columns, index=X_train.index)

    # drop any columns with no variation
    stats = X_train.describe()
    X_train = X_train.drop(stats.columns[stats.loc['std'] == 0], axis=1)

    #optionally clip the data
    if clip is not None:
        X_train = X_train.clip(*clip)

    return X_train


def get_tree(training_data, clustering_graph, linkage, metric):

    clusterer = AgglomerativeClustering(linkage=linkage,
                                        connectivity = clustering_graph,
                                        metric=metric,
                                        compute_full_tree=True,
                                        compute_distances=True)
    model = clusterer.fit(training_data)
    linkage_matrix = get_linkage_matrix(model)
    return linkage_matrix


def post_process_clusters(component_buildings_data, component_graph, component_clusters):


    component_clusters = component_clusters.copy()
        
    clrs, counts = np.unique(component_clusters, return_counts=True)
    
    ## assign each group of contiguous noise points to their own cluster
    if -1 in clrs:
        noise = component_buildings_data[component_clusters == -1].index.values
        noise_labels = component_graph.subgraph(noise).component_labels.values + max(clrs) + 1
        component_clusters[component_clusters == -1] = noise_labels
    
    ## assign singletons to median of neighbours
    clrs, counts = np.unique(component_clusters, return_counts=True)
    for c in clrs[counts == 1]:
        bid = component_buildings_data.iloc[np.where(component_clusters == c)].index.values[0]
        mode_cluster = stats.mode(component_clusters[np.where(component_buildings_data.index.isin(component_graph[bid].index))])[0]
        component_clusters[component_clusters == c] = mode_cluster

    return component_clusters


def get_clusters(linkage_matrix, min_cluster_size, eom_clusters=True):

    condensed_tree = condense_tree(linkage_matrix, 
                               min_cluster_size=min_cluster_size)
    cluster_tree = cluster_tree_from_condensed_tree(condensed_tree)

    if eom_clusters:
        selected_clusters = extract_eom_clusters(
            condensed_tree, cluster_tree, allow_single_cluster=False
        )
    else:
        selected_clusters = extract_leaves(
                condensed_tree, allow_single_cluster=False
            )
    return get_cluster_label_vector(condensed_tree, selected_clusters, 0)



def cluster_data(X_train, graph, to_drop, clip, min_cluster_size, linkage, metric):

    # label building input data, could work with empty tess as well
    building_graph = graph.subgraph(graph.unique_ids[graph.unique_ids >= 0])
    labels = building_graph.component_labels
    
    results = {}
    
    for label, group in labels.groupby(labels):
    
        if group.shape[0] <= min_cluster_size:
            component_clusters = np.ones(group.shape[0])
    
        else:
            component_buildings_data = preprocess_clustering_data(X_train.loc[group.index.values], clip=clip, to_drop=to_drop)
            component_graph = building_graph.subgraph(group.index.values)
            ward_tree = get_tree(component_buildings_data, component_graph.transform('B').sparse, linkage, metric)
    
            # # sometimes ward linkage breaks the monotonic increase in the MST
            # # if that happens shift all distances by the max drop
            # # need a loop because several connections might be problematic
            # problem_idxs = np.where(ward_tree[1:, 2] < ward_tree[0:-1, 2])[0]
            # while problem_idxs.shape[0]:
            #     ward_tree[problem_idxs + 1, 2] = ward_tree[problem_idxs, 2] + .01
            #     problem_idxs = np.where(ward_tree[1:, 2] < ward_tree[0:-1, 2])[0]
            # # check if ward tree distances are always increasing
            # assert (ward_tree[1:, 2] >= ward_tree[0:-1, 2]).all()
            
            component_clusters = get_clusters(ward_tree, min_cluster_size, eom_clusters=True)
    
            
            # component_clusters = fcluster(ward_tree, t=80, criterion='distance')
            
            component_clusters = post_process_clusters(component_buildings_data, component_graph, component_clusters)
            
            for c in np.unique(component_clusters):
                # if c == -1: continue
                cluster_graph = component_graph.subgraph(group.index[component_clusters == c].values)
                assert cluster_graph.n_components == 1
        
            # if label ==3: break
        results[label] = component_clusters
    
    label_groups = labels.groupby(labels)
    region_cluster_labels = []
    for label, component_clusters in results.items():
        group = label_groups.get_group(label)
        component_labels = str(label) + '_' + pd.Series(component_clusters.astype(str), 
                                                        index=group.index.values)
        region_cluster_labels.append(component_labels)
    
    region_cluster_labels = pd.concat(region_cluster_labels).sort_index()
    assert (X_train[X_train.index >= 0].index == region_cluster_labels.index).all()
    
    return region_cluster_labels