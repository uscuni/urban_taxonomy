import glob

import geopandas as gpd
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from libpysal.graph import read_parquet
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from scipy import stats
import shapely

from sklearn.neighbors import KDTree
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score
from fast_hdbscan.cluster_trees import (
    cluster_tree_from_condensed_tree,
    condense_tree,
    extract_eom_clusters,
    extract_leaves,
    get_cluster_label_vector,
    mst_to_linkage_tree,
)
from fast_hdbscan.numba_kdtree import kdtree_to_numba
import datetime
from core.cluster_validation import get_linkage_matrix
from core.generate_context import spatially_weighted_partial_lag


tessellations_dir = '/data/uscuni-ulce/processed_data/tessellations/'
chars_dir = "/data/uscuni-ulce/processed_data/chars/"
graph_dir = "/data/uscuni-ulce/processed_data/neigh_graphs/"
morphotopes_dir = '/data/uscuni-ulce/processed_data/morphotopes/'
regions_datadir = "/data/uscuni-ulce/"

from core.utils import largest_regions

def preprocess_clustering_data(X_train, clip, to_drop):
    '''Data pre-processing before clustering is carried out.'''
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
    '''Carry out AgglomerativeClustering and return the linkage matrix.'''
    clusterer = AgglomerativeClustering(linkage=linkage,
                                        connectivity = clustering_graph,
                                        metric=metric,
                                        compute_full_tree=True,
                                        compute_distances=True)
    model = clusterer.fit(training_data)
    linkage_matrix = get_linkage_matrix(model)
    return linkage_matrix


def post_process_clusters_noise(component_buildings_data, component_graph, component_clusters):
    '''Process noise points and singletons within a set of clusters.'''

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


def post_process_clusters_tightening(group, min_cluster_size, t=7):
    '''Drop cluster parts that are ``t`` standard deviations away from every part of the morphotope'''
    
    if group.name == -1: return pd.Series(np.full(group.shape[0], -1), group.index)

    clusterer = AgglomerativeClustering(linkage='single',
                                    metric='euclidean',
                                    compute_full_tree=True,
                                    compute_distances=True)
    model = clusterer.fit(group.values)
    linkage_matrix = get_linkage_matrix(model)
    clusters = fcluster(linkage_matrix, t=7, criterion='distance')
    
    chars_clusters = pd.Series(clusters).value_counts()
    chars_clusters[chars_clusters < min_cluster_size] = -1
    chars_clusters[chars_clusters >= min_cluster_size] = group.name
    clusters = pd.Series(clusters).map(lambda x: chars_clusters.loc[x]).values
    return pd.Series(clusters, group.index)
    

def get_clusters(linkage_matrix, min_cluster_size, n_samples, eom_clusters=True):
    '''Extract hdbscan cluster types from a linkage matrix.'''
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
    
    return get_cluster_label_vector(condensed_tree, selected_clusters, 0, n_samples)



def cluster_data(X_train, graph, to_drop, clip, min_cluster_size, linkage, metric, eom_clusters=True):
    '''Split the input data into connected components and carry out an agglomerative clustering for each component independently.
    Pre-process the input data, cluster and then carry out post-processing and finally combine all the seperate clusterings into one set of clusters.'''
    
    # label building input data, could work with empty tess as well
    building_graph = graph.subgraph(graph.unique_ids[graph.unique_ids >= 0])
    labels = building_graph.component_labels
    
    results = {}
    
    for label, group in labels.groupby(labels):
    
        if group.shape[0] <= min_cluster_size:
            component_clusters = np.full(group.shape[0], -1)
    
        else:
            component_buildings_data = preprocess_clustering_data(X_train.loc[group.index.values], clip=clip, to_drop=to_drop)
            component_graph = building_graph.subgraph(group.index.values)
            ward_tree = get_tree(component_buildings_data, component_graph.transform('B').sparse, linkage, metric)
    
            # # sometimes ward linkage breaks the monotonic increase in the MST
            # # if that happens shift all distances by the max drop
            # # need a loop because several connections might be problematic
            problem_idxs = np.where(ward_tree[1:, 2] < ward_tree[0:-1, 2])[0]
            while problem_idxs.shape[0]:
                ward_tree[problem_idxs + 1, 2] = ward_tree[problem_idxs, 2] + .01
                problem_idxs = np.where(ward_tree[1:, 2] < ward_tree[0:-1, 2])[0]
            # check if ward tree distances are always increasing
            assert (ward_tree[1:, 2] >= ward_tree[0:-1, 2]).all()
            
            component_clusters = get_clusters(ward_tree, min_cluster_size, component_buildings_data.shape[0], eom_clusters=eom_clusters)
                
           ## post process
            res = component_buildings_data.groupby(component_clusters).apply(post_process_clusters_tightening, min_cluster_size=min_cluster_size)
            if res.shape[0] == 1:
                component_clusters = pd.Series(res.values[0], res.columns)
            else:
                component_clusters = pd.Series(res.values, res.index.get_level_values(1)).loc[component_buildings_data.index].values
            
            # for c in np.unique(component_clusters):
            #     # if c == -1: continue
            #     cluster_graph = component_graph.subgraph(group.index[component_clusters == c].values)
            #     assert cluster_graph.n_components == 1
        
        results[label] = component_clusters

    ### relabel local clusters(0,1,2,0,1) to regional clusters(0_0,0_1,0_2, 0_0,0_1,) etc
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



def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_
    

def process_single_region_morphotopes(region_id):

    print(datetime.datetime.now(), "----Processing ------", region_id)
    X_train = pd.read_parquet(chars_dir + f'primary_chars_{region_id}.parquet')
    graph = read_parquet(graph_dir + f"tessellation_graph_{region_id}.parquet")
    tessellation = gpd.read_parquet(
            tessellations_dir + f"tessellation_{region_id}.parquet"
    )
    
    building_graph = graph.subgraph(graph.unique_ids[graph.unique_ids >= 0])
    labels = building_graph.component_labels


    ### clustering parameters
    min_cluster_size = 100
    
    # spatial_lag = 3
    # kernel='gaussian' 
    # lag_type = '_median'

    lag_type = None
    spatial_lag = 0
    kernel='None'

    clip = None
    to_drop = ['stcSAl','stbOri','stcOri','stbCeA', 
               'ldkAre', 'ldkPer', 'lskCCo', 'lskERI',
               'lskCWA', 'ltkOri', 'ltkWNB', 'likWBB', 'likWCe']
    
    linkage='ward'
    metric='euclidean'
    eom_clusters = False

    print("--------Generating lag----------")
    ## generate lag, filter and attack to data
    
    
    if lag_type is not None:
        centroids = shapely.get_coordinates(tessellation.representative_point())
        lag = spatially_weighted_partial_lag(X_train, graph, centroids, kernel=kernel, k=spatial_lag, n_splits=10, bandwidth=-1)
        lag = lag[[c for c in lag.columns if lag_type in c]]
        clustering_data = X_train.join(lag, how='inner')
    else:
        clustering_data = X_train

    print("--------Generating morphotopes----------")
    # run morphotopes clustering
    region_cluster_labels = cluster_data(clustering_data, graph, to_drop, clip, min_cluster_size, linkage, metric, eom_clusters=eom_clusters)
    region_cluster_labels.to_frame('morphotope_label').to_parquet(morphotopes_dir + f'tessellation_labels_morphotopes_{region_id}_{min_cluster_size}_{spatial_lag}_{lag_type}_{kernel}_{eom_clusters}.pq')

    ## generate morphotopes boundaries
    # clrs_geometry = tessellation.loc[region_cluster_labels.index]
    # clrs_geometry['label'] = region_cluster_labels.values
    # clrs_geometry = clrs_geometry.dissolve('label').simplify(1).to_frame()
    # clrs_geometry.columns = ['geometry']
    # morph_clrs_geometry = clrs_geometry.set_geometry('geometry').reset_index()
    # morph_clrs_geometry.to_parquet(morphotopes_dir + f'shapes_morphotopes_{region_id}_{min_cluster_size}_{spatial_lag}_{lag_type}_{kernel}.pq')

    # generate morphotopes data
    print("--------Generating morphotopes data----------")
    component_data = X_train.loc[region_cluster_labels.index]
    component_data = component_data.groupby(region_cluster_labels.values).agg([percentile(25), 
                                                             'median', 
                                                             percentile(75), 'std', 'mean'] )
    # save sizes for clustering
    component_data[('Size', 'Size')] = X_train.loc[region_cluster_labels.index].groupby(region_cluster_labels.values).size()

    # store morphotopes data
    component_data.to_parquet(morphotopes_dir + f'data_morphotopes_{region_id}_{min_cluster_size}_{spatial_lag}_{lag_type}_{kernel}_{eom_clusters}.pq')


def process_regions(largest):

    region_hulls = gpd.read_parquet(
        regions_datadir + "regions/" + "cadastre_regions_hull.parquet"
    )

    # region_hulls = region_hulls[~region_hulls.index.isin(largest_regions)]
    region_hulls = region_hulls[region_hulls.index == 69333]
    from joblib import Parallel, delayed
    n_jobs = -1
    new = Parallel(n_jobs=n_jobs)(
        delayed(process_single_region_morphotopes)(region_id) for region_id, _ in region_hulls.iterrows()
    )

if __name__ == '__main__':
    process_regions(largest=False)
