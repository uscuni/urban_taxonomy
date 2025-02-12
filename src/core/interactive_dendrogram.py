import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

def get_dendrogram_scatter_data(linkage_matrix):
    '''Add x and y coordinates to the linkage matrix, in order to plot it as a scatter plot.
       Every point is a cluster merger and has x,y,group and order attributes. 
       The x coordinate and item order is calculated by scipy dendrogram, the y is the height of the connection.
       The group and order fields show how to draw the links between the x and y points.
       The data is doubled, since every point can be a part of only one group, but most points are both a parent and child.
       Therefore, during the plotting two points are plotted one over the other.
       '''
    R = dendrogram(linkage_matrix, no_plot=True)
    xs = np.array(R['icoord'])[:, 1:3].sum(axis=1)
    ys = np.array(R['dcoord'])[:, 1]

    ### fast way to find out where in ii, each node is equivalent to np.nonzero(ii == j)
    ii = np.argsort(ys)
    sorter = np.argsort(ii)
    positions = sorter[np.searchsorted(ii, np.arange(0, len(R['dcoord'])), sorter=sorter)]
    nodes = linkage_matrix[positions, ] 

    ## group for lines
    n = linkage_matrix.shape[0] + 1
    line_groups = np.zeros(linkage_matrix.shape[0])
    line_orders = np.zeros(linkage_matrix.shape[0])
    for i in range(linkage_matrix.shape[0]-1, 0, -1):
        
        if line_groups[i] != 0: continue
            
        child, parent = linkage_matrix[i, 0].astype(int), linkage_matrix[i, 1].astype(int)
        line_groups[i] = i
        line_orders[i] = 2
        if child >= n:
            line_groups[child - n] = i
            line_orders[child - n] = 3
        if parent >= n:
            line_groups[parent - n] = i
            line_orders[parent - n] = 1
    
    scatter_df = pd.DataFrame(nodes, columns=['child', 'parent', 'dist', 'size'])
    scatter_df['x'] = xs
    scatter_df['y'] = ys
    scatter_df['group1'] = line_groups[positions].astype('str')
    scatter_df['order1'] = line_orders[positions].astype(int)

    ## double the data, in order to plot lines
    nodes = linkage_matrix[positions, ] 
    line_groups = np.zeros(linkage_matrix.shape[0])
    line_orders = np.zeros(linkage_matrix.shape[0])
    for i in range(linkage_matrix.shape[0]-2, 0, -1):
        
        if line_groups[i] != 0: continue
            
        child, parent = linkage_matrix[i, 0].astype(int), linkage_matrix[i, 1].astype(int)
        line_groups[i] = i
        line_orders[i] = 2
        if child >= n:
            line_groups[child - n] = i
            line_orders[child - n] = 3
        if parent >= n:
            line_groups[parent - n] = i
            line_orders[parent - n] = 1
    
    scatter_df2 = pd.DataFrame(nodes, columns=['child', 'parent', 'dist', 'size'])
    scatter_df2['x'] = xs
    scatter_df2['y'] = ys
    scatter_df2['group1'] = line_groups[positions].astype('str')
    scatter_df2['order1'] = line_orders[positions].astype(int)
    
    return pd.concat((scatter_df, scatter_df2), ignore_index=True)





def zoom_to_label_target(target_morphotope, component_data, scatter_df, scatter, selection=False):
    target_morphotope_idx = np.where(component_data.index == target_morphotope)[0][0]
    target_scatter_node = np.where((scatter_df['child'] == target_morphotope_idx) | (scatter_df['parent'] == target_morphotope_idx))
    scatter.zoom([target_scatter_node])
    if selection:
        scatter.selection(target_scatter_node)


def get_original_observations(Z, node_id, n):
    """
    Recursively retrieves all original observations that belong to a cluster node.

    Parameters:
    Z : numpy.ndarray
        Linkage matrix of shape (n-1, 4) where each row [Z[i, 0], Z[i, 1], Z[i, 2], Z[i, 3]]
        contains two merged clusters and additional metadata.
    node_id : int
        The node (cluster) ID for which to retrieve original observations.
    n : int
        The total number of original observations.

    Returns:
    observations : list
        List of original observation indices that are part of the specified node_id.
    """
    # If the node_id refers to an original observation, return it
    if node_id < n:
        return [node_id]
    
    # Otherwise, recursively find observations for the two merged clusters
    cluster_idx = node_id - n # Adjust the index because new clusters start from n
    
    left_cluster = int(Z[cluster_idx, 0])
    right_cluster = int(Z[cluster_idx, 1])
    
    left_observations = get_original_observations(Z, left_cluster, n)
    right_observations = get_original_observations(Z, right_cluster, n)
    
    return left_observations + right_observations

def get_children(linkage_matrix, left, right):
    '''Get get all original items below a node in the linkage matrix.
    Left, right correspond to parent child in a linkage matrix node.'''
    res1 = get_original_observations(linkage_matrix, left, linkage_matrix.shape[0] + 1)
    res2 = get_original_observations(linkage_matrix, right, linkage_matrix.shape[0] + 1)
    res = np.union1d(res1, res2)
    return res



# def dendogram_idx_nodes(idx, node_id, linkage, n_children, out=[]):
    
#     if node_id < 0:
#        return
#     left, right = linkage[idx]
#     out.append((idx, node_id))
#     # recurse over the right node
#     if right >= n_children: # make sure it's not a leaf node
#         node_id -= 1
#         node_id = dendogram_idx_nodes(right - (n_children + 1), node_id,
#                                      linkage, n_children, out)
#     if left >= n_children: # make sure it's not a leaf node
#         node_id -= 1
#         node_id = dendogram_idx_nodes(left - (n_children +1), node_id,
#                                      linkage, n_children, out)
#     return node_id
