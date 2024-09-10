import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from lonboard.colormap import apply_categorical_cmap
from palettable.colorbrewer.qualitative import Set3_12



def generate_detailed_clusters(tessellation, include_random_sample=False, random_sample_size=1000,
                               path='../data/prague_validation/morphotopes.pq'):
    """Use predefined clusters to label tessellation cells"""
    clusters = gpd.read_parquet(path)
    inp, res = tessellation.sindex.query(clusters.geometry, predicate='intersects')
    index = tessellation.iloc[res].index
    to_keep = ((~index.duplicated()) & (index >= 0))
    index = index[to_keep]
    values = clusters.iloc[inp, 0].values[to_keep]
    tess_groups = pd.Series(values, index)
    
    if include_random_sample:
        random_sample_index = tessellation.sample(
            random_sample_size, random_state=1
        ).index
        random_sample = pd.Series("random", index=random_sample_index)
        tess_groups = pd.concat((tess_groups, random_sample))
    
    return tess_groups[~tess_groups.index.duplicated()]
    

def generate_neigbhourhood_groups(
    tessellation, buffer=400, include_random_sample=False, random_sample_size=1000): 
    """Create a buffer around a specific point, then return all tessellation cell ids inside the buffer.
    Also add random points to test dimensionality reduction/clustering algorithm performance."""
    xs = [4639418.73732028, 4638766.14693257, 4636102.61687298,
           4632830.04468479, 4634059.47490346, 4637037.54752477,
           4638734.18270978, 4644599.25531156, 4636980.614170434]
    ys = [3007593.31449975, 3005492.26689675, 3006724.3212875 ,
           2999944.25664089, 3000331.73031417, 3006410.00813384,
           3003706.91345186, 3006464.38673084, 3007018.3956980314]
    names = ['karlin',
     'vinohrady',
     'mala strana',
     'holyne',
     'housing estate',
     'stare mesto',
     'nusle',
     'malesice',
            'josefov']
    buffers = gpd.GeoSeries.from_xy(xs, ys, crs=3035).buffer(buffer)
    group_dict = pd.Series(names)
    
    areas, tids = tessellation.sindex.query(buffers, predicate="intersects")
    tess_groups = pd.Series(areas, index=tessellation.index[tids]).replace(group_dict)
    
    if include_random_sample:
        random_sample_index = tessellation.sample(
            random_sample_size, random_state=1
        ).index
        random_sample = pd.Series("random", index=random_sample_index)
        tess_groups = pd.concat((tess_groups, random_sample))
    
    return tess_groups[~tess_groups.index.duplicated()]


def get_tess_groups_original_ilocs(tessellation, tess_groups):
    '''Find the location of the tess_groups members in the tessellation index.'''
    return (
        pd.Series(np.arange(0, len(tessellation)), index=tessellation.index)
        .loc[tess_groups.index]
        .values
    )


def pprint_cluster_percentiles(X_train, labels):
    """'Print cluster statistics and highlight the distribution of each feature."""
    cluster_stats = X_train.groupby(labels).describe()
    cluster_stats = cluster_stats.loc[:, (slice(None), ["25%", "50%", "75%"])].T
    counts = X_train.groupby(labels).size()
    cluster_stats.loc[("count", "count"), :] = counts.values
    extended_col_names = {}
    for c in X_train.columns.values:
        if "_" in c:
            orig_key = c.split("_")[0]
            attach = c.split("_")[1]
            extended_col_names[c] = used_keys[orig_key] + " (" + attach + ")"
        else:
            extended_col_names[c] = used_keys[c]
    cluster_stats = cluster_stats.rename(extended_col_names, axis=0)
    f = {
        k: "{:.4f}" for k in cluster_stats.columns.values
    }
    return cluster_stats.style.format(f).background_gradient(axis=1, cmap="BuGn")


def colored_crosstab(vals1, vals2):
    """"Print a cross tabulation between the two sets of values, using continous highlighting"""
    ct = pd.crosstab(vals1, vals2)
    return ct.style.background_gradient(axis=1, cmap="BuGn")

def print_distance(groups, metric='euclidean'):
    """print the distance between the points in the groups dataframe."""
    from scipy.spatial.distance import pdist, squareform
    vals = squareform(pdist(groups, metric=metric))
    df = pd.DataFrame(vals, index=groups.index, columns=groups.index)
    return df.style.background_gradient(axis=1, cmap="BuGn")

def get_feature_importance(input_data, clusters):
    '''Train a random forest classifier per cluster and output the feature importance.
    Used to identify dominating features.'''
    imps = pd.DataFrame()
    
    for cluster in np.unique(clusters):
        
        cluster_bool = clusters == cluster
        
        clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42, verbose=1)
        clf = clf.fit(input_data.values, cluster_bool)
    
        importances = pd.Series(clf.feature_importances_, index=input_data.columns).sort_values(ascending=False)
    
        imps[f'cluster_{cluster}'] = importances.head(50).index.values
        imps[f'cluster_{cluster}_vals'] = importances.head(50).values
    return imps


def get_color(labels_to_color, noise_color=[0, 0, 0]):
    '''Generate n colors for n labels. Labels with -1 are black.'''
    import glasbey
    
    def hex_to_rgb(hexa):
        return tuple(int(hexa[i : i + 2], 16) for i in (0, 2, 4))
    
    if labels_to_color.max() >= 11:
        gb_cols = glasbey.extend_palette(
            Set3_12.hex_colors, palette_size=np.unique(labels_to_color).shape[0] + 1
        )
    else:
        gb_cols = Set3_12.hex_colors
    
    gb_cols = [hex_to_rgb(c[1:]) for c in gb_cols]
    
    colors = apply_categorical_cmap(
        labels_to_color, cmap=dict(zip(np.unique(labels_to_color), gb_cols, strict=False))
    )
    colors[labels_to_color == -1] = noise_color
    return colors

def get_linkage_matrix(model):
    """" Create a linkage matrix from a sklearn hierarchical clustering model.
    Requires the full tree and the distances stored in the model instance."""

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix