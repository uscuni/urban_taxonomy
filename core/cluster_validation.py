import numpy as np
import pandas as pd
import geopandas as gpd

def generate_validation_groups(
    tessellation, buffer=1000, include_random_sample=False, random_sample_size=1000): 
    """Create a buffer around a specific point, then return all tessellation cell ids inside the buffer.
    Also add random points to test dimensionality reduction/clustering algorithm performance."""
    xs = [4639418.73732028, 4638766.14693257, 4636102.61687298,
           4632830.04468479, 4634059.47490346, 4637037.54752477,
           4638734.18270978, 4644599.25531156]
    ys = [3007593.31449975, 3005492.26689675, 3006724.3212875 ,
           2999944.25664089, 3000331.73031417, 3006410.00813384,
           3003706.91345186, 3006464.38673084]
    names = ['karlin',
     'vinohrady',
     'mala strana',
     'holyne',
     'housing estate',
     'stare mesto',
     'nusle',
     'malesice']
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
    return (
        pd.Series(np.arange(0, len(tessellation)), index=tessellation.index)
        .loc[tess_groups.index]
        .values
    )


def pprint_cluster_percentiles(X_train, labels):
    """'print cluster statistics and highlight the distribution of each feature."""
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
    }  # column col A to 2 decimals
    return cluster_stats.style.format(f).background_gradient(axis=1, cmap="BuGn")
