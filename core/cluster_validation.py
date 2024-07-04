import numpy as np
import pandas as pd

## each of this is an enclosure ID
focus_areas = {
    3103: "karlin",
    13295: "vinohrady",
    909: "mala strani",
    4429: "holyne",
    4406: "housing estate",
    2265: "stare mesto",
    1544: "nusle",
    18215: "malesice",
}


def generate_enc_groups(
    tessellation, enclosures, include_random_sample=False, random_sample_size=1000
):
    """Create a buffer around a specific enclosure, then return all tessellation cell ids inside the buffer.
    Also add random points to test dimensionality reduction/clustering algorithm performance."""
    buffers = enclosures.loc[list(focus_areas.keys())].buffer(500)
    group_dict = pd.Series(focus_areas).reset_index(drop=True).to_dict()
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
