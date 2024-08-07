import momepy as mm
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.graph import Graph


def lazy_higher_order(graph, k, n_splits, iteration_order=None):
    """Generate a higher order pysal.Graph in chunks"""
    A = graph.transform("B").sparse
    ids = graph.unique_ids.values
    id_to_numeric = pd.Series(np.arange(len(ids)), index=ids)

    if iteration_order is None:
        iteration_order = ids

    for source in np.array_split(iteration_order, n_splits):
        nodes = id_to_numeric.loc[source]

        ## get higher order topological neighbours
        Q = A[nodes, :].copy()
        for _ in range(1, k):
            next_step = Q @ A
            Q += next_step

        sparray = Q.tocoo(copy=False)
        sorter = sparray.row.argsort()
        head = source[sparray.row][sorter]
        tail = ids[sparray.col][sorter]

        unique_head = np.unique(head)
        unique_tail = np.unique(tail)

        buffers = np.setdiff1d(unique_tail, unique_head, assume_unique=True)
        buffers.sort()

        ## need to add buffers from tail to focals, since graph constructor drops them
        graph_head = np.append(head, buffers)
        graph_tail = np.append(tail, buffers)
        graph_weights = np.ones(len(graph_head))
        graph_weights[len(head) :] = 0

        partial_higher = Graph.from_arrays(graph_head, graph_tail, graph_weights)

        yield partial_higher


def partial_apply(graph, higher_order_k, n_splits, func, **kwargs):
    """apply a function across all partial higher order graphs and aggregate the result"""
    res = pd.Series(np.nan, index=graph.unique_ids)
    for partial_higher in lazy_higher_order(graph, k=higher_order_k, n_splits=n_splits):
        partial_focals = np.setdiff1d(
            partial_higher.unique_ids, partial_higher.isolates
        )

        partial_result = func(partial_focals, partial_higher, **kwargs)

        res.loc[partial_focals] = partial_result.loc[partial_focals].values
        del partial_higher
        del partial_result

    return res


def partial_describe_reached_agg(
    y, graph_index, graph, higher_order, n_splits, q=None, statistics=None
):
    """spread momepy.describe_reached_agg across all partial higher order graphs and aggregate the result"""
    reverse_graph_index = pd.Series(
        graph_index[graph_index.notna()].index,
        index=graph_index[graph_index.notna()].values.astype(int),
    )

    res = pd.DataFrame(np.nan, columns=statistics, index=graph.unique_ids)

    for partial_higher in lazy_higher_order(graph, k=higher_order, n_splits=n_splits):
        partial_focals = np.setdiff1d(
            partial_higher.unique_ids, partial_higher.isolates
        )

        relevant_groups = np.intersect1d(
            partial_higher.unique_ids,
            reverse_graph_index.index.values,
            assume_unique=True,
        )
        relevant_ys_index = reverse_graph_index.loc[relevant_groups].unique()

        partial_result = mm.describe_reached_agg(
            y.loc[relevant_ys_index],
            graph_index.loc[relevant_ys_index],
            graph=partial_higher,
            statistics=statistics,
            q=q,
        )
        res.loc[partial_focals] = partial_result.loc[partial_focals]
        del partial_higher
        del partial_result
    return res

def partial_mean_intb_dist(partial_focals, partial_higher, buildings, bgraph):
    """spread momepy.mean_interbuilding_distance across partial higher order graphs.
    This function has to be passed to partial_apply be used with partial_apply."""
    pos_unique_higher = partial_higher.unique_ids
    pos_unique_higher = pos_unique_higher[pos_unique_higher >= 0]
    partial_buildings = buildings.loc[pos_unique_higher]
    partial_bgraph = bgraph.subgraph(partial_buildings.index.values)
    partial_bgraph3 = partial_higher.subgraph(partial_buildings.index.values)

    res = pd.Series(np.nan, index=partial_higher.unique_ids)
    mibd = mm.mean_interbuilding_distance(
        buildings.loc[pos_unique_higher], partial_bgraph, partial_bgraph3
    )
    res.loc[mibd.index] = mibd.values
    return res
        

char_names = {
    "sdbAre": "area of building",
    "sdbPer": "perimeter of building",
    "sdbCoA": "courtyard area of building",
    "ssbCCo": "circular compactness of building",
    "ssbCor": "corners of building",
    "ssbSqu": "squareness of building",
    "ssbERI": "equivalent rectangular index of building",
    "ssbElo": "elongation of building",
    "ssbCCM": "centroid - corner mean distance of building",
    "ssbCCD": "centroid - corner distance deviation of building",
    "stbOri": "orientation of building",
    "sdcLAL": "longest axis length of ETC",
    "sdcAre": "area of ETC",
    "sscCCo": "circular compactness of ETC",
    "sscERI": "equivalent rectangular index of ETC",
    "stcOri": "orientation of ETC",
    "sicCAR": "covered area ratio of ETC",
    "stbCeA": "cell alignment of building",
    "mtbAli": "alignment of neighbouring buildings",
    "mtbNDi": "mean distance between neighbouring buildings",
    "mtcWNe": "perimeter-weighted neighbours of ETC",
    "mdcAre": "area covered by neighbouring cells",
    "ltcWRE": "weighted reached enclosures of ETC",
    "ltbIBD": "mean inter-building distance",
    "sdsSPW": "width of street profile",
    "sdsSWD": "width deviation of street profile",
    "sdsSPO": "openness of street profile",
    "sdsLen": "length of street segment",
    "sssLin": "linearity of street segment",
    "ldsMSL": "mean segment length within 3 steps",
    "degree": "node degree of junction",
    "meshedness": "local meshedness of street network",
    "proportion_3": "local proportion of 3-way intersections of street network",
    "proportion_4": "local proportion of 4-way intersections of street network",
    "proportion_0": "local proportion of cul-de-sacs of street network",
    "local_closeness": "local closeness of street network",
    "ldsCDL": "local cul-de-sac length of street network",
    "xcnSCl": "square clustering of street network",
    "mtdMDi": "mean distance to neighbouring nodes of street network",
    "lddNDe": "local node density of street network",
    "linWID": "local degree weighted node density of street network",
    "stbSAl": "street alignment of building",
    "stcSAl": "street alignment of ETC",
    "mtbSWR": "shared walls ratio of buildings",
    "sddAre": "area covered by node-attached ETCs",
    "sdsAre": "area covered by edge-attached ETCs",
    "sisBpM": "buildings per meter of street segment",
    "misCel": "reached ETCs by neighbouring segments",
    "mdsAre": "reached area by neighbouring segments",
    "lisCel": "reached ETCs by local street network",
    "ldsAre": "reached area by local street network",
    "ltcRea": "reached ETCs by tessellation contiguity",
    "ltcAre": "reached area by tessellation contiguity",
    "ldeAre": "area of enclosure",
    "ldePer": "perimeter of enclosure",
    "lseCCo": "circular compactness of enclosure",
    "lseERI": "equivalent rectangular index of enclosure",
    "lseCWA": "compactness-weighted axis of enclosure",
    "lteOri": "orientation of enclosure",
    "lteWNB": "perimeter-weighted neighbours of enclosure",
    "lieWCe": "area-weighted ETCs of enclosure",
    "ldbPWL": "perimeter wall length of adjacent buildings",
    "libNCo": "number of courtyards within adjacent buildings",
    "misRea": "cells reached within neighbouring street segments",
    "ltcBuA": "level of building adjacency",
    "ldsRea": "number of reached tessellation cells in street network",
    "ldsAre": "total tessellation cell area reached in street network",
    "mtdDeg": "number of streets that cross node",
    "lcdMes": "street network messhedness in node neighbourhood",
    "linP3W": "proportion of threeway intersections in node neighbourhood",
    "linP4W": "proportion of fourway intersections in node neighbourhood",
    "linPDE": "proportion of deadend intersections in node neighbourhood",
    "lcnClo": "local closeness centrality in node neighbourhood",
    "midRea": "number of tess cells in node neigbhorhood",
    "midAre": "total area of tess cells in node neigbhorhood",
    "ldkAre": "enclosure area",
    "ldkPer": "enclosure perimeter",
    "lskCCo": "number of couryards in enclosure",
    "lskERI": "rectangular index of enclosure",
    "lskCWA": "compactness weighted axis of enclosure",
    "ltkOri": "enclosure orientation",
    "ltkWNB": "perimeter-weighted neighbours of enclosure",
    "likWBB": "total of building areas within the enclosure, normalised by enclosure area",
    "ltcWRB": "number of unique enclosures in ETC neigbhourhood",
}

used_keys = {
    "sdbAre": "area of building",
    "sdbPer": "perimeter of building",
    "sdbCoA": "courtyard area of building",
    "ssbCCo": "circular compactness of building",
    "ssbCor": "corners of building",
    "ssbSqu": "squareness of building",
    "ssbERI": "equivalent rectangular index of building",
    "ssbElo": "elongation of building",
    "ssbCCM": "centroid - corner mean distance of building",
    "ssbCCD": "centroid - corner distance deviation of building",
    "stbOri": "orientation of building",
    "mtbSWR": "shared walls ratio of buildings",
    "libNCo": "number of courtyards within adjacent buildings",
    "ldbPWL": "perimeter wall length of adjacent buildings",
    "ltcBuA": "level of building adjacency",
    "mtbAli": "alignment of neighbouring buildings",
    "mtbNDi": "mean distance between neighbouring buildings",
    "ltbIBD": "mean inter-building distance",
    "stbCeA": "cell alignment of building",
    "stbSAl": "street alignment of building",
    "sdsLen": "length of street segment",
    "sssLin": "linearity of street segment",
    "ldsMSL": "mean segment length within 3 steps",
    "ldsRea": "reached ETCs by local street network",
    "ldsAre": "reached total ETC area by local street network",
    "sisBpM": "buildings per meter of street segment",
    "sdsSPW": "width of street profile",
    "sdsSPO": "openness of street profile",
    "sdsSWD": "width deviation of street profile",
    "mtdDeg": "node degree of junction",
    "lcdMes": "local meshedness of street network",
    "linP3W": "local proportion of 3-way intersections of street network",
    "linP4W": "local proportion of 4-way intersections of street network",
    "linPDE": "local proportion of cul-de-sacs of street network",
    "lcnClo": "local closeness of street network",
    "lddNDe": "local node density of street network",
    "linWID": "local degree weighted node density of street network",
    "ldsCDL": "local cul-de-sac length of street network",
    "xcnSCl": "square clustering of street network",
    "mtdMDi": "mean distance to neighbouring nodes of street network",
    "sddAre": "area covered by node-attached ETCs",
    "midRea": "number of tess cells in node neigbhorhood",
    "midAre": "total area of tess cells in node neigbhorhood",
    "stcOri": "orientation of ETC",
    "sdcLAL": "longest axis length of ETC",
    "sdcAre": "area of ETC",
    "sscCCo": "circular compactness of ETC",
    "sscERI": "equivalent rectangular index of ETC",
    "mtcWNe": "perimeter-weighted neighbours of ETC",
    "mdcAre": "area covered by neighbouring cells",
    "ltcWRB": "number of unique enclosures in ETC neigbhourhood",
    "sicCAR": "covered area ratio of ETC",
    "stcSAl": "street alignment of ETC",
    "ldkAre": "area of enclosure",
    "ldkPer": "perimeter of enclosure",
    "lskCCo": "circular compactness of enclosure",
    "lskERI": "equivalent rectangular index of enclosure",
    "lskCWA": "compactness-weighted axis of enclosure",
    "ltkOri": "orientation of enclosure",
    "ltkWNB": "perimeter-weighted neighbours of enclosure",
    "likWBB": "total of building areas within the enclosure, normalised by enclosure area",
    "sdsAre": "area covered by edge-attached ETCs",
    "likWCe": "area-weighted ETCs of enclosure"
}

char_units = {
    "sdbAre": "area",
    "sdbPer": "metres",
    "sdbCoA": "area",
    "ssbCCo": "ratio",
    "ssbCor": "count",
    "ssbSqu": "degrees",
    "ssbERI": "ratio",
    "ssbElo": "ratio",
    "ssbCCM": "metres",
    "ssbCCD": "ratio",
    "stbOri": "degrees",
    "mtbSWR": "ratio",
    "libNCo": "count-neighbourhood",
    "ldbPWL": "metres-neighbourhood",
    "ltcBuA": "ratio-neigbhourhood",
    "mtbAli": "degrees",
    "mtbNDi": "metres-neighbourhood",
    "ltbIBD": "metres-neighbourhood",
    "stbCeA": "degrees",
    "stbSAl": "degrees",
    "sdsLen": "metres",
    "sssLin": "ratio",
    "ldsMSL": "metres-neighbourhood",
    "ldsRea": "count-neighbourhood",
    "ldsAre": "area-neigbourhood",
    "sisBpM": "ratio",
    "sdsSPW": "metres",
    "sdsSPO": "ratio",
    "sdsSWD": "ratio",
    "mtdDeg": "count",
    "lcdMes": "ratio",
    "linP3W": "ratio",
    "linP4W": "ratio",
    "linPDE": "ratio",
    "lcnClo": "ratio",
    "lddNDe": "ratio",
    "linWID": "ratio",
    "ldsCDL": "metres",
    "xcnSCl": "ratio",
    "mtdMDi": "metres",
    "sddAre": "area",
    "midRea": "count",
    "midAre": "area",
    "stcOri": "degrees",
    "sdcLAL": "metres",
    "sdcAre": "area",
    "sscCCo": "ratio",
    "sscERI": "ratio",
    "mtcWNe": "ratio",
    "mdcAre": "area",
    "ltcWRB": "count",
    "sicCAR": "ratio",
    "stcSAl": "degrees",
    "ldkAre": "area",
    "ldkPer": "metres",
    "lskCCo": "count",
    "lskERI": "ratio",
    "lskCWA": "metres",
    "ltkOri": "degrees",
    "ltkWNB": "ratio",
    "likWBB": "ratio",
    "sdsAre": "area-neigbourhood"
}
