import datetime
import gc
import glob

import geopandas as gpd
import momepy as mm
import numpy as np
import pandas as pd
from libpysal.graph import read_parquet
from core.utils import partial_apply, partial_describe_reached_agg, partial_mean_intb_dist

def process_regions():
    region_hulls = gpd.read_parquet(
        regions_datadir + "regions/" + "regions_hull.parquet"
    )
    # 69300 - prague
    # 12199 - small test
    # large regions = [4, 226, 3607, 8754, 16501, 55713, 62929, 107685, 115457]

    # for region_id, region_hull in region_hulls.iterrows():
    for region_id in reversed([69300]):
        print(datetime.datetime.now(), "----Processing ------", region_id)

        process_street_chars(region_id)
        gc.collect()

        process_enclosure_chars(region_id)
        gc.collect()

        process_building_chars(region_id)
        gc.collect()

        process_tessellation_chars(region_id)
        gc.collect()


def run_parallel_regions():
    building_region_mapping = pd.read_parquet(
        regions_datadir + "regions/" + "id_to_region.parquet", engine="pyarrow"
    )
    counts = building_region_mapping.groupby("region")["id"].size()
    del building_region_mapping
    gc.collect()
    parallel_regions = counts[counts < 6e5].index.values

    from joblib import Parallel, delayed

    n_jobs = -1
    new = Parallel(n_jobs=n_jobs)(
        delayed(process_single_region_chars)(region_id) for region_id in parallel_regions
    )


def process_single_region_chars(
    region_id,
    graph_dir,
    buildings_dir,
    streets_dir,
    enclosures_dir,
    tessellations_dir,
    chars_dir
):
    print(datetime.datetime.now(), "----Processing ------", region_id)

    try:
        process_street_chars(
            region_id,
            graph_dir,
            buildings_dir,
            streets_dir,
            enclosures_dir,
            tessellations_dir,
            chars_dir
        )
        gc.collect()

        process_enclosure_chars(
            region_id,
            graph_dir,
            buildings_dir,
            streets_dir,
            enclosures_dir,
            tessellations_dir,
            chars_dir
        )
        gc.collect()

        process_building_chars(
            region_id,
            graph_dir,
            buildings_dir,
            streets_dir,
            enclosures_dir,
            tessellations_dir,
            chars_dir
        )
        gc.collect()

        process_tessellation_chars(
            region_id,
            graph_dir,
            buildings_dir,
            streets_dir,
            enclosures_dir,
            tessellations_dir,
            chars_dir
        )
        gc.collect()
    
    except Exception as e:
        print("PROBLEM IN REGION", region_id)
        print(e)


def process_street_chars(
    region_id,
    graph_dir,
    buildings_dir,
    streets_dir,
    enclosures_dir,
    tessellations_dir,
    chars_dir
):
    print("Processing streets")
    streets = gpd.read_parquet(streets_dir + f"/streets_{region_id}.parquet")
    
    graph = mm.gdf_to_nx(streets, preserve_index=True)
    graph = mm.node_degree(graph)
    graph = mm.subgraph(
        graph,
        radius=5,
        meshedness=True,
        cds_length=False,
        mode="sum",
        degree="degree",
        length="mm_len",
        mean_node_degree=False,
        proportion={0: True, 3: True, 4: True},
        cyclomatic=False,
        edge_node_ratio=False,
        gamma=False,
        local_closeness=True,
        closeness_weight="mm_len",
        node_density=True,
        verbose=False,
    )
    graph = mm.cds_length(graph, radius=3, name="ldsCDL", verbose=False)
    graph = mm.clustering(graph, name="xcnSCl")
    graph = mm.mean_node_dist(graph, name="mtdMDi", verbose=False)

    nodes, edges = mm.nx_to_gdf(graph, spatial_weights=False)

    edges["sdsLen"] = edges.geometry.length
    street_orientation = mm.orientation(edges)
    edges["sssLin"] = mm.linearity(edges)

    str_q1 = read_parquet(graph_dir + f"street_graph_{region_id}_knn1.parquet")

    def mean_edge_length(partical_focals, partial_higher, y):
        return partial_higher.describe(
            y.loc[partial_higher.unique_ids], statistics=["mean"]
        )["mean"]

    edges["ldsMSL"] = partial_apply(
        str_q1,
        higher_order_k=3,
        n_splits=30,
        func=mean_edge_length,
        y=edges.geometry.length,
    )

    ## tesselation street interactions
    tessellation = gpd.read_parquet(
        tessellations_dir + f"tessellation_{region_id}.parquet"
    )
    tess_nid = mm.get_nearest_street(
        tessellation, edges
    )
    
    edges["sdsAre"] = mm.describe_agg(
        tessellation.geometry.area, tess_nid, statistics=["count", "sum"]
    )["sum"]

    res = partial_describe_reached_agg(
        tessellation.geometry.area,
        tess_nid,
        str_q1,
        higher_order=3,
        n_splits=30,
        q=None,
        statistics=["sum", "count"],
    )
    edges["ldsRea"] = res["count"]
    edges["ldsAre"] = res["sum"]

   

    ## street building interactions
    buildings = gpd.read_parquet(buildings_dir + f"buildings_{region_id}.parquet")

    blg_nid = mm.get_nearest_street(
        buildings, streets
    )
    edges["sisBpM"] = blg_nid.value_counts() / edges.length

    profile = mm.street_profile(edges, buildings, height=None, distance=3)
    edges["sdsSPW"] = profile["width"]
    edges["sdsSPO"] = profile["openness"]
    edges["sdsSWD"] = profile["width_deviation"]

    ## nodes tessellation interactions
    nodes_graph = read_parquet(graph_dir + f"nodes_graph_{region_id}_knn1.parquet")

    edges["nID"] = edges.index.values
    tessellation["nID"] = tess_nid

    
    tessellation["nodeID"] = mm.get_nearest_node(
        tessellation, nodes, edges,  tessellation["nID"]
    )
    
    nodes["sddAre"] = mm.describe_agg(
        tessellation.geometry.area, tessellation["nodeID"]
    )["sum"]
    res = mm.describe_reached_agg(
        tessellation.geometry.area, tessellation["nodeID"], graph=nodes_graph
    )
    nodes["midRea"] = res["count"]
    nodes["midAre"] = res["sum"]
    nodes.rename(
        columns={
            "degree": "mtdDeg",
            "meshedness": "lcdMes",
            "local_closeness": "lcnClo",
            "proportion_3": "linP3W",
            "proportion_4": "linP4W",
            "proportion_0": "linPDE",
            "node_density": "lddNDe",
            "node_density_weighted": "linWID",
        },
        inplace=True,
    )
    nodes.to_parquet(chars_dir + f"nodes_chars_{region_id}.parquet")
    edges.to_parquet(chars_dir + f"streets_chars_{region_id}.parquet")

    del edges
    del nodes
    del tessellation
    del buildings


def process_enclosure_chars(
    region_id,
    graph_dir,
    buildings_dir,
    streets_dir,
    enclosures_dir,
    tessellations_dir,
    chars_dir
):
    print("Processing enclosures")
    enclosures = gpd.read_parquet(
        enclosures_dir + f"enclosure_{region_id}.parquet"
    )
    enclosures["ldkAre"] = enclosures.geometry.area
    enclosures["ldkPer"] = enclosures.geometry.length
    enclosures["lskCCo"] = mm.circular_compactness(enclosures)
    enclosures["lskERI"] = mm.equivalent_rectangular_index(enclosures)
    enclosures["lskCWA"] = mm.compactness_weighted_axis(enclosures)
    enclosures["ltkOri"] = mm.orientation(enclosures)

    blo_q1 = read_parquet(graph_dir + f"enclosure_graph_{region_id}_knn1.parquet")
    enclosures["ltkWNB"] = mm.neighbors(enclosures, blo_q1, weighted=True)


    #enclosures tessellation interactions
    tessellation = gpd.read_parquet(
        tessellations_dir + f"tessellation_{region_id}.parquet"
    )
    encl_counts = tessellation.groupby('enclosure_index').count()
    enclosures['likWCe'] = encl_counts['geometry'] / enclosures.geometry.area

    ## buildings enclosures interactions
    buildings = gpd.read_parquet(buildings_dir + f"buildings_{region_id}.parquet")


    beid = buildings.merge(
        tessellation["enclosure_index"], left_index=True, right_index=True
    )["enclosure_index"]

    res = mm.describe_agg(
        buildings.geometry.area,
        beid,
        statistics=["count", "sum"],
    )

    enclosures["likWBB"] = res["sum"] / enclosures.geometry.area


    

    enclosures.to_parquet(chars_dir + f"enclosures_chars_{region_id}.parquet")

    del enclosures
    del blo_q1
    del tessellation


def process_building_chars(
    region_id,
    graph_dir,
    buildings_dir,
    streets_dir,
    enclosures_dir,
    tessellations_dir,
    chars_dir
):
    print("Processing buildings")
    buildings = gpd.read_parquet(buildings_dir + f"buildings_{region_id}.parquet")
    
    buildings["ssbCCo"] = mm.circular_compactness(buildings)
    buildings["ssbCor"] = mm.corners(buildings, eps=15)
    buildings.loc[buildings['ssbCCo'] >= .95, 'ssbCor'] = 0
    buildings['ssbSqu'] = mm.squareness(buildings, eps=15)
    buildings.loc[buildings['ssbCCo'] >= .95, 'ssbSqu'] = 90
    cencon = mm.centroid_corner_distance(buildings, eps=15)
    buildings["ssbCCM"] = cencon["mean"]
    buildings["ssbCCD"] = cencon["std"]
    
    buildings["sdbAre"] = buildings.geometry.area
    buildings["sdbPer"] = buildings.geometry.length
    buildings["sdbCoA"] = mm.courtyard_area(buildings.geometry)
    buildings["ssbCCo"] = mm.circular_compactness(buildings)
    buildings["ssbERI"] = mm.equivalent_rectangular_index(buildings.geometry)
    buildings["ssbElo"] = mm.elongation(buildings.geometry)
    buildings["stbOri"] = mm.orientation(buildings)

    ### sometimes shared walls gives GEOS exceptions, region-12199 for example
    try:
        buildings["mtbSWR"] = mm.shared_walls(buildings, strict=False, tolerance=.15) / buildings.geometry.length
    except Exception as e:
        print(e, region_id)
        buildings["mtbSWR"] = (
            mm.shared_walls(buildings.set_precision(1e-6), strict=False, tolerance=.15) / buildings.geometry.length
        )

    buildings_q1 = read_parquet(graph_dir + f"building_graph_{region_id}_knn1.parquet")
    buildings["libNCo"] = mm.courtyards(buildings, buildings_q1, buffer=.25)
    buildings["ldbPWL"] = mm.perimeter_wall(buildings, buildings_q1, buffer=.25)

    ## building tessellation interactions

    queen_1 = read_parquet(graph_dir + f"tessellation_graph_{region_id}_knn1.parquet")
    bgraph = queen_1.subgraph(buildings_q1.unique_ids)

    buildings["ltcBuA"] = mm.building_adjacency(buildings_q1, bgraph)
    buildings["mtbAli"] = mm.alignment(buildings["stbOri"], bgraph)
    buildings["mtbNDi"] = mm.neighbor_distance(buildings, bgraph)

    res = partial_apply(
        graph=queen_1,
        higher_order_k=3,
        n_splits=50,
        func=partial_mean_intb_dist,
        buildings=buildings,
        bgraph=bgraph,
    )
    buildings["ltbIBD"] = res[res.index >= 0]

    del queen_1
    del bgraph
    gc.collect()

    tessellation = gpd.read_parquet(
        tessellations_dir + f"tessellation_{region_id}.parquet"
    )
    tessellation["stcOri"] = mm.orientation(tessellation)
    buildings["stbCeA"] = mm.cell_alignment(
        buildings["stbOri"], tessellation[tessellation.index >= 0]["stcOri"]
    )

    ## building streets interactions
    streets = gpd.read_parquet(streets_dir + f"streets_{region_id}.parquet")
    graph = mm.gdf_to_nx(streets, preserve_index=True)
    nodes, edges = mm.nx_to_gdf(graph, spatial_weights=False)
    tess_nid = mm.get_nearest_street(
        tessellation, edges
    )
    blg_nid = mm.get_nearest_street(
        buildings, streets
    )
    street_orientation = mm.orientation(streets)
    buildings["nID"] = blg_nid
    edges["nID"] = edges.index.values
    buildings["stbSAl"] = mm.street_alignment(
        buildings["stbOri"][~blg_nid.isna()],
        street_orientation,
        blg_nid[~blg_nid.isna()],
    )

    buildings["nodeID"] = mm.get_nearest_node(
        buildings, nodes, edges,  buildings["nID"]
    )

    buildings.to_parquet(chars_dir + f"buildings_chars_{region_id}.parquet")
    del buildings
    del nodes
    del edges
    gc.collect()


def process_tessellation_chars(
    region_id,
    graph_dir,
    buildings_dir,
    streets_dir,
    enclosures_dir,
    tessellations_dir,
    chars_dir
):
    print("Processing tessellation")
    tessellation = gpd.read_parquet(
        tessellations_dir + f"/tessellation_{region_id}.parquet"
    )

    tessellation["stcOri"] = mm.orientation(tessellation)
    tessellation["sdcLAL"] = mm.longest_axis_length(tessellation)
    tessellation["sdcAre"] = tessellation.geometry.area
    tessellation["sscCCo"] = mm.circular_compactness(tessellation)
    tessellation["sscERI"] = mm.equivalent_rectangular_index(tessellation.geometry)

    queen_1 = read_parquet(graph_dir + f"tessellation_graph_{region_id}_knn1.parquet")
    tessellation["mtcWNe"] = mm.neighbors(tessellation, queen_1, weighted=True)
    tessellation["mdcAre"] = queen_1.describe(
        tessellation.geometry.area, statistics=["sum"]
    )["sum"]

    def partial_block_count(partial_focal, partial_higher, y):
        return partial_higher.describe(
            y.loc[partial_higher.unique_ids], statistics=["nunique"]
        )["nunique"]

    block_counts = partial_apply(
        queen_1,
        higher_order_k=3,
        n_splits=30,
        func=partial_block_count,
        y=tessellation["enclosure_index"],
    )

    def partial_tess_area(partial_focal, partial_higher, y):
        return partial_higher.describe(
            y.loc[partial_higher.unique_ids], statistics=["sum"]
        )["sum"]

    block_sums = partial_apply(
        queen_1,
        higher_order_k=3,
        n_splits=30,
        func=partial_tess_area,
        y=tessellation.geometry.area,
    )

    tessellation['ltcWRB'] = block_counts / block_sums

    # tesselation buildings interactions
    buildings = gpd.read_parquet(buildings_dir + f"buildings_{region_id}.parquet")
    tessellation["sicCAR"] = buildings.geometry.area / tessellation.geometry.area

    ## tesselation street interactions
    streets = gpd.read_parquet(streets_dir + f"streets_{region_id}.parquet")
    graph = mm.gdf_to_nx(streets, preserve_index=True)
    nodes, edges = mm.nx_to_gdf(graph, spatial_weights=False)
    street_orientation = mm.orientation(edges)
    tess_nid = mm.get_nearest_street(
        tessellation, edges
    )

    tessellation["stcSAl"] = mm.street_alignment(
        tessellation["stcOri"][~tess_nid.isna()],
        street_orientation,
        tess_nid[~tess_nid.isna()].astype(int).values,
    )
    
    edges["nID"] = edges.index.values
    tessellation["nID"] = tess_nid
    tessellation["nodeID"] = mm.get_nearest_node(
        tessellation, nodes, edges,  tessellation["nID"]
    )

    tessellation.to_parquet(chars_dir + f"tessellations_chars_{region_id}.parquet")
    del tessellation
    del buildings
    del graph
    del nodes
    del edges
    gc.collect()


if __name__ == "__main__":
    process_regions()
