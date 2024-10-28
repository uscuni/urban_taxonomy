import datetime
import gc
import glob

import geopandas as gpd
import momepy as mm
from libpysal.graph import Graph

regions_datadir = "/data/uscuni-ulce/"
data_dir = "/data/uscuni-ulce/processed_data/"
eubucco_files = glob.glob(regions_datadir + "eubucco_raw/*")
regions_buildings_dir = '/data/uscuni-ulce/regions/buildings/'
buildings_dir = '/data/uscuni-ulce/processed_data/buildings/'
overture_streets_dir = '/data/uscuni-ulce/overture_streets/'
streets_dir = '/data/uscuni-ulce/processed_data/streets/'
enclosures_dir = '/data/uscuni-ulce/processed_data/enclosures/'
tessellations_dir = '/data/uscuni-ulce/processed_data/tessellations/'
graph_dir = '/data/uscuni-ulce/processed_data/neigh_graphs/'
chars_dir = '/data/uscuni-ulce/processed_data/chars/'

def process_all_regions_graphs():
    
    region_hulls = gpd.read_parquet(
        regions_datadir + "regions/" + "cadastre_regions_hull.parquet"
    )
    
    for region_id, _ in region_hulls.iterrows():

        print(
            datetime.datetime.now(),
            "----Processing ------",
            region_id,
        )

        process_region_graphs(
        region_id,
        graph_dir,
        buildings_dir,
        streets_dir,
        enclosures_dir,
        tessellations_dir,
        )
        
        gc.collect()


def process_region_graphs(
    region_id,
    graph_dir,
    buildings_dir,
    streets_dir,
    enclosures_dir,
    tessellations_dir,
):
    process_tessellation_graph(region_id, graph_dir, tessellations_dir)
    process_buildings_graph(region_id, graph_dir, buildings_dir)
    process_edges_graph(region_id, graph_dir, streets_dir)
    process_enclosure_graph(region_id, graph_dir, enclosures_dir)
    process_nodes_graph(region_id, graph_dir, streets_dir)


def process_tessellation_graph(region_id, graph_dir, tessellations_dir):
    ## tessellation graphs
    tessellation = gpd.read_parquet(
        tessellations_dir + f"tessellation_{region_id}.parquet"
    )

    graph = Graph.build_fuzzy_contiguity(tessellation, buffer=1e-6).assign_self_weight()
    graph.to_parquet(
        graph_dir + f"tessellation_graph_{region_id}.parquet"
    )
    print("Built tess graph knn=1")

    del graph
    del tessellation
    gc.collect()


def process_buildings_graph(region_id, graph_dir, buildings_dir):
    buildings = gpd.read_parquet(buildings_dir + f"buildings_{region_id}.parquet")

    ## the buffer has to be higher than the simplification value
    graph = Graph.build_fuzzy_contiguity(buildings, buffer=.25).assign_self_weight()

    graph.to_parquet(
        graph_dir + f"building_graph_{region_id}.parquet"
    )
    print("Built buildings graph knn=1")

    del graph
    del buildings
    gc.collect()


def process_edges_graph(region_id, graph_dir, streets_dir):
    ### streets graph
    streets = gpd.read_parquet(streets_dir + f"streets_{region_id}.parquet")

    graph = Graph.build_contiguity(streets, rook=False).assign_self_weight()
    graph.to_parquet(
        graph_dir + f"street_graph_{region_id}.parquet"
    )
    print("Built streets graph knn=1")

    del graph
    del streets
    gc.collect()


def process_enclosure_graph(region_id, graph_dir, enclosures_dir):
    ## enclosure graphs
    inputdf = gpd.read_parquet(enclosures_dir + f"enclosure_{region_id}.parquet")

    graph = Graph.build_contiguity(inputdf, rook=False).assign_self_weight()
    graph.to_parquet(
        graph_dir + f"enclosure_graph_{region_id}.parquet"
    )
    print("Built enclosure graph knn=1")


def process_nodes_graph(region_id, graph_dir, streets_dir):
    ## nodes graphs
    streets = gpd.read_parquet(streets_dir + f"streets_{region_id}.parquet")

    nx_graph = mm.gdf_to_nx(streets, preserve_index=True)
    nx_graph = mm.node_degree(nx_graph)
    _, _, w = mm.nx_to_gdf(nx_graph, spatial_weights=True)

    graph = Graph.from_W(w)
    graph.to_parquet(
        graph_dir + f"nodes_graph_{region_id}.parquet"
    )
    print("Built nodes graph knn=1")

    del graph
    del streets
    del nx_graph
    del w
    gc.collect()


if __name__ == "__main__":
    process_all_regions_graphs()
