import momepy as mm
import numpy as np
import pandas as pd
from libpysal.graph import Graph
import geopandas as gpd
import pytest
import glob
import shapely
import gc
from shapely import coverage_simplify
import datetime
from libpysal.graph import read_parquet
from scipy import sparse



regions_datadir = '/data/uscuni-ulce/'
data_dir = '/data/uscuni-ulce/processed_data/'
eubucco_files = glob.glob(regions_datadir + 'eubucco_raw/*')

def process_regions():
    region_hulls = gpd.read_parquet(regions_datadir + 'regions/' + 'regions_hull.parquet')
    for region_id, _ in region_hulls.iterrows():
    
        if region_id != 69300: continue    
        
        print(datetime.datetime.now(), '----Processing ------', region_id,)
    
        process_tessellation_graph(region_id)
        process_buildings_graph(region_id)
        process_edges_graph(region_id)
        process_enclosure_graph(region_id)
        process_nodes_graph(region_id)
        gc.collect()
    

def process_tessellation_graph(region_id):
    ## tessellation graphs
    tessellation = gpd.read_parquet(data_dir + f'tessellations/tessellation_{region_id}.parquet')
    
    graph = Graph.build_fuzzy_contiguity(tessellation, buffer=1e-6).assign_self_weight()
    graph.to_parquet(data_dir + 'neigh_graphs/' + f'tessellation_graph_{region_id}_knn1.parquet')
    print('Built tess graph knn=1')

    del graph
    del tessellation
    gc.collect()    


def process_buildings_graph(region_id):
    
    buildings = gpd.read_parquet(data_dir + f'/buildings/buildings_{region_id}.parquet')

    graph = Graph.build_fuzzy_contiguity(buildings, buffer=1e-6).assign_self_weight()
    
    graph.to_parquet(data_dir + 'neigh_graphs/' + f'building_graph_{region_id}_knn1.parquet')
    print('Built buildings graph knn=1')
    
    del graph
    del buildings
    gc.collect()

def process_edges_graph(region_id):

    streets = gpd.read_parquet(data_dir + f'/streets/streets_{region_id}.parquet')
    
    graph = Graph.build_contiguity(streets, rook=False).assign_self_weight()
    graph.to_parquet(data_dir + 'neigh_graphs/' + f'street_graph_{region_id}_knn1.parquet')
    print('Built streets graph knn=1')
    
    del graph
    del streets
    gc.collect()


def process_enclosure_graph(region_id):
    ## tessellation graphs
    inputdf = gpd.read_parquet(data_dir + f'enclosures/enclosure_{region_id}.parquet')

    graph = Graph.build_contiguity(inputdf, rook=False).assign_self_weight()
    graph.to_parquet(data_dir + 'neigh_graphs/' + f'enclosure_graph_{region_id}_knn1.parquet')
    print('Built enclosure graph knn=1')


def process_nodes_graph(region_id):
    ## tessellation graphs
    streets = gpd.read_parquet(data_dir + f'streets/streets_{region_id}.parquet')

    nx_graph = mm.gdf_to_nx(streets)
    nx_graph = mm.node_degree(nx_graph)
    _, _, w = mm.nx_to_gdf(nx_graph, spatial_weights=True)

    graph = Graph.from_W(w)
    graph.to_parquet(data_dir + 'neigh_graphs/' + f'nodes_graph_{region_id}_knn1.parquet')
    print('Built nodes graph knn=1')
    
    del graph
    del streets
    del nx_graph
    del w
    gc.collect()


if __name__ == '__main__':
    process_regions()
    # process_regions_further()
