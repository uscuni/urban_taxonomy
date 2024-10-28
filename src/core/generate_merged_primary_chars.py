import datetime
import gc
import glob

import geopandas as gpd
import pandas as pd
from core.utils import used_keys

regions_buildings_dir = '/data/uscuni-ulce/regions/buildings/'
buildings_dir = '/data/uscuni-ulce/processed_data/buildings/'
overture_streets_dir = '/data/uscuni-ulce/overture_streets/'
streets_dir = '/data/uscuni-ulce/processed_data/streets/'
enclosures_dir = '/data/uscuni-ulce/processed_data/enclosures/'
tessellations_dir = '/data/uscuni-ulce/processed_data/tessellations/'
graph_dir = '/data/uscuni-ulce/processed_data/neigh_graphs/'
chars_dir = '/data/uscuni-ulce/processed_data/chars/'

regions_datadir = "/data/uscuni-ulce/"


def process_regions():
    
    region_hulls = gpd.read_parquet(
            regions_datadir + "regions/" + "cadastre_regions_hull.parquet"
    )

    from joblib import Parallel, delayed
    n_jobs = -1
    new = Parallel(n_jobs=n_jobs)(
        delayed(merge_into_primary)(region_id) for region_id, _ in region_hulls.iterrows()
    )


def merge_into_primary(region_id):
    print("Processing region", region_id)
    tessellation = gpd.read_parquet(chars_dir + f"tessellations_chars_{region_id}.parquet")
    buildings = gpd.read_parquet(chars_dir + f"buildings_chars_{region_id}.parquet")
    enclosures = gpd.read_parquet(chars_dir + f"enclosures_chars_{region_id}.parquet")
    streets = gpd.read_parquet(chars_dir + f"streets_chars_{region_id}.parquet")
    nodes = gpd.read_parquet(chars_dir + f"nodes_chars_{region_id}.parquet")


    merged = pd.merge(
        tessellation.drop(columns=["geometry"]),
        buildings.drop(columns=["nodeID", "geometry", 'nID']),
        right_index=True,
        left_index=True,
        how="left",
    )
    
    merged = merged.merge(
        enclosures.drop(columns="geometry"),
        right_on="eID",
        left_on="enclosure_index",
        how="left",
    )
    
    merged = merged.merge(streets.drop(columns="geometry"), on="nID", how="left")
    merged = merged.merge(nodes.drop(columns="geometry"), on="nodeID", how="left")
    
    merged = merged.drop(
        columns=[
            "nID",
            "eID",
            "nodeID",
            "mm_len",
            "cdsbool",
            "node_start",
            "node_end",
            "x",
            "y",
            "enclosure_index",
            # "id",
            # "osm_id",
            # "index",  ## maybe keep
        ]
    )
    merged = merged.set_index(tessellation.index)

    primary = merged[list(used_keys.keys())]
    primary.to_parquet(chars_dir + f'primary_chars_{region_id}.parquet')


if __name__ == '__main__':
    process_regions()
