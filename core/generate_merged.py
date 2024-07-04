import datetime
import gc
import glob

import geopandas as gpd
import pandas as pd
from utils import used_keys

regions_datadir = "/data/uscuni-ulce/"
data_dir = "/data/uscuni-ulce/processed_data/"
eubucco_files = glob.glob(regions_datadir + "eubucco_raw/*")
graph_dir = data_dir + "neigh_graphs/"
chars_dir = "/data/uscuni-ulce/processed_data/chars/"


def process_regions():
    region_hulls = gpd.read_parquet(
        regions_datadir + "regions/" + "regions_hull.parquet"
    )
    for region_id, _ in region_hulls.iterrows():
        if region_id != 69300:
            continue

        print(
            datetime.datetime.now(),
            "----Processing ------",
            region_id,
        )

        generate_merged(region_id)

        gc.collect()


def generate_merged(region_id):
    tessellation = gpd.read_parquet(
        chars_dir + f"tessellations/chars_{region_id}.parquet"
    )
    buildings = gpd.read_parquet(chars_dir + f"buildings/chars_{region_id}.parquet")
    enclosures = gpd.read_parquet(chars_dir + f"enclosures/chars_{region_id}.parquet")
    streets = gpd.read_parquet(chars_dir + f"streets/chars_{region_id}.parquet")
    nodes = gpd.read_parquet(chars_dir + f"nodes/chars_{region_id}.parquet")

    merged = pd.merge(
        tessellation.drop(columns=["geometry"]),
        buildings.drop(columns=["nodeID", "geometry"]),
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
            "id",
            "osm_id",
            "index",  ## maybe keep
        ]
    )
    merged = merged.set_index(tessellation.index)
    primary = merged[list(used_keys.keys())]
    return primary
