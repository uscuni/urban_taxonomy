import datetime
import glob

import geopandas as gpd
import geoplanar
import numpy as np
import pandas as pd

regions_datadir = "/data/uscuni-ulce/"
data_dir = "/data/uscuni-ulce/processed_data/"
eubucco_files = glob.glob(regions_datadir + "eubucco_raw/*")


def process_all_regions_buildings():
    building_region_mapping = pd.read_parquet(
        regions_datadir + "regions/" + "id_to_region.parquet", engine="pyarrow"
    )
    typed_dict = pd.Series(
        np.arange(building_region_mapping["id"].values.shape[0]),
        index=building_region_mapping["id"].values,
    )
    region_ids = building_region_mapping.groupby("region")["id"].unique()
    del building_region_mapping  # its 2/3 gb
    region_hulls = gpd.read_parquet(
        regions_datadir + "regions/" + "regions_hull.parquet"
    )

    for region_id, region_hull in region_hulls.iterrows():
        region_hull = region_hull["convex_hull"]

        if region_id != 69300: continue

        print("----", "Processing region: ", region_id, datetime.datetime.now())
        buildings = read_region_buildings(
            typed_dict, region_ids, region_hull, region_id
        )

        buildings = process_region_buildings(buildings, True)

        buildings.to_parquet(data_dir + f"buildings/buildings_{region_id}.parquet")

        del buildings


def process_region_buildings(buildings, simplify, simplification_tolerance=.1, merge_limit=25):
    
    initial_shape = buildings.shape

    ## fix invalid geometry
    buildings["geometry"] = buildings.make_valid()

    ## explode multipolygons
    buildings = buildings.explode(ignore_index=True)

    ## keep only polygons
    buildings = buildings[buildings["geometry"].geom_type == "Polygon"].reset_index(
        drop=True
    )

    ## simplify geometry - most eubucco data has topological issues
    if simplify:
        buildings["geometry"] = buildings.simplify(simplification_tolerance)

    ## merge buildings that overlap either 1) at least .10 percent or are smaller than 30m^2
    buildings = geoplanar.merge_overlaps(
        buildings, merge_limit=merge_limit, overlap_limit=0.1
    )

    ## drop remaining overlaps
    buildings = geoplanar.trim_overlaps(buildings, largest=False)

    ## fix any multipolygons
    buildings = buildings.explode(ignore_index=True)

    print(
        "Percent polygons: ",
        (buildings.geom_type == "Polygon").sum() / buildings.shape[0],
    )

    # drop non-polygons
    buildings = buildings[buildings.geom_type == "Polygon"].reset_index(drop=True)

    # merge touching collapsing buildings
    shrink = buildings.buffer(-0.5, resolution=2)
    buildings = geoplanar.merge_touching(
        buildings, np.where(shrink.is_empty), largest=True
    )
    # drop non polygons
    buildings = buildings.explode(ignore_index=True)
    buildings = buildings[buildings.geom_type == "Polygon"].reset_index(drop=True)

    ## need one more pass to ensure only valid geometries
    if simplify:
        buildings["geometry"] = buildings.simplify(simplification_tolerance)
        buildings["geometry"] = buildings.make_valid()
        buildings = buildings[buildings.geom_type == "Polygon"].reset_index(drop=True)

    print(
        "Final polygons: ",
        buildings.shape[0],
        ", dropped: ",
        1 - (buildings.shape[0] / initial_shape[0]),
    )

    buildings["geometry"] = buildings.normalize()
    return buildings


def read_region_buildings(typed_dict, region_ids, region_hull, region_id):
    """Filter all buildings and only get those with the specific region id."""
    typed_region_buildings = typed_dict.loc[region_ids[region_id]].values
    read_mask = region_hull.buffer(100)

    res = None
    for filepath in eubucco_files:
        gdf = gpd.read_file(
            filepath,
            engine="pyogrio",
            columns=["id", "geometry"],
            bbox=read_mask.bounds,
        )
        typed_gdf_buildings = typed_dict.loc[gdf["id"].values].values
        to_keep = np.isin(
            typed_gdf_buildings, typed_region_buildings, assume_unique=True
        )

        res = pd.concat((res, gdf[to_keep]))
    buildings = res

    return buildings


if __name__ == "__main__":
    process_regions()
