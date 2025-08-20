import datetime
import geopandas as gpd
import geoplanar
import numpy as np
import pandas as pd

REGION_ID = 0  # dortmund region id
regions_datadir = "D:/Work/Github_Morphotopes/data/" #for cadaster_region_hull.parquet
input_buildings_dir = "D:/Work/Github_Morphotopes/data/"
buildings_dir = "D:/Work/Github_Morphotopes/data/simplified_buildings/"



# --------------------
# PIPELINE
# --------------------
def process_single_region_buildings(region_id: int):
    print("----", "Processing region:", region_id, datetime.datetime.now())

    # Load pre-split buildings for this region
    in_path = f"{input_buildings_dir}buildings_{region_id}.parquet"
    buildings = gpd.read_parquet(in_path)

    # Run cleaning / simplification
    buildings = process_region_buildings(
        buildings,
        simplify=True,
        simplification_tolerance=0.1,  # meters (EPSG:25832)
        merge_limit=25,
    )

    # Save output
    out_path = f"{buildings_dir}buildings_{region_id}.parquet"
    buildings.to_parquet(out_path)
    print("Wrote:", out_path)


def process_region_buildings(
    buildings: gpd.GeoDataFrame,
    simplify: bool,
    simplification_tolerance: float = 0.1,
    merge_limit: int = 25,
) -> gpd.GeoDataFrame:
    """
    Pass the region buildings through the geoplanar simplification pipeline.
    Assumes / enforces EPSG:25832 for metric operations.
    """
    # Ensure metric CRS (EPSG:25832)
    if buildings.crs is None:
        # If you *know* they are 25832 but CRS is missing:
        buildings = buildings.set_crs(epsg=25832)
    elif buildings.crs.to_epsg() != 25832:
        buildings = buildings.to_crs(25832)

    initial_shape = buildings.shape

    # Fix invalids
    buildings["geometry"] = buildings.make_valid()

    # Explode multipolygons -> keep only Polygons
    buildings = buildings.explode(ignore_index=True)
    buildings = buildings[buildings.geometry.geom_type == "Polygon"].reset_index(drop=True)

    # Simplify (tolerance in meters) + normalize
    if simplify:
        buildings["geometry"] = buildings.simplify(simplification_tolerance).normalize()

    # Drop very large polygons (area in m²)
    buildings = buildings[buildings.area < 200_000].reset_index(drop=True)

    # Merge overlaps, then trim remaining overlaps
    buildings = geoplanar.merge_overlaps(
        buildings, merge_limit=merge_limit, overlap_limit=0.1
    )
    buildings = geoplanar.trim_overlaps(buildings, largest=False)

    # Clean any multipolygons that may have emerged
    buildings = buildings.explode(ignore_index=True)
    buildings = buildings[buildings.geom_type == "Polygon"].reset_index(drop=True)

    # Merge touching after slight shrink (0.5 m)
    shrink = buildings.buffer(-0.5, resolution=2)
    buildings = geoplanar.merge_touching(
        buildings, np.where(shrink.is_empty), largest=True
    )

    # Clean again
    buildings = buildings.explode(ignore_index=True)
    buildings = buildings[buildings.geom_type == "Polygon"].reset_index(drop=True)

    # Final validity pass
    if simplify:
        buildings["geometry"] = buildings.simplify(simplification_tolerance)
        buildings["geometry"] = buildings.make_valid()
        buildings = buildings[buildings.geom_type == "Polygon"].reset_index(drop=True)

    print(
        "Final polygons:",
        buildings.shape[0],
        ", dropped fraction:",
        1 - (buildings.shape[0] / initial_shape[0]),
    )

    buildings["geometry"] = buildings.normalize()
    return buildings

# --------------------
# ENTRY POINT
# --------------------
if __name__ == "__main__":
    process_single_region_buildings(REGION_ID)

gdf = gpd.read_parquet('D:/Work/Github_Morphotopes/data/simplified_buildings/buildings_0.parquet')
gdf.to_file('D:/Work/Github_Morphotopes/data/simplified_buildings/buildings_0.gpkg', driver="GPKG")  