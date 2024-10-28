import datetime
import glob

import geopandas as gpd
import geoplanar
import numpy as np
import pandas as pd
from core.utils import largest_regions

regions_datadir = "/data/uscuni-ulce/"
data_dir = "/data/uscuni-ulce/processed_data/"
eubucco_files = glob.glob(regions_datadir + "eubucco_raw/*")
buildings_dir = '/data/uscuni-ulce/processed_data/simplified_buildings/'
regions_buildings_dir = '/data/uscuni-ulce/regions/buildings/'

def process_regions(largest):

    region_hulls = gpd.read_parquet(
        regions_datadir + "regions/" + "cadastre_regions_hull.parquet"
    )

    if largest:
        for region_id in largest_regions:
            process_single_region_buildings(region_id)
            
    else:
        regions_hulls = region_hulls[~region_hulls.index.isin(largest_regions)]
        from joblib import Parallel, delayed
        n_jobs = -1
        new = Parallel(n_jobs=n_jobs)(
            delayed(process_single_region_buildings)(region_id) for region_id, _ in regions_hulls.iterrows()
        )


def process_single_region_buildings(region_id):
    print("----", "Processing region: ", region_id, datetime.datetime.now())
    buildings = gpd.read_parquet(regions_buildings_dir + f'buildings_{region_id}.pq')
    buildings = process_region_buildings(buildings, True, simplification_tolerance=.1, merge_limit=25)

    ## drop buildings that intersect streets
    if region_id in [55763, 16242]:
        buildings = drop_buildings_intersecting_streets(buildings, region_id)
    
    buildings.to_parquet(buildings_dir + f"buildings_{region_id}.parquet")


def process_region_buildings(buildings, simplify, simplification_tolerance=.1, merge_limit=25):
    '''Pass the region buildings through the geoplanar simplification pipeline.'''
    
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
    ## one region - 109491 - has an issue with simplification, without normalisation
    if simplify:
        buildings["geometry"] = buildings.simplify(simplification_tolerance).normalize()

    # drop very large buildings
    buildings = buildings[buildings.area < 200_000].reset_index(drop=True)

    
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



def drop_buildings_intersecting_streets(buildings, region_id):
    
    from core.generate_streets import to_drop_tunnel
    streets = gpd.read_parquet('/data/uscuni-ulce/overture_streets/streets_55763.pq')
        
    ## service road removed
    approved_roads = ['living_street',
                     'motorway',
                     'motorway_link',
                     'pedestrian',
                     'primary',
                     'primary_link',
                     'residential',
                     'secondary',
                     'secondary_link',
                     'tertiary',
                     'tertiary_link',
                     'trunk',
                     'trunk_link',
                     'unclassified']
    streets = streets[streets['class'].isin(approved_roads)]
    
    
    ## drop tunnels
    to_filter = streets.loc[~streets.road_flags.isna(), ].set_crs(epsg=4236).to_crs(epsg=3035)
    tunnels_to_drop = to_filter.apply(to_drop_tunnel, axis=1)
    streets = streets.drop(to_filter[tunnels_to_drop].index)
    
    streets = streets.set_crs(epsg=4326).to_crs(epsg=3035)
    streets = streets.sort_values('id')[['id', 'geometry', 'class']].reset_index(drop=True)
    blg_idxs, str_idxs = streets.buffer(.3).sindex.query(buildings.geometry.to_crs(epsg=3035),
                                              predicate='intersects')
    return buildings[~buildings.index.isin(blg_idxs)].reset_index(drop=True)
    


if __name__ == "__main__":
    process_regions(False)
    process_regions(True)
