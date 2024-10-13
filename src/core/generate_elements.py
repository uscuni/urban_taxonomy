import datetime
import gc
import glob

import geopandas as gpd
import momepy as mm
import numpy as np
from libpysal.graph import Graph
from shapely import GEOSException
from shapely import Point

regions_buildings_dir = '/data/uscuni-ulce/regions/buildings/'
buildings_dir = '/data/uscuni-ulce/processed_data/buildings/'
overture_streets_dir = '/data/uscuni-ulce/overture_streets/'
streets_dir = '/data/uscuni-ulce/processed_data/streets/'
enclosures_dir = '/data/uscuni-ulce/processed_data/enclosures/'
tessellations_dir = '/data/uscuni-ulce/processed_data/tessellations/'
regions_datadir = '/data/uscuni-ulce/'


def process_all_regions_elements():

    region_hulls = gpd.read_parquet(
            regions_datadir + "regions/" + "cadastre_regions_hull.parquet"
    )

    ## run one region at a time, since the inner function is parallelized
    for region_id, region_hull in region_hulls.iterrows():

        print("----", "Processing region: ", region_id, datetime.datetime.now())
        enclosures, tesselations = process_region_elements(buildings_dir, streets_dir, region_id)

        enclosures.to_parquet(enclosures_dir + f"enclosure_{region_id}.parquet")
        print("Processed enclosures")

        ## save files
        tesselations.to_parquet(
            tessellations_dir + f"tessellation_{region_id}.parquet"
        )
        print("processed tesselations")

        del enclosures
        del tesselations
        gc.collect()


def process_region_elements(buildings_data_dir, streets_data_dir, region_id):
    n_workers = -1

    print("----", "Processing region: ", region_id, datetime.datetime.now())
    buildings = gpd.read_parquet(
        buildings_data_dir + f"buildings_{region_id}.parquet"
    )
    streets = gpd.read_parquet(streets_data_dir + f"streets_{region_id}.parquet")
    enclosures = generate_enclosures_representative_points(buildings, streets)


    tesselations = None

    ## we need a try/except block here because shapely.voronoi_polygons throws a top. exception
    ## for some buildings and there is no way to filter them out before running it
    ## there are only 2 buildings in the entire dataset that have the issue in regions - 47090,21894
    try:
    
        tesselations = generate_tess(buildings,
                             enclosures,
                             n_workers=-1)
    
    except GEOSException as e:
        txt = str(e)
        print('Problem with topology, ', txt)
        problem_point_coords = [float(s) for s in txt[45:].split('. T')[0].split(' ')]
        problem_point = Point(*problem_point_coords)
        problem_building = buildings.iloc[buildings.sindex.query(problem_point, predicate='intersects')]
        buildings = buildings.drop(buildings.sindex.query(problem_point, predicate='intersects'))
        tesselations = generate_tess(buildings,
                             enclosures,
                             n_workers=-1)



    ### there are some edge cases for long and narrow buildings and
    ## completely wrong polygons that are dropped by voronoi_frames
    ## region 10 has this problem
    tesselation_coverage = np.isin(
        buildings.index.values, tesselations.index.values
    )
    num_problem_buildings = 0
    
    while (not tesselation_coverage.all()):
        print(
            "Retrying tesselation with less buildings, potentially changing building data."
        )
        ## assume all missing buildings are problematic polygons, drop them and retry the tessellation
        num_problem_buildings += (~tesselation_coverage).sum()
        buildings = buildings[tesselation_coverage].reset_index()
        enclosures = generate_enclosures_representative_points(buildings, streets)
        tesselations = generate_tess(buildings, enclosures, n_workers=-1)
        tesselation_coverage = np.isin(
            buildings.index.values, tesselations.index.values
        )

    # quality check, there should be at least one tess cell per building in the end.
    assert tesselation_coverage.all()
    
    # if this results in a correct tesselation, save the new region buildings
    if num_problem_buildings >= 1:
        print(
            "Dropping",
            num_problem_buildings,
            "buildings due to tesselation problems",
        )
        buildings.to_parquet(
            buildings_data_dir + f"buildings_{region_id}.parquet"
        )



    # free some memory
    del buildings
    del streets
    gc.collect()

    return enclosures, tesselations


def generate_tess(buildings, enclosures, n_workers=1):
    tessellation = mm.enclosed_tessellation(
        buildings, enclosures.geometry, simplify=True, n_jobs=n_workers
    )
    # deal with split buildings
    tessellation = tessellation.dissolve(by=tessellation.index.values)

    # drop empty spaces with no buildings and a positive index,
    # leave negatives in the geodataframe
    tessellation = tessellation.explode()
    inp, res = buildings.geometry.centroid.sindex.query(tessellation.geometry)
    to_keep = np.append(np.unique(inp), np.where(tessellation.index.values < 0)[0])
    tessellation = tessellation.iloc[to_keep]

    ### drop any remaining duplicates
    ## sometimes -1s have multiple tesselation cells
    tessellation = tessellation[~tessellation.index.duplicated()].sort_index()
    return tessellation



def generate_enclosures(buildings, streets):
    ## generate additional_boundaries
    buffered_buildings = mm.buffered_limit(buildings, buffer='adaptive')
    enclosures = mm.enclosures(streets, limit=buffered_buildings, clip=True)
    return enclosures


def generate_enclosures_representative_points(buildings, streets):
    ## generate additional_boundaries
    min_buffer: float = 0
    max_buffer: float = 100

    gabriel = Graph.build_triangulation(
        buildings.representative_point(), "gabriel", kernel="identity"
    )
    max_dist = gabriel.aggregate("max")
    buffer = np.clip(max_dist / 2 + max_dist * 0.1, min_buffer, max_buffer).values
    buffered_buildings = buildings.buffer(buffer, resolution=2).union_all()

    enclosures = mm.enclosures(streets, limit=buffered_buildings, clip=True)
    return enclosures


if __name__ == "__main__":
    process_all_regions_elements()
