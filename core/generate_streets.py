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
import geoplanar

regions_datadir = '/data/uscuni-ulce/'
data_dir = '/data/uscuni-ulce/processed_data/'
eubucco_files = glob.glob(regions_datadir + 'eubucco_raw/*')

def process_regions():
    
    region_hulls = gpd.read_parquet(regions_datadir + 'regions/' + 'regions_hull.parquet')

    for region_id, region_hull in region_hulls.iterrows():
        
        region_hull = region_hull['convex_hull']

        #if region_id != 12199: continue


        print('----', 'Processing region: ', region_id, datetime.datetime.now())

        streets = read_region_streets(region_hull, region_id)

        ## processs streets

        ## save streets
        streets.to_parquet(data_dir + f'streets/streets_{region_id}.parquet')
        del streets
        gc.collect()


def read_region_streets(region_hull, region_id):

    read_mask = region_hull.buffer(100)
    
    streets = gpd.read_parquet(regions_datadir + 'streets/central_europe_streets_eubucco_crs.parquet')
    streets = streets[streets.intersects(read_mask)].reset_index(drop=True)
    
    return streets

if __name__ == '__main__':
    process_regions()
