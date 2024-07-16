import datetime
import gc
import glob
from typing import List, Optional
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as fs
import json
import os
import sys
from typing import Optional
import pyarrow.parquet as pq
import shapely.wkb
import geopandas as gpd

regions_datadir = "/data/uscuni-ulce/"
data_dir = "/data/uscuni-ulce/processed_data/"
eubucco_files = glob.glob(regions_datadir + "eubucco_raw/*")


def process_all_regions_streets():
    region_hulls = gpd.read_parquet(
        regions_datadir + "regions/" + "regions_hull.parquet"
    )

    for region_id, region_hull in region_hulls.to_crs('epsg:4326').iterrows():
        region_hull = region_hull["convex_hull"]

        if region_id != 69300: continue

        print("----", "Processing region: ", region_id, datetime.datetime.now())

        ## processs streets
        streets = process_region_streets(region_hull, region_id)

        ## save streets
        streets.to_parquet(data_dir + f"streets/streets_{region_id}.parquet")
        del streets
        gc.collect()


def process_region_streets(region_hull, region_id):
    streets = read_overture_region_streets(region_hull, region_id)
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
    streets = streets[~streets.road.str.contains('is_tunnel').fillna(False)]
    streets = streets.set_crs(epsg=4326).to_crs(epsg=3035)
    streets = streets.sort_values('id')[['id', 'geometry', 'class']].reset_index(drop=True)
    return streets

def read_overture_region_streets(region_hull, region_id):

    batches = record_batch_reader('segment', region_hull.bounds).read_all()
    gdf = gpd.GeoDataFrame.from_arrow(batches)
    gdf = gdf.iloc[gdf.sindex.query(region_hull, predicate='intersects')]
    return gdf

def read_region_streets(region_hull, region_id):
    read_mask = region_hull.buffer(100)

    streets = gpd.read_parquet(
        regions_datadir + "streets/central_europe_streets_eubucco_crs.parquet"
    )
    streets = streets[streets.intersects(read_mask)].reset_index(drop=True)

    return streets



## from overturemaps-py
def record_batch_reader(overture_type, bbox=None) -> Optional[pa.RecordBatchReader]:
    """
    Return a pyarrow RecordBatchReader for the desired bounding box and s3 path
    """
    path = _dataset_path(overture_type)

    if bbox:
        xmin, ymin, xmax, ymax = bbox
        filter = (
            (pc.field("bbox", "xmin") < xmax)
            & (pc.field("bbox", "xmax") > xmin)
            & (pc.field("bbox", "ymin") < ymax)
            & (pc.field("bbox", "ymax") > ymin)
        )
    else:
        filter = None

    dataset = ds.dataset(
        path, filesystem=fs.S3FileSystem(anonymous=True, region="us-west-2")
    )
    batches = dataset.to_batches(filter=filter)

    # to_batches() can yield many batches with no rows. I've seen
    # this cause downstream crashes or other negative effects. For
    # example, the ParquetWriter will emit an empty row group for
    # each one bloating the size of a parquet file. Just omit
    # them so the RecordBatchReader only has non-empty ones. Use
    # the generator syntax so the batches are streamed out
    non_empty_batches = (b for b in batches if b.num_rows > 0)

    geoarrow_schema = geoarrow_schema_adapter(dataset.schema)
    reader = pa.RecordBatchReader.from_batches(geoarrow_schema, non_empty_batches)
    return reader


def geoarrow_schema_adapter(schema: pa.Schema) -> pa.Schema:
    """
    Convert a geoarrow-compatible schema to a proper geoarrow schema

    This assumes there is a single "geometry" column with WKB formatting

    Parameters
    ----------
    schema: pa.Schema

    Returns
    -------
    pa.Schema
    A copy of the input schema with the geometry field replaced with
    a new one with the proper geoarrow ARROW:extension metadata

    """
    geometry_field_index = schema.get_field_index("geometry")
    geometry_field = schema.field(geometry_field_index)
    geoarrow_geometry_field = geometry_field.with_metadata(
        {b"ARROW:extension:name": b"geoarrow.wkb"}
    )

    geoarrow_schema = schema.set(geometry_field_index, geoarrow_geometry_field)

    return geoarrow_schema


type_theme_map = {
    "locality": "admins",
    "locality_area": "admins",
    "administrative_boundary": "admins",
    "building": "buildings",
    "building_part": "buildings",
    "division": "divisions",
    "division_area": "divisions",
    "place": "places",
    "segment": "transportation",
    "connector": "transportation",
    "infrastructure": "base",
    "land": "base",
    "land_cover": "base",
    "land_use": "base",
    "water": "base",
}


def _dataset_path(overture_type: str) -> str:
    """
    Returns the s3 path of the Overture dataset to use. This assumes overture_type has
    been validated, e.g. by the CLI

    """
    # Map of sub-partition "type" to parent partition "theme" for forming the
    # complete s3 path. Could be discovered by reading from the top-level s3
    # location but this allows to only read the files in the necessary partition.
    theme = type_theme_map[overture_type]
    return f"overturemaps-us-west-2/release/2024-06-13-beta.1/theme={theme}/type={overture_type}/"
    

if __name__ == "__main__":
    process_all_regions_streets()
