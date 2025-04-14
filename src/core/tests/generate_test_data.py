import momepy as mm
import geopandas as gpd
from core.generate_streets import record_batch_reader, read_overture_region_streets
from shapely import box

def generate_test_data():

    region_id = 'temp_region'
    regions_buildings_dir = buildings_dir = overture_streets_dir = enclosures_dir = graph_dir = tesselations_dir = streets_dir = chars_dir = 'test/'
    test_file_path = mm.datasets.get_path("bubenec")
    buildings = gpd.read_file(test_file_path, layer="buildings")

    # download and save buildings
    batches = record_batch_reader('building', buildings.to_crs(epsg=4326).total_bounds.tolist()).read_all()
    gdf = gpd.GeoDataFrame.from_arrow(batches)
    gdf['source'] = gdf['sources'].str[0].str['dataset']
    gdf = gdf.set_crs(epsg=4326)
    gdf.to_crs(epsg=3035).to_parquet(chars_dir + f'buildings_{region_id}.pq')

    # download and save streets
    region_hull = box(*buildings.to_crs(epsg=4326).total_bounds)
    streets = read_overture_region_streets(region_hull, region_id)
    streets.to_parquet(streets_dir + f'streets_{region_id}.pq')
    
if __name__ == '__main__':
    generate_test_data()