import momepy as mm
import geopandas as gpd
from core.generate_streets import record_batch_reader, read_overture_region_streets
import shapely

def generate_test_data():

    region_id = 'temp_region'
    region_hull = gpd.GeoSeries([shapely.Point( 14.45449, 50.07455 ,)], name='geometry').set_crs(epsg=4236).to_crs(epsg=3035).buffer(500).to_crs(epsg=4236).iloc[0]
    regions_buildings_dir = buildings_dir = overture_streets_dir = enclosures_dir = graph_dir = tesselations_dir = streets_dir = chars_dir = 'test/'
    

    # download and save buildings
    batches = record_batch_reader('building', region_hull.bounds).read_all()
    gdf = gpd.GeoDataFrame.from_arrow(batches)
    gdf['source'] = gdf['sources'].str[0].str['dataset']
    gdf = gdf.set_crs(epsg=4326)
    gdf.to_crs(epsg=3035).to_parquet(chars_dir + f'buildings_{region_id}.pq')

    # download and save streets
    streets = read_overture_region_streets(region_hull, region_id)
    streets.to_parquet(streets_dir + f'streets_{region_id}.pq')
    
if __name__ == '__main__':
    generate_test_data()