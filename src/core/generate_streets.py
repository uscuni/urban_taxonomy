import datetime
import geopandas as gpd
import neatnet

REGION_ID = 0
TARGET_EPSG = 25832

overture_streets_dir = "D:/Work/Github_Morphotopes/data/overture_streets/"
buildings_dir        = "D:/Work/Github_Morphotopes/data/"
out_streets_dir      = "D:/Work/Github_Morphotopes/data/streets/"

APPROVED_ROADS = [
    "living_street","motorway","motorway_link","pedestrian","primary","primary_link",
    "residential","secondary","secondary_link","tertiary","tertiary_link","trunk",
    "trunk_link","unclassified"
]

def to_drop_tunnel(row):
    tunnel_length_total = -1.0
    seg_len = row.geometry.length
    for flag in row.road_flags:
        if "values" in flag and "is_tunnel" in flag["values"]:
            tunnel_length_total = 0.0 if tunnel_length_total < 0 else tunnel_length_total
            if "between" in flag and flag["between"] is not None:
                s, e = flag["between"]
                tunnel_length_total += (e - s)
    if tunnel_length_total == 0.0:
        return True
    if tunnel_length_total > 0:
        return (tunnel_length_total * seg_len) > 50.0
    return False

print("Reading streets…")
streets = gpd.read_parquet(f"{overture_streets_dir}streets_{REGION_ID}.parquet")
if streets.crs is None:  # only if you know the source is WGS84
    streets = streets.set_crs(4326)
streets = streets.to_crs(TARGET_EPSG)

if "class" in streets.columns:
    streets = streets[streets["class"].isin(APPROVED_ROADS)].copy()

# Optional: drop tunnel segments if road_flags are present
if "road_flags" in streets.columns:
    tf = streets.loc[~streets["road_flags"].isna()].copy()
    if not tf.empty:
        mask = tf.apply(to_drop_tunnel, axis=1)
        streets = streets.drop(tf.index[mask])

# Keep minimal columns (if present)
keep = [c for c in ["id", "geometry", "class"] if c in streets.columns]
streets = streets.reset_index(drop=True)[keep]

print("Reading buildings…")
buildings = gpd.read_parquet(f"{buildings_dir}buildings_{REGION_ID}.parquet", columns=["geometry"])
if buildings.crs != streets.crs:
    buildings = buildings.to_crs(TARGET_EPSG)

print("Simplifying with neatnet…", datetime.datetime.now())
simplified = neatnet.neatify(
    streets,
    exclusion_mask=buildings.geometry,
    artifact_threshold_fallback=7,
)
out_path = f"{out_streets_dir}streets_{REGION_ID}.parquet"
simplified.to_parquet(out_path)
print("Wrote:", out_path, "| features:", len(simplified))

read_streets = gpd.read_parquet(out_streets_dir + 'streets_0.parquet' )
output = read_streets.to_file(out_streets_dir + 'streets_0.gpkg', driver='GPKG')