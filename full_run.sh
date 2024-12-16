## download buildings
## download streets
# remove old assigned buildings if alvailable - rm /data/uscuni-ulce/regions/buildings/*
## generate regions and spread out buildings
## run below



#clean previous processed data, so there are no conflicts
#  enclosures   neigh_graphs          streets        umap_embeddings chars      morphotopes  simplified_buildings  tessellations
# remove /buildings/ folder



# pixi run python src/core/generate_buildings.py > buildings_log.txt
# cp -r /data/uscuni-ulce/processed_data/simplified_buildings/ /data/uscuni-ulce/processed_data/buildings/
# run prcessing_socialist apartments
#pixi run python src/core/generate_streets.py > streets_log.txt
# pixi run python src/core/generate_elements.py &> elements_log.txt
# pixi run python src/core/generate_ngraphs.py &> ngraphs_log.txt


# pixi run python src/core/generate_chars.py &> chars_log.txt
# pixi run python src/core/generate_merged_primary_chars.py &> merged_log.txt

pixi run python src/core/generate_clusters.py &> morphotopes_log.txt

