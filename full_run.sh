# pixi run python src/core/generate_buildings.py >> buildings_log.txt
# cp -r /data/uscuni-ulce/processed_data/simplified_buildings/ /data/uscuni-ulce/processed_data/buildings/
#pixi run python src/core/generate_streets.py >> streets_log.txt
#pixi run python src/core/generate_elements.py >> elements_log.txt
#pixi run python src/core/generate_ngraphs.py >> ngraphs_log.txt
#pixi run python src/core/generate_chars.py >> chars_log.txt
#pixi run python src/core/generate_merged_primary_chars.py >> merged_log.txt
pixi run python src/core/generate_clusters.py >> morphotopes_log.txt
