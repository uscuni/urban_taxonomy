## download buildings
## download streets
# remove old assigned buildings if alvailable - rm /data/uscuni-ulce/regions/buildings/*
## generate regions and spread out buildings
## run below



#clean previous processed data, so there are no conflicts
#  enclosures   neigh_graphs          streets        umap_embeddings chars      morphotopes  simplified_buildings  tessellations
# remove /buildings/ folder



pixi run python src/core/generate_buildings.py > buildings_log.txt
# run this in bash: pixi run python src/core/generate_buildings.py *> D:\Work\Github_Morphotopes\data\bash_txt_files\buildings_log.txt

# To copy Simplified_buildings folder into builings folder
cp -r D:/Work/Github_Morphotopes/data/simplified_buildings/ D:/Work/Github_Morphotopes/data/buildings/

# run prcessing_socialist apartments

pixi run python src/core/generate_streets.py > &> D:/Work/Github_Morphotopes/data/bash_txt_files/streets_log.txt
# run this in bash: pixi run python src/core/generate_streets.py *> D:\Work\Github_Morphotopes\data\bash_txt_files\streets_log.txt


###### DID BEFORE'''''''''''''''''''''''''''''''''''''''''''''
pixi run python src/core/generate_elements.py &> D:/Work/Github_Morphotopes/data/bash_txt_files/elements_log.txt
# run this in bash: pixi run python src/core/generate_ngraphs.py *> D:\Work\Github_Morphotopes\data\bash_txt_files\elements_log.txt

pixi run python src/core/generate_ngraphs.py &> D:/Work/Github_Morphotopes/data/bash_txt_files/ngraphs_log.txt
# run this in bash: pixi run python src/core/generate_ngraphs.py *> D:\Work\Github_Morphotopes\data\bash_txt_files\ngraphs_log.txt

pixi run python src/core/generate_chars.py &> D:/Work/Github_Morphotopes/data/bash_txt_files/chars_log.txt
# run this in bash: pixi run python src/core/generate_chars.py *> D:\Work\Github_Morphotopes\data\bash_txt_files\chars_log.txt




#pixi run python src/core/generate_merged_primary_chars.py &> D:/Work/Github_Morphotopes/data/bash_txt_files/merged_log.txt

#pixi run python src/core/generate_clusters.py &> D:/Work/Github_Morphotopes/data/bash_txt_files/morphotopes_log.txt

