echo "Running generate_streets.py"
pixi run python src/core/generate_streets.py &> streets_log.txt
echo "Running generate_elements.py"
pixi run python src/core/generate_elements.py &> elements_log.txt
echo "Running generate_ngraphs.py"
pixi run python src/core/generate_ngraphs.py &> ngraphs_log.txt
echo "Running generate_chars.py"
pixi run python src/core/generate_chars.py &> chars_log.txt
echo "Running generate_merged_primary_chars.py"
pixi run python src/core/generate_merged_primary_chars.py &> merged_log.txt
echo "Running generate_clusters.py"
pixi run python src/core/generate_clusters.py &> morphotopes_log.txt