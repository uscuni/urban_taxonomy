# Urban landscape of Central Europe

Notebooks, enrivonment and code to generate the taxonomy of central european urban fabric.

# To run:

1. Clone this repository.

2. Run `pixi install`, then `pixi run build` and optionally `pixi run tests` . These commands setup the enviroment and all the required packages.

    - Alternatively you can manually install the conda-forge dependencies, but you have to still run the pixi build and tests commands:
           `pixi add momepy umap-learn fast_hdbscan jupyterlab pyarrow matplotlib lonboard folium mapclassify datashader  dask pip sidecar glasbey scikit-image colorcet pandas holoviews bokeh=3.1 esda pytest hdbscan`

4. To run jupyter use either `pixi run jupyter lab` or pass extra arguments like `pixi run jupyter lab --port 8888`.

5. To run the analysis on the whole dataset - first, make sure you have the correct folder structure in place. Then, run:

    - `code/download_buildings.ipynb` to download all the cadastre data for central europe
    - `code/explore_cadastre_data.ipynb` to standardise all the cadstre data from different countries into a single format
    - `code/generate_regions.ipynb` to split the buildings into regions for independent processing
    - `code/download_streets.ipynb` to download the raw overture streets for every region
    - `code/processing_apartment_blocks.ipynb` to update socialist housing in Czechia ( needs to be run after building simplification)
    - `bash full_run.sh` to run the entire processing pipeline from building, street preprocessing, element generation, characters calculations and morphotope creation.
    - `code/divisive_kmeans.ipynb` to generate the heirarchy of morphotopes.
    - `code/noise.ipynb` to assign the noise points to the nearest clusters.
    - `code/cluster_naming.ipynb` to see the cluster naming notebook.
    - `code/results.ipynb` to generate comparisons with other data products and figures.

    - `code/add_regions_from_new_buildings.ipynb` to process a new set of buildings - split into regions, name the regions and add them to the existing regions directory.
    - `code/process_region.ipynb` to process individual regions or groups of specific regions sequentially or in parallel. 

    - `code/cluster_exploration.ipynb` to map specific regions.
    - `code/interactive_chars_exploration.ipynb` to interactively plot characters in specific regions.