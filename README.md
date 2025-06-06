# Urban landscape of Central Europe

Notebooks, enrivonment and code to generate the taxonomy of Central European Urban Fabric.

# Reproducing the paper

1. First, make sure you have the correct folder structure in place or you will have to change it in the python scripts/notebooks.

2. Clone this repository and checkout the ``results`` branch.

3. Run `pixi install`, then `pixi run build` and optionally `pixi run tests` . These commands setup the enviroment and all the required packages.

    - Alternatively you can manually install the conda-forge dependencies, but you have to still run the pixi build and tests commands:
           `pixi add momepy umap-learn fast_hdbscan jupyterlab pyarrow matplotlib lonboard folium mapclassify datashader  dask pip sidecar glasbey scikit-image colorcet pandas holoviews bokeh=3.1 esda pytest hdbscan`

Then, run:
1. `code/download_buildings.ipynb` to download all the cadastre data for central europe
2. `code/explore_cadastre_data.ipynb` to standardise all the cadstre data from different countries into a single format
3. `code/generate_regions.ipynb` to split the buildings into regions for independent processing
4. `code/download_streets.ipynb` to download the raw overture streets for every region
5. `code/processing_apartment_blocks.ipynb` to update socialist housing in Czechia ( needs to be run after building simplification)
6. `bash full_run.sh` to run the entire processing pipeline from building, street preprocessing, element generation, characters calculations and morphotope creation.
7. `code/morphotope_postprocessing.ipynb` to fix morphotope geometries based on adjacency.
8. `code/morphotope_chars.ipynb` to generate characteristics specific to morphotopes.
9. `code/clustering.ipynb` to generate the heirarchy of morphotopes, and store the data.
10. `code/noise.ipynb` to assign the noise points to the nearest clusters.
11. `code/results.ipynb` to generate comparisons with other data products and figures.

Additional notebooks:

- `code/process_region.ipynb` to process individual regions or groups of specific regions sequentially or in parallel.

- `code/cluster_exploration.ipynb` to map specific regions and explore the cluster assignments.
- `code/interactive_chars_exploration.ipynb` to interactively plot characters in specific regions.

# Extending the hierarchy or running from main

## Running from main
1. Clone the repository
2. Run `pixi install`, then optionally `pixi run generate_test_data` and `pixi run tests`.
3. Follow the same structure as above.

## Adding additional regions
To extend the hierarchy with new data:
1. Download the new building footprints first
2. `code/add_regions_from_new_buildings.ipynb` to process the new set of buildings - split buildings into regions, name the regions and add them to the existing regions directory.
2. `code/download_streets.ipynb` to download the streets for the new regions from overture
3. `code/process_region.ipynb` for the new regions to run the full processing pipeline - building and street processing, generating elements, morphometric characteristics and morphotope deliniation.
4. `code/morphotope_postprocessing.ipynb` to fix morphotope geometries based on adjacency.
5. `code/morphotope_chars.ipynb` to generate characteristics specific to morphotopes.
Then either create a new clustering or assign the new morphotopes to the existing hierarchy directly.

# Generating additional data
To generate the PM tiles, website assests and the data product:

- `code/generate_pmtiles.ipynb` to generate pm tiles from the clustering results.
- `code/tree_for_viz.ipynb` to generate the data for the website taxonomy visualisations
- `code/data_products.ipynb` to generate the data products
- `code/data_products_guide.ipynb` - guide how to use the data products.
