# Urban landscape of Central Europe

Data product

# To run:

1. Clone this repository.

2. Run `pixi install`, then `pixi run build` and optionally `pixi run tests` . These commands setup the enviroment and all the required packages.

    - Alternatively you can manually install the conda-forge dependencies, but you have to still run the pixi build and tests commands:
           `pixi add momepy umap-learn fast_hdbscan jupyterlab pyarrow matplotlib lonboard folium mapclassify datashader  dask pip sidecar glasbey scikit-image colorcet pandas holoviews bokeh=3.1 esda pytest hdbscan`

4. To run jupyter use either `pixi run jupyter lab` or pass extra arguments like `pixi run jupyter lab --port 8888`.

5. Run the process_regions.ipynb notebook to generate all elements, graph and characters for a particular region.

6. Run the clustering.ipynb notebook to generate and store clusters.
