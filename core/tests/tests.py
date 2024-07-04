import geopandas as gpd
import pandas as pd
from libpysal.graph import Graph


class TestDimensions:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_buildings = gpd.read_file(test_file_path, layer="buildings")
        self.df_streets = gpd.read_file(test_file_path, layer="streets")
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
        self.graph = (
            Graph.build_contiguity(self.df_tessellation)
            .higher_order(k=3, lower_order=True)
            .assign_self_weight()
        )

    def test_lazy_higher_order(self):
        res = pd.Series(np.nan, index=graph.unique_ids)
        for partial_higher in lazy_higher_order_scipy(graph, k=3, n_splits=2):
            partial_focals = np.setdiff1d(
                partial_higher.unique_ids, partial_higher.isolates
            )

            partial_result = partial_higher.describe(
                df_tessellation["area"][partial_higher.unique_ids], statistics=["sum"]
            )["sum"]

            res[partial_focals] = partial_result[partial_focals]
