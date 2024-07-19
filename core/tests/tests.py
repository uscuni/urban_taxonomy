import geopandas as gpd
import pandas as pd
from libpysal.graph import Graph
from core.utils import lazy_higher_order, partial_apply
import momepy as mm
import numpy as np
from pandas.testing import assert_series_equal


class TestUtils:
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.df_tessellation["area"] = self.df_tessellation.geometry.area
        
        self.cont_graph1 = Graph.build_contiguity(self.df_tessellation).assign_self_weight()
        self.cont_graph3 = self.cont_graph1.higher_order(k=3, lower_order=True, diagonal=True)
        
        self.fuzzy_graph1 = Graph.build_fuzzy_contiguity(
            self.df_tessellation, buffer=1e-6
        ).assign_self_weight()
        self.fuzzy_graph3 = self.fuzzy_graph1.higher_order(k=3, lower_order=True, diagonal=True)
        
        self.knn_graph1 = Graph.build_knn(self.df_tessellation.centroid, k=3).assign_self_weight()
        self.knn_graph3 = self.knn_graph1.higher_order(k=3, lower_order=True, diagonal=True)

    def test_lazy_higher_order_apply(self):

        def sum_area(partical_focals, partial_higher, y):
            return partial_higher.describe(
                y.loc[partial_higher.unique_ids], statistics=["sum"]
            )["sum"]

        ## contiguity equivalence
        res = partial_apply(
        graph=self.cont_graph1,
        higher_order_k=3,
        n_splits=2,
        func=sum_area,
        y=self.df_tessellation["area"],
        )
        expected = self.cont_graph3.describe(self.df_tessellation["area"], statistics=["sum"])["sum"]
        assert_series_equal(res, expected, check_names=False)

        ## fuzzy equivalence
        res = partial_apply(
        graph=self.fuzzy_graph1,
        higher_order_k=3,
        n_splits=2,
        func=sum_area,
        y=self.df_tessellation["area"],
        )
        expected = self.fuzzy_graph3.describe(self.df_tessellation["area"], statistics=["sum"])["sum"]
        assert_series_equal(res, expected, check_names=False)

        ## knn equivalence
        res = partial_apply(
            graph=self.knn_graph1,
            higher_order_k=3,
            n_splits=2,
            func=sum_area,
            y=self.df_tessellation["area"],
        )
        expected = self.knn_graph3.describe(self.df_tessellation["area"], statistics=["sum"])["sum"]
        assert_series_equal(res, expected, check_names=False)


        ### custom indices
        # string
        string_tess = self.df_tessellation.set_index(map(str, self.df_tessellation.index.values))
        graph1 = Graph.build_contiguity(string_tess, rook=False).assign_self_weight()
        graph3 = graph1.higher_order(k=3, lower_order=True, diagonal=True)
        
        old_expected = self.cont_graph3.describe(self.df_tessellation["area"], statistics=["sum"])["sum"]
        new_expected = graph3.describe(string_tess["area"], statistics=["sum"])["sum"]
        assert_series_equal(old_expected, new_expected, check_index=False)
        res = partial_apply(
            graph=graph1, higher_order_k=3, n_splits=2, func=sum_area, y=string_tess["area"]
        )
        assert_series_equal(new_expected, res, check_names=False)


        ## negative
        ii = self.df_tessellation.index.values
        ii[:10] = np.arange(-10, 0)
        neg_tess = self.df_tessellation.set_index(ii)
        graph1 = Graph.build_contiguity(neg_tess, rook=False).assign_self_weight()
        graph3 = graph1.higher_order(k=3, lower_order=True, diagonal=True)
        
        new_expected = graph3.describe(neg_tess["area"], statistics=["sum"])["sum"]
        assert_series_equal(old_expected, new_expected, check_index=False)
        
        
        res = partial_apply(
            graph=graph1, higher_order_k=3, n_splits=2, func=sum_area, y=neg_tess["area"]
        )
        assert_series_equal(new_expected, res, check_names=False)
