import geopandas as gpd
import pandas as pd
from libpysal.graph import Graph
from core.utils import lazy_higher_order, partial_apply, partial_describe_reached_agg, partial_mean_intb_dist 
from core.generate_context import parallel_higher_order_context, spatially_weighted_partial_lag
from core.generate_elements import generate_enclosures_representative_points
import momepy as mm
import numpy as np
from pandas.testing import assert_series_equal
import shapely

class TestCore:
    
    def setup_method(self):
        test_file_path = mm.datasets.get_path("bubenec")
        self.df_tessellation = gpd.read_file(test_file_path, layer="tessellation")
        self.edges = gpd.read_file(test_file_path, layer="streets")
        self.buildings = gpd.read_file(test_file_path, layer="buildings")
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


    def test_describe_reached_agg(self):
        
        tess_nid = mm.get_nearest_street(
            self.df_tessellation, self.edges
        )
        edges_graph = Graph.build_contiguity(self.edges, rook=False).assign_self_weight()
        res = partial_describe_reached_agg(
            self.df_tessellation.geometry.area,
            tess_nid,
            edges_graph,
            higher_order=3,
            n_splits=30,
            q=None,
            statistics=["sum", "count"],
        )
        higher = edges_graph.higher_order(k=3, lower_order=True, diagonal=True)
        expected_res = mm.describe_reached_agg(
            self.df_tessellation.geometry.area,
            tess_nid,
            higher,
            q=None,
            statistics=["sum", "count"]
        )
        assert_series_equal(res['sum'], expected_res['sum'], check_names=False)
        assert_series_equal(res['count'], expected_res['count'], check_names=False)

    def test_intb(self):

        bgraph = Graph.build_contiguity(self.buildings, rook=False).assign_self_weight()
        res = partial_apply(
            graph=self.fuzzy_graph1,
            higher_order_k=3,
            n_splits=20,
            func=partial_mean_intb_dist,
            buildings=self.buildings,
            bgraph=bgraph,
        )
        higher = self.fuzzy_graph1.higher_order(k=3, lower_order=True, diagonal=True)
        expected_res = mm.mean_interbuilding_distance(self.buildings, bgraph, higher)
        assert_series_equal(res, expected_res, check_names=False)


    def test_context(self):

        spatial_lag = 3
        context = parallel_higher_order_context(
            self.df_tessellation[['area']], self.fuzzy_graph1, k=spatial_lag, n_splits=10, output_vals=3
        )
        context.columns = np.concatenate(
            [(c + "_lower", c + "_median", c + "_higher") for c in self.df_tessellation[['area']].columns]
        )
        higher = self.fuzzy_graph1.higher_order(k=spatial_lag, lower_order=True, diagonal=True)
        r = higher.describe(self.df_tessellation['area'], statistics=['median'])['median']
        assert_series_equal(context['area_median'], r, check_names=False)

    def test_spatially_weighted_context(self):
        
        spatial_lag = 3
        centroids = shapely.get_coordinates(self.df_tessellation.representative_point())
        n_splits=10
        context = spatially_weighted_partial_lag(self.df_tessellation[['area']],
                                                 self.fuzzy_graph1, centroids, 'inverse', 
                                                 k=spatial_lag, n_splits=n_splits)

        higher = self.fuzzy_graph1.higher_order(k=spatial_lag, lower_order=True)
        from shapely import distance
        centroids = self.df_tessellation.representative_point()
        def _distance_decay_weights(group):
            focal = group.index[0][0]
            neighbours = group.index.get_level_values(1)
            distances = distance(centroids.loc[focal], centroids.loc[neighbours])
            distance_decay = 1 / distances
            return distance_decay.values
        
        decay_graph = higher.transform(_distance_decay_weights)
        expected_context = mm.percentile(self.df_tessellation['area'], decay_graph, q=[50])

        isolates = self.fuzzy_graph1.assign_self_weight(0).isolates
        assert_series_equal(expected_context.drop(isolates)[50], 
                            context.drop(isolates).iloc[:, 0], 
                            check_names=False)

    def test_tess_simplification_env_setup(self):
        enclosures = generate_enclosures_representative_points(self.buildings, self.edges)
        tessellation = mm.enclosed_tessellation(self.buildings, enclosures.geometry, simplify=True)
        assert True # only testing if the above 2 functions work in the environment
        