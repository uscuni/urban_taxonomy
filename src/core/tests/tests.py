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
from pandas.testing import assert_frame_equal, assert_series_equal
from geopandas.testing import assert_geoseries_equal
from libpysal.graph import read_parquet


from core.generate_buildings import read_region_buildings, process_region_buildings
from core.generate_streets import process_region_streets, process_single_region_streets
from core.generate_elements import process_region_elements, generate_enclosures_representative_points, generate_tess
from core.generate_ngraphs import process_region_graphs
from core.generate_chars import process_single_region_chars
from core.generate_merged_primary_chars import merge_into_primary
from core.generate_clusters import process_single_region_morphotopes


import glob

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
        assert True # only testing if the above 2 functions work in the environment, the real test is in momepy


    def test_pipeline_with_overture_data(self):        

        region_id = 'temp_region'
        regions_buildings_dir = overture_streets_dir = 'test/'
        buildings_dir = enclosures_dir = graph_dir = tessellations_dir = streets_dir = chars_dir = morphotopes_dir = 'test/processed_data/'
        
        # run pipeline
        buildings = gpd.read_parquet(regions_buildings_dir + f'buildings_{region_id}.pq')
        buildings = process_region_buildings(buildings, True, simplification_tolerance=.1, merge_limit=25)
        buildings.to_parquet(buildings_dir + f"buildings_{region_id}.parquet")
    
        ## processs streets
        streets = process_region_streets(region_id, overture_streets_dir, buildings_dir)
        streets.to_parquet(streets_dir + f'streets_{region_id}.parquet')
    
        # elements
        enclosures, tesselations = process_region_elements(buildings_dir, streets_dir, region_id)
        enclosures.to_parquet(enclosures_dir + f"enclosure_{region_id}.parquet")
        tesselations.to_parquet(
            tessellations_dir + f"tessellation_{region_id}.parquet"
        )
    
        #graphs
        process_region_graphs(
        region_id,
        graph_dir,
        buildings_dir,
        streets_dir,
        enclosures_dir,
        tessellations_dir,
        )
    
        #chars
        process_single_region_chars(
            region_id,
            graph_dir,
            buildings_dir,
            streets_dir,
            enclosures_dir,
            tessellations_dir,
            chars_dir
        )
    
        # #primary merge
        merge_into_primary(region_id,
            graph_dir,
            buildings_dir,
            streets_dir,
            enclosures_dir,
            tessellations_dir,
            chars_dir)

        
    def test_morphotope_delin(self):

        # use the preprocessed data, since even small changes to the input data can change the order within the tree
        regions_buildings_dir = overture_streets_dir = 'test/'
        morphotopes_dir = 'test/processed_data/'
        buildings_dir = enclosures_dir = graph_dir = tessellations_dir = streets_dir = chars_dir = 'test/actual_data/'
        region_id = 'temp_region'
        
        # # morphotope
        process_single_region_morphotopes(region_id,
            graph_dir,
            buildings_dir,
            streets_dir,
            enclosures_dir,
            tessellations_dir,
            chars_dir, morphotopes_dir)

    def test_pipeline_values(self):
        
        test_dir = 'test/processed_data/'
        actual_dir = 'test/actual_data/'
        
        tess_chars = [
         'sdcLAL',
         'sdcAre',
         'sscCCo',
         'sscERI',
         'mtcWNe',
         'sicCAR']

        # test graphs
        for test_file in glob.glob(test_dir + '*graph*'):
            test_data = read_parquet(test_file)
            expected_data = read_parquet(actual_dir + test_file.split('/')[-1])
            # do not check index because behavoir has changed
            assert_series_equal(test_data.cardinalities,
                                expected_data.cardinalities,
                                check_index_type=False,
                                check_index=False)

        # test characters - tessellation characters are either ignored or have a higher tolerance
        # since the simplification behaviour has changed.
        to_skip = ['stbCeA', 'stcOri', 'stcSAl']
        new_data = pd.read_parquet('test/processed_data/primary_chars_temp_region.parquet')
        expected_data = pd.read_parquet('test/actual_data/primary_chars_temp_region.parquet')
        for col in new_data.columns:
        
            if col in to_skip:
                continue
            
            if col in tess_chars:
               rtol = .1
            else:
                rtol = .001
            
            s1 = new_data[col]
            s2 = expected_data[col]
        
            assert_series_equal(s1, s2, rtol=rtol)

        # tests morphotope calculations
        new_morph = pd.read_parquet('test/processed_data/data_morphotopes_temp_region_75_0_None_None_False.pq')
        expected_morph = pd.read_parquet('test/actual_data/data_morphotopes_temp_region_75_0_None_None_False.pq')
        
        new_morph_labels = pd.read_parquet('test/processed_data/tessellation_labels_morphotopes_temp_region_75_0_None_None_False.pq')
        expected_morph_labels = pd.read_parquet('test/actual_data/tessellation_labels_morphotopes_temp_region_75_0_None_None_False.pq')
          
        # test morphotope labels
        assert_series_equal(new_morph_labels.morphotope_label, expected_morph_labels.morphotope_label)
        
        # test morphotope agg data
        for col in new_morph.columns:
            s1 = new_morph[col]
            s2 = expected_morph[col]
            assert_series_equal(s1, s2)
  