import subprocess
import pickle
import pandas as pd
from PyQt5 import QtWidgets
import os
import numpy as np


# import matplotlib as mpl
# mpl.use('Agg')
# from matplotlib import pyplot as plt


class SessionData:
    def __init__(self):
        # variables
        self.OS = str
        self.screen_size = []
        self.file_name = str
        self.save_folder = str
        self.sample_name = str
        self.path_name = str
        self.params = tuple
        self.raw = pd.DataFrame
        self.raw_filtered = pd.DataFrame
        self.data_size = int
        self.noise_cluster_idx = int
        self.default_transformed = pd.DataFrame
        self.linear_transformed = pd.DataFrame
        self.default_ternary = pd.DataFrame
        self.custom_transformed = pd.DataFrame
        self.custom_ternary = pd.DataFrame
        self.linear_ternary = pd.DataFrame
        self.param_combo_box_list = list
        self.cluster_data_idx = [] # this will always be the most up-to-date clustering solution
        self.auto_cluster_idx = [] # this will always be the original auto clustering solution
        self.tab_cluster_data = pd.DataFrame
        self.cluster_eval = []
        self.outlier_scores = []
        self.outliers_removed = bool
        self.use_previous_clustering_solution = bool

        self.h_canvas_3d = None
        self.h_view_3d = None
        self.view_3d_options = {}
        self.h_canvas_2d = None
        self.h_view_2d = None
        self.h_scatter_2d = None
        self.h_scatter_3d = None

    # methods

    def fcs_read(self, sample_size, reload=False):

        import fcsparser

        if reload:
            pass
        else:
            self.file_name, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select flow cytometry file',
                                                                      directory='/Users/jon/Desktop',
                                                                      filter='FCS file ( *.fcs);; Text file (*.csv)')

        #self.file_name ='/Users/jon/Desktop/WKM_fish_023_023_cfp+yfp+ or rfp+_myeloid.fcs'  # @DEBUG temporary to speed up debugging

        self.path_name = os.path.dirname(os.path.abspath(self.file_name))
        _, file_extension = os.path.splitext(self.file_name)

        if file_extension == '.fcs':
            # read in the data
            meta, self.raw = fcsparser.parse(self.file_name, meta_data_only=False, reformat_meta=True)
            self.params = meta['_channel_names_']
            self.data_size = self.raw.__len__()

            self.sample_name = os.path.splitext(os.path.basename(self.file_name))[0]

        elif file_extension == '.csv':
            data = pd.read_csv(self.file_name, index_col=False, header=0)

            self.raw = data.drop('clusterID', axis=1)
            self.cluster_data_idx = data['clusterID']

            metadata_file_name = self.file_name.replace('cluster_solution', 'metadata')

            metadata = pd.read_csv(metadata_file_name, index_col=0, header=None)
            self.noise_cluster_idx = int(metadata.loc['noise_cluster_idx'][1])

            self.data_size = self.raw.__len__()
            self.params = self.raw.columns.values.tolist()

            self.sample_name = os.path.splitext(os.path.basename(self.file_name))[0]
            self.sample_name = self.sample_name.replace('_cluster_solution', '')

            self.use_previous_clustering_solution = True

        else:
            print('Cannot load file. Please make sure it is .fcs or .csv') #TODO change this to dialog warning

        # take random sample if user wants a smaller sample size

        if sample_size < self.data_size:
            self.raw = self.raw.sample(sample_size, replace=False)

    def transform_data(self, outliers_removed=False):
        import logicle
        # initialize outputs

        param_idx = self.parse_params()
        default_param_idx = param_idx[0:12]  # only want first 12 indices for default transform
        custom_param_idx = param_idx[6:9]  # only want RFP, YFP, and CFP for custom transform
        linear_param_idx = param_idx[0:12]  # only want first 12 indices for linear

        default_params = [self.params[i] for i in default_param_idx]
        custom_params = [self.params[j] for j in custom_param_idx]
        linear_params = [self.params[k] for k in linear_param_idx]

        if outliers_removed:
            data = self.raw_filtered
        else:
            data = self.raw

        self.default_transformed = logicle.default_transform_data(data, default_params)
        self.default_transformed = self.normalize_transform(self.default_transformed)

        self.custom_transformed = logicle.custom_transform_data(data, custom_params)
        self.custom_transformed = self.normalize_transform(self.custom_transformed)

        self.linear_transformed = data[linear_params]
        self.linear_transformed = self.normalize_transform(self.linear_transformed)

        # rename columns for future functions:
        current_column_names_default = list(self.default_transformed)
        current_column_names_default[6] = 'RFP'
        current_column_names_default[7] = 'YFP'
        current_column_names_default[8] = 'CFP'
        self.default_transformed.columns = current_column_names_default
        self.default_ternary = self.ternary_transform(self.default_transformed[['RFP', 'YFP', 'CFP']])

        current_column_names_custom = list(self.custom_transformed)
        current_column_names_custom[0] = 'RFP'
        current_column_names_custom[1] = 'YFP'
        current_column_names_custom[2] = 'CFP'
        self.custom_transformed.columns = current_column_names_custom
        self.custom_ternary = self.ternary_transform(self.custom_transformed)

        current_column_names_linear = list(self.linear_transformed)
        current_column_names_linear[6] = 'RFP'
        current_column_names_linear[7] = 'YFP'
        current_column_names_linear[8] = 'CFP'
        self.linear_transformed.columns = current_column_names_linear
        self.linear_ternary = self.ternary_transform(self.linear_transformed[['RFP', 'YFP', 'CFP']])
        print('Transform ended successfully \n')  # @DEBUG

    def parse_params(self):
        # this function will get the indices of the proper params from the GUI for transformation and
        # store them in a list
        # @TODO should this be a dict with param names? I'm assuming that params will always be in the same order here

        # loop through combo boxes of first 12 parameters (we don't need time or events) and get the index of the param
        idx = []
        for c in self.param_combo_box_list:
            idx.append(c.currentIndex())

        return idx

    def normalize_transform(self, data):
        norm = (data - data.min()) / (data.max() - data.min())
        return norm

    def ternary_transform(self, rgb_data):
        import math

        # old way before I found library. Still bugs with the old method
        total = rgb_data.sum(axis='columns')
        rgb_data = rgb_data.divide(total, axis=0)

        y = rgb_data['YFP'].multiply(math.sin(math.pi / 3), axis=0)
        x = rgb_data['RFP'].add(y.multiply(1 / math.tan(math.pi / 3)))

        tern_coords = pd.concat([x, y], axis='columns')
        tern_coords.columns = ['x', 'y']

        # # new way with library
        # rgb_data = [tuple(x) for x in rgb_data.values]
        # x, y = helpers.project_sequence(rgb_data, permutation=None)
        #
        # tern_coords = pd.DataFrame(data=[x, y])
        # tern_coords = tern_coords.transpose()
        # tern_coords.columns = ['x', 'y']

        return tern_coords

    def get_outliers(self):
        threshold = pd.Series(self.outlier_scores).quantile(0.99)
        outliers = self.outlier_scores > threshold
        return outliers

    def get_data_to_cluster_on(self, cluster_on_data):
        if cluster_on_data == 0:
            data = self.custom_ternary
        elif cluster_on_data == 1:
            data = self.custom_transformed[['RFP', 'YFP', 'CFP']]
        elif cluster_on_data == 2:
            data = self.default_ternary
        elif cluster_on_data == 3:
            data = self.default_transformed[['RFP', 'YFP', 'CFP']]
        elif cluster_on_data == 4:
            data = self.linear_ternary
        elif cluster_on_data == 5:
            data = self.linear_transformed[['RFP', 'YFP', 'CFP']]
        else:
            print('Could not retrieve data')  # TODO make dialog error message here

        return(data)

    def make_tabulated_cluster_data(self):
        # cluster_num = max(self.cluster_data_idx) + 1
        cluster_num = len(set(self.cluster_data_idx))
        old_cluster_id = set(self.cluster_data_idx)

        old_cluster_data = self.cluster_data_idx

        mean_cluster_color = pd.DataFrame(data=None, columns=['R', 'G', 'B'])
        mean_sil = []
        cluster_id = []
        for j, val in enumerate(old_cluster_id):
            mean_cluster_color.loc[j, 'R'] = np.mean(self.custom_transformed.loc[self.cluster_data_idx == val, 'RFP'])
            mean_cluster_color.loc[j, 'G'] = np.mean(self.custom_transformed.loc[self.cluster_data_idx == val, 'YFP'])
            mean_cluster_color.loc[j, 'B'] = np.mean(self.custom_transformed.loc[self.cluster_data_idx == val, 'CFP'])

            sil_idx = self.cluster_data_idx == val
            sil_idx = pd.Series(sil_idx, name='bools')
            sil_data = self.cluster_eval[sil_idx.values]
            mean_sil_data = np.mean(sil_data)
            mean_sil_data = round(mean_sil_data, 3)

            mean_sil.append(mean_sil_data)
            cluster_id.append(int(j))

            self.cluster_data_idx[self.cluster_data_idx == val] = j  # resets the cluster ID to be consecutive

        cluster_data = pd.Series(self.cluster_data_idx)
        cluster_data.reset_index(drop=True, inplace=True)
        cluster_data_counts = cluster_data.value_counts(normalize=False, sort=False)
        cluster_data_freq = cluster_data.value_counts(normalize=True, sort=False)
        cluster_data_freq = round(100 * cluster_data_freq, 1)

        # cluster_data_counts.index
        tab_cluster_data = {'id': cluster_id,
                            'mean R': mean_cluster_color['R'],
                            'mean G': mean_cluster_color['G'],
                            'mean B': mean_cluster_color['B'],
                            'num of cells': cluster_data_counts,
                            'percentage': cluster_data_freq,
                            'mean sil': mean_sil}

        self.tab_cluster_data = pd.DataFrame(data=tab_cluster_data)
        self.tab_cluster_data.loc[self.noise_cluster_idx, 'id'] = 'noise'  # if we change where the noise cluster is, must change this
        self.tab_cluster_data = self.tab_cluster_data.sort_values(by="percentage", ascending=False)
        self.tab_cluster_data = self.tab_cluster_data[['id', 'num of cells', 'percentage', 'mean sil', 'mean R', 'mean G', 'mean B']]

    def evaluate_cluster_solution(self, data):
        from sklearn import metrics
        # evaluate clustering solution
        self.cluster_eval = metrics.silhouette_samples(data.as_matrix(), self.cluster_data_idx)

    def auto_cluster(self, cluster_on_data, min_cluster_size=25, min_samples=1, evaluate_cluster=False, prev_clustering_solution=False):
        # from sklearn.cluster import DBSCAN
        import hdbscan

        if prev_clustering_solution:
            self.auto_cluster_idx = self.cluster_data_idx
        else:
            data = self.get_data_to_cluster_on(cluster_on_data)
            # auto_cluster_data = DBSCAN(eps=eps, n_jobs=-1).fit_predict(data.as_matrix())
            auto_cluster_method = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(data.as_matrix())
            auto_cluster_data = auto_cluster_method.labels_
            self.outlier_scores = auto_cluster_method.outlier_scores_

            max_cluster = max(auto_cluster_data)

            # DBSCAN returns -1 for 'halo' cells that don't belong in clusters. We will make them their own cluster for
            #  the time being. Remember that we also change the 'id' to 'noise' in the cluster
            #  table using this index as well

            for k in range(0, len(auto_cluster_data)):
                if auto_cluster_data[k] == -1:
                    auto_cluster_data[k] = max_cluster + 1

            self.noise_cluster_idx = max_cluster + 1
            self.auto_cluster_idx = auto_cluster_data
            self.cluster_data_idx = auto_cluster_data

        if evaluate_cluster:
            self.evaluate_cluster_solution(data)
        else:
            self.cluster_eval = [-2] * len(self.cluster_data_idx)  # make default -2 to identify later
            self.cluster_eval = pd.Series(self.cluster_eval)

        self.make_tabulated_cluster_data()

    def split_cluster_in_two(self, cluster_to_split, cluster_on_data, evaluate_cluster=False):
        from sklearn import cluster

        # cluster_to_split = self.tab_cluster_data[self.tab_cluster_data['id'] == cluster_to_split].index[0]

        cluster_data_idx = self.cluster_data_idx
        data = self.get_data_to_cluster_on(cluster_on_data)
        split_idx = cluster_data_idx == cluster_to_split
        split_data = data[split_idx]

        kmeans_idx = cluster.KMeans(n_clusters=2).fit_predict(split_data)

        # loop through kmeans_idx to replace 0 with -1 and 1 with the cluster_to_split. Then, we will re-write
        #  all cluster numbers to add 1

        for i in range(0, len(kmeans_idx)):
            if kmeans_idx[i] == 0:
                kmeans_idx[i] = -1
            elif kmeans_idx[i] == 1:
                kmeans_idx[i] = cluster_to_split

        cluster_data_idx[split_idx] = kmeans_idx
        cluster_data_idx = cluster_data_idx + 1
        self.cluster_data_idx = cluster_data_idx

        self.noise_cluster_idx = self.noise_cluster_idx + 1  # because we added a cluster

        if evaluate_cluster:
            self.evaluate_cluster_solution(data)
        else:
            self.cluster_eval = [-2] * len(self.cluster_data_idx)  # make default -2 to identify later
            self.cluster_eval = pd.Series(self.cluster_eval)

        self.make_tabulated_cluster_data()

    def join_clusters_together(self, clusters_to_join, cluster_on_data, evaluate_cluster=False):
        # num_of_clusters = max(cluster_data_idx) + 1
        data = self.get_data_to_cluster_on(cluster_on_data)

        # num_of_clusters_to_join = len(clusters_to_join)

        join_idx = [x in clusters_to_join for x in self.cluster_data_idx]

        # make all joined cluster idxs to be the smallest cluster
        self.cluster_data_idx[join_idx] = min(clusters_to_join)

        # need to figure out how many clusters we took out below the noise cluster
        below_noise_count = sum(clusters_to_join < self.noise_cluster_idx)
        self.noise_cluster_idx = self.noise_cluster_idx - (below_noise_count-1)  # because we reduce 2 clusters to 1

        if evaluate_cluster:
            self.evaluate_cluster_solution(data)
        else:
            self.cluster_eval = [-2] * len(self.cluster_data_idx)  # make default -2 to identify later
            self.cluster_eval = pd.Series(self.cluster_eval)

        self.make_tabulated_cluster_data()

    def zbow_3d_plot(self, parent, scale, color, update=False, highlight_cells=None, highlight_color=False):
        from vispy import app, visuals, scene
        from vispy.color import Color, ColorArray
        import helper
        # @TODO need to parse user options here for data

        new_window_position = parent.pos()

        if update:
            options = self.h_view_3d.camera.get_state()

        # get scale data: scale_list = ['custom', 'default', 'linear']
        if scale == 0:
            scale_data = self.custom_transformed.as_matrix()
        elif scale == 1:
            scale_data = self.default_transformed[['RFP', 'YFP', 'CFP']].as_matrix()
        elif scale == 2:
            scale_data = self.linear_transformed[['RFP', 'YFP', 'CFP']].as_matrix()

        # get color data:color_list = ['custom', 'default', 'cluster color', 'linear']
        if color == 0:
            color_data = self.custom_transformed.as_matrix()
        elif color == 1:
            color_data = self.default_transformed[['RFP', 'YFP', 'CFP']].as_matrix()
        elif color == 2:
            if self.tab_cluster_data.empty:
                color_data = helper.distinguishable_colors(1)
            else:
                pseudo_color = helper.distinguishable_colors(self.tab_cluster_data['id'].count())
                pseudo_color[self.noise_cluster_idx] = "#646464"
                color_data = [None] * scale_data.shape[0]
                for i in range(0, scale_data.shape[0]):
                    color_data[i] = pseudo_color[self.cluster_data_idx[i]]
        elif color == 3:
            color_data = self.linear_transformed[['RFP', 'YFP', 'CFP']].as_matrix()
        elif color == 4:
            color_data = np.empty([self.custom_transformed.shape[0], self.custom_transformed.shape[1]])
            color_data[:] = 0.3  # grey for non-highlighted cells
            highlight_cells = pd.Series(highlight_cells, name='bools')
            if highlight_color:
                color_data[highlight_cells, :] = [0.9, 0.9, 0.9]
            else:
                color_data[highlight_cells, :] = self.custom_transformed[['RFP', 'YFP', 'CFP']][highlight_cells.values].as_matrix()

        if not update:
            # build your visuals
            scatter = scene.visuals.create_visual_node(visuals.MarkersVisual)

            # build initial canvas if one doesn't exist
            self.h_canvas_3d = scene.SceneCanvas(title='zbow 3D scatter plot',
                                                 keys='interactive',
                                                 show=True,
                                                 bgcolor=Color([0, 0, 0, 1]),
                                                 )
            parent.setCentralWidget(self.h_canvas_3d.native)

            # Add a ViewBox to let the user zoom/rotate
            default_options = {'fov': 5, 'distance': 25, 'elevation': 30, 'azimuth': 130, 'scale_factor': 3.0}

            self.h_view_3d = self.h_canvas_3d.central_widget.add_view()
            self.h_view_3d.camera = 'turntable'
            self.h_view_3d.camera.fov = default_options['fov']
            self.h_view_3d.camera.distance = default_options['distance']
            self.h_view_3d.camera.elevation = default_options['elevation']
            self.h_view_3d.camera.azimuth = default_options['azimuth']
            self.h_view_3d.camera.scale_factor = default_options['scale_factor']


        # if update:
        #     self.h_view_3d = self.h_canvas_3d.central_widget.add_view()
        #     self.h_view_3d.camera = 'turntable'
        #     self.h_view_3d.camera.set_state(options)
        #
        # else:
        #     self.h_view_3d = self.h_canvas_3d.central_widget.add_view()
        #     self.h_view_3d.camera = 'turntable'
        #     self.h_view_3d.camera.fov = default_options['fov']
        #     self.h_view_3d.camera.distance = default_options['distance']
        #     self.h_view_3d.camera.elevation = default_options['elevation']
        #     self.h_view_3d.camera.azimuth = default_options['azimuth']
        #     self.h_view_3d.camera.scale_factor = default_options['scale_factor']

            # plot 3D RGB axis
            scene.visuals.XYZAxis(parent=self.h_view_3d.scene)

            # this isn't supported, apparently
            # x_axis = scene.Axis(parent=self.h_view_3d.scene, pos=[[0, 0, 0], [1, 0, 0]],
            #                          font_size=12, axis_color='k', tick_color='k', text_color='r',
            #                          axis_width=3)
            #
            # y_axis = scene.Axis(parent=self.h_view_3d.scene, pos=[[0, 0, 0], [0, 1, 0]],
            #                         font_size=12, axis_color='k', tick_color='k', text_color='g',
            #                         axis_width=3)
            #
            # left_axis = scene.Axis(parent=self.h_view_3d.scene, pos=[[0, 0, 0], [0, 0, 1]], tick_direction=(1, -1),
            #                        font_size=12, axis_color='k', tick_color='k', text_color='b',
            #                        axis_width=3)

            self.h_scatter_3d = scatter(parent=self.h_view_3d.scene)
            self.h_scatter_3d.set_gl_state('translucent')
            # h_scatter.set_gl_state(blend=False, depth_test=True)

        cell_color = ColorArray(color=color_data, alpha=1)
        # @BUG I want to use a different alpha here, but Vispy has a bug where you can see through the main canvas with alpha

        self.h_scatter_3d.set_data(pos=scale_data,
                                   symbol='o',
                                   size=5,
                                   edge_width=0,
                                   face_color=cell_color)

        # h_scatter.symbol = visuals.marker_types[10]

        if not update:
            parent.move(new_window_position.x(), new_window_position.y())
            parent.show()

    def zbow_2d_plot(self, parent, scale, color, update=False, highlight_cells=None, highlight_color=False):
        from vispy import app, visuals, scene
        from vispy.color import Color, ColorArray
        import helper
        import scipy.stats as st
        # @TODO make axes and center the plot better
        # @TODO Add matplotlib stacked bar graph for cluster %

        new_window_position = parent.pos()

        if update:
            options = self.h_view_2d.camera.get_state()

        # get scale data: scale_list = ['custom', 'default', 'linear']
        if scale == 0:
            scale_data = self.custom_ternary.as_matrix()
        elif scale == 1:
            scale_data = self.default_ternary.as_matrix()
        elif scale == 2:
            scale_data = self.linear_ternary.as_matrix()

        # get color data:color_list = ['custom', 'default', 'cluster color', 'linear']
        if color == 0:
            color_data = self.custom_transformed.as_matrix()
        elif color == 1:
            color_data = self.default_transformed[['RFP', 'YFP', 'CFP']].as_matrix()
        elif color == 2:
            if self.tab_cluster_data.empty:
                color_data = helper.distinguishable_colors(1)
            else:
                pseudo_color = helper.distinguishable_colors(self.tab_cluster_data['id'].count())
                pseudo_color[self.noise_cluster_idx] = "#646464"

                color_data = [None] * scale_data.shape[0]
                for i in range(0, scale_data.shape[0]):
                    color_data[i] = pseudo_color[self.cluster_data_idx[i]]
        elif color == 3:
            color_data = self.linear_transformed[['RFP', 'YFP', 'CFP']].as_matrix()
        elif color == 4:
            color_data = np.empty([self.custom_transformed.shape[0], self.custom_transformed.shape[1]])
            color_data[:] = 0.8  # grey for non-highlighted cells
            highlight_cells = pd.Series(highlight_cells, name='bools')
            if highlight_color:
                color_data[highlight_cells, :] = [0.2, 0.2, 0.2]
            else:
                color_data[highlight_cells, :] = self.custom_transformed[['RFP', 'YFP', 'CFP']][
                    highlight_cells.values].as_matrix()

        if not update:
            # build your visuals
            scatter = scene.visuals.create_visual_node(visuals.MarkersVisual)



            # build canvas
            self.h_canvas_2d = scene.SceneCanvas(title='zbow 2D ternary plot',
                                                 keys='interactive',
                                                 show=True,
                                                 bgcolor=Color([1, 1, 1, 1]),
                                                 )

            parent.setCentralWidget(self.h_canvas_2d.native)

        # Add a ViewBox to let the user zoom/rotate
        if not update:
            self.h_view_2d = self.h_canvas_2d.central_widget.add_view()
            self.h_view_2d.camera = 'panzoom'
            self.h_view_2d.camera.set_range(x=(-0.2, 1.2), y=(-0.2, 1.2))
        # if update:
        #     self.h_view_2d = self.h_canvas_2d.central_widget.add_view()
        #     self.h_view_2d.camera = 'panzoom'
        #     self.h_view_2d.camera.set_state(options)
        # else:
        #     self.h_view_2d = self.h_canvas_2d.central_widget.add_view()
        #     self.h_view_2d.camera = 'panzoom'

            self.h_scatter_2d = scatter(parent=self.h_view_2d.scene)
        # p1.set_gl_state('translucent', blend=True, depth_test=True)
            self.h_scatter_2d.set_gl_state('translucent', blend=True, depth_test=False)

            # plot 2D ternary axis

            bottom_axis = scene.Axis(parent=self.h_view_2d.scene, pos=[[0, 0], [1, 0]], tick_direction=(0, 1),
                                     font_size=12, axis_color='k', tick_color='k', tick_font_size=0,
                                     axis_width=3, axis_label='red', axis_label_margin=20, axis_font_size=18)

            right_axis = scene.Axis(parent=self.h_view_2d.scene, pos=[[1, 0], [0.5, 1]], tick_direction=(-1, -1),
                                    font_size=12, axis_color='k', tick_color='k', tick_font_size=0,
                                    axis_width=3, axis_label='green', axis_label_margin=20, axis_font_size=18)

            left_axis = scene.Axis(parent=self.h_view_2d.scene, pos=[[0, 0], [0.5, 1]], tick_direction=(1, -1),
                                   font_size=12, axis_color='k', tick_color='k', tick_font_size=0,
                                   axis_width=3, axis_label='blue', axis_label_margin=20, axis_font_size=18)

        cell_color = ColorArray(color=color_data, alpha=1)
        # @BUG I want to use a different alpha here, but Vispy has a bug where you can see through the main canvas with alpha

        self.h_scatter_2d.set_data(pos=scale_data,
                                   symbol='o',
                                   size=5,
                                   edge_width=0,
                                   face_color=cell_color)

        # make contour plot (this doesn't work currently. May just do it in export with matplotlib functionality).
        # kernel = st.gaussian_kde(scale_data.transpose())
        #
        #
        # xmin, xmax = 0, 1
        # ymin, ymax = 0, 1
        #
        # xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        # positions = np.vstack([xx.ravel(), yy.ravel()])
        #
        #
        # f = np.reshape(kernel(positions).T, xx.shape)
        #
        # mesh = scene.visuals.GridMesh(xx, yy, f)
        #self.h_view_2d.add(mesh)


        # if not update:
        #     self.h_scatter_2d.symbol = visuals.marker_types[10]

        if not update:
            parent.move(new_window_position.x(), new_window_position.y())
            parent.show()

    def make_output_plots(self, scale, color):
        import ternary
        from matplotlib import pyplot as plt
        import helper
        from scipy import stats as sd

        # get scale data: scale_list = ['custom', 'default', 'linear']
        if scale == 0:
            scale_data = self.custom_transformed.as_matrix()
        elif scale == 1:
            scale_data = self.default_transformed.as_matrix()
        elif scale == 2:
            scale_data = self.linear_transformed.as_matrix()

        if scale == 0:
            contour_data = self.custom_ternary.as_matrix()
        elif scale == 1:
            contour_data = self.default_ternary.as_matrix()
        elif scale == 2:
            contour_data = self.linear_ternary.as_matrix()

        # get color data:color_list = ['custom', 'default', 'cluster color', 'linear']
        if color == 0:
            color_data = self.custom_transformed.as_matrix()
        elif color == 1:
            color_data = self.default_transformed[['RFP', 'YFP', 'CFP']].as_matrix()
        elif color == 2:
            if self.tab_cluster_data.empty:
                color_data = helper.distinguishable_colors(1)
            else:
                pseudo_color = helper.distinguishable_colors(self.tab_cluster_data['id'].count())
                pseudo_color[self.noise_cluster_idx] = "#646464"

                color_data = [None] * scale_data.shape[0]
                for i in range(0, scale_data.shape[0]):
                    color_data[i] = pseudo_color[self.cluster_data_idx[i]]
        elif color == 3:
            color_data = self.linear_transformed[['RFP', 'YFP', 'CFP']].as_matrix()

        # this assures that R + G + B = scale, which is required for the ternary library
        total = scale_data.sum(axis=1)
        scale_data = scale_data/total[:, None]

        # get Gaussian kernel for contour plot
        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1

        X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([contour_data[:, 0], contour_data[:, 1]])
        kernel = sd.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)

        # new way with library
        scale = 1
        figure, tern_plot = ternary.figure(scale=scale)
        figure.set_size_inches(5.37, 5)
        figure.set_dpi(300)
        # tern_plot.set_title("ternary plot", fontsize=18)
        tern_plot.boundary(linewidth=1.0)
        tern_plot.gridlines(multiple=0.1, color='grey')

        tern_plot.scatter(scale_data, marker='o', color=color_data, s=2)

        tern_plot.clear_matplotlib_ticks()

        plt.contour(X, Y, Z, colors='k', alpha=0.6, linewidths=1)

        ternary_fname = os.path.join(self.save_folder, 'ternary_plots', self.sample_name)
        plt.savefig(ternary_fname + '.png', dpi=300, transparent=True, pad_inches=0, Bbox='tight')
        plt.savefig(ternary_fname + '.eps', dpi=300, transparent=True, pad_inches=0, Bbox='tight')

