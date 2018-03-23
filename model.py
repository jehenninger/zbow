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
        self.sample_name = str
        self.path_name = str
        self.params = tuple
        self.raw = pd.DataFrame
        self.data_size = int
        self.noise_cluster_idx = int
        self.default_transformed = pd.DataFrame
        self.linear_transformed = pd.DataFrame
        self.default_ternary = pd.DataFrame
        self.custom_transformed = pd.DataFrame
        self.custom_ternary = pd.DataFrame
        self.linear_ternary = pd.DataFrame
        self.param_combo_box_list = list
        self.cluster_data_idx = []
        self.auto_cluster_idx = []
        self.tab_cluster_data = pd.Series

        self.h_canvas_3d = None
        self.h_view_3d = None
        self.view_3d_options = {}
        self.h_canvas_2d = None
        self.h_view_2d = None

    # methods

    def fcs_read(self, sample_size):
        # Old method to call command line
        # command = '/Users/jon/PycharmProjects/zbow/fcs_read.py %s' % self.file_name
        #
        # process = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        # return_code = process.returncode
        # output = process.stdout
        # print('Python2 finished with return code %d\n' % return_code)
        # return output

        import fcsparser

        # @TODO Save the pathname and use an if statement so that if the program is kept open, it will re-open the
        # previous path

        # self.file_name, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select flow cytometry file',
        #                                                           directory='/Users/jon/Desktop',
        #                                                           filter='FCS file ( *.fcs);; Text file (*.tsv)')

        self.file_name = '/Users/jon/Desktop/wkm_fish_018_006.myeloid.fcs'  # @DEBUG temporary to speed up debugging
        # @TODO make sure that we output the data in tab-separated format, otherwise change this

        self.path_name = os.path.dirname(os.path.abspath(self.file_name))
        self.sample_name = os.path.basename(self.file_name)

        # read in the data
        meta, self.raw = fcsparser.parse(self.file_name, meta_data_only=False, reformat_meta=True)
        self.params = meta['_channel_names_']
        self.data_size = self.raw.__len__()

        # take random sample if user wants a smaller sample size

        if sample_size < self.data_size:
            self.raw = self.raw.sample(sample_size, replace=False)

    def transform_data(self):
        import logicle
        # initialize outputs

        param_idx = self.parse_params()
        default_param_idx = param_idx[0:12]  # only want first 12 indices for default transform
        custom_param_idx = param_idx[6:9]  # only want RFP, YFP, and CFP for custom transform
        linear_param_idx = param_idx[0:12]  # only want first 12 indices for linear

        default_params = [self.params[i] for i in default_param_idx]
        custom_params = [self.params[j] for j in custom_param_idx]
        linear_params = [self.params[k] for k in linear_param_idx]

        self.default_transformed = logicle.default_transform_data(self.raw, default_params)
        self.default_transformed = self.normalize_transform(self.default_transformed)

        self.custom_transformed = logicle.custom_transform_data(self.raw, custom_params)
        self.custom_transformed = self.normalize_transform(self.custom_transformed)

        self.linear_transformed = self.raw[linear_params]
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
        from ternary import helpers

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

    def auto_cluster(self, cluster_on_data):
        from sklearn.cluster import DBSCAN

        # cluster_data_list = ['custom ternary', 'custom rgb', 'default ternary', 'default rgb', 'linear']
        eps_tern = 0.01
        eps_transformed = 0.05
        if cluster_on_data == 0:
            data = self.custom_ternary
            eps = eps_tern
        elif cluster_on_data == 1:
            data = self.custom_transformed[['RFP', 'YFP', 'CFP']]
            eps = eps_transformed
        elif cluster_on_data == 2:
            data = self.default_ternary
            eps = eps_tern
        elif cluster_on_data == 3:
            data = self.default_transformed[['RFP', 'YFP', 'CFP']]
            eps = eps_transformed
        elif cluster_on_data == 4:
            data = self.linear_ternary
            eps = eps_tern
        elif cluster_on_data == 5:
            data = self.linear_transformed[['RFP', 'YFP', 'CFP']]
            eps = eps_transformed

        auto_cluster_data = DBSCAN(eps=eps, n_jobs=-1).fit_predict(data.as_matrix())
        max_cluster = max(auto_cluster_data)
        # @TODO Change the non-clustered cells to always choose 'gray' as their color
        #DBSCAN returns -1 for 'halo' cells that don't belong in clusters. We will make them their own cluster for the time being.
        # remember that we also change the 'id' to 'noise' in the cluster table using this index as well
        for k in range(0, len(auto_cluster_data)):
            if auto_cluster_data[k] == -1:
                auto_cluster_data[k] = max_cluster + 1

        self.noise_cluster_idx = max_cluster + 1
        self.auto_cluster_idx = auto_cluster_data
        self.cluster_data_idx = auto_cluster_data

        cluster_num = max(auto_cluster_data) + 1
        mean_cluster_color = pd.DataFrame(data=None, columns=['R', 'G', 'B'])
        for j in range(0, cluster_num):
            mean_cluster_color.loc[j, 'R'] = np.mean(self.custom_transformed.loc[self.cluster_data_idx == j, 'RFP'])
            mean_cluster_color.loc[j, 'G'] = np.mean(self.custom_transformed.loc[self.cluster_data_idx == j, 'YFP'])
            mean_cluster_color.loc[j, 'B'] = np.mean(self.custom_transformed.loc[self.cluster_data_idx == j, 'CFP'])

        print('Mean cluster color is: \n')
        print(mean_cluster_color)

        cluster_data = pd.Series(auto_cluster_data)
        cluster_data_counts = cluster_data.value_counts(normalize=False, sort=True, ascending=False)
        cluster_data_freq = cluster_data.value_counts(normalize=True, sort=True, ascending=False)
        cluster_data_freq = 100 * cluster_data_freq

        tab_cluster_data = {'id': cluster_data_counts.index,
                            'mean R': mean_cluster_color['R'],
                            'mean G': mean_cluster_color['G'],
                            'mean B': mean_cluster_color['B'],
                            'num of cells': cluster_data_counts,
                            'percentage': cluster_data_freq}

        self.tab_cluster_data = pd.DataFrame(tab_cluster_data)
        self.tab_cluster_data.loc[self.noise_cluster_idx, 'id'] = 'noise'  #if we change where the noise cluster is, must change this
        self.tab_cluster_data = self.tab_cluster_data.sort_values(by="percentage", ascending=False)

        print('Number of clusters: ', self.tab_cluster_data['id'].count(), '\n')
        print('Tab cluster data: \n', self.tab_cluster_data, '\n')

    def decision_graph(self, cluster_on_data):

        # cluster_data_list = ['custom ternary', 'custom rgb', 'default ternary', 'default rgb', 'linear']

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

        self.rho_delta(data.as_matrix(), 1.0)

    def rho_delta(self, data, percent):
        from scipy import spatial
        import itertools

        # get pairwise distances
        dist = spatial.distance.pdist(data, 'euclidean')
        # dist = spatial.distance.squareform(distxx)

        # ND: number of data points
        nd = data.shape[0]
        nc = data.shape[1]

        n = dist.shape[0]

        position = round(n * percent / 100)
        sda = np.sort(dist)
        dc = sda[position]
        print('Computing Rho with gaussian kernel of radius: %12.6f\n' % (dc))

        # try without squareform
        # @START HERE: Looks like this is slow, look into numpy.vectorize()
        pair_idx = list(itertools.combinations(range(nd), nc))
        print(pair_idx, '\n')

        def gauss_func(single_dist, cutoff):
            output = np.exp(-np.square(single_dist/cutoff))

            return output

        def get_index_of_tuples_with_cell_distance(input_list, value, num_of_observations):
            from itertools import compress
            distance_idx = [None]*input_list.__len__()
            for pos, t in enumerate(input_list):
                if t[0] == value or t[1] == value:  # we always expect 2-tuples because it's pairwise distances
                    distance_idx[pos] = True
                else:
                    distance_idx[pos] = False

            distance_output = list(compress(range(len(distance_idx)), distance_idx))
            return distance_output

        # get indices of where each cell distance begins

        tuple_idx = [None] * nd
        for j in range(0, nd):
            tuple_idx[j] = get_index_of_tuples_with_cell_distance(pair_idx, j, nd)

        # loop over number of data points, calculate gaussian distance, and sum for Rho
        rho = [None]*nd
        for i in range(0, nd):
            temp_data = dist[tuple_idx[i]]
            rho_element = [None]*(nd-1)
            for j, item in enumerate(temp_data):
                rho_element[j] = gauss_func(item, dc)

            rho[i] = sum(rho_element)

            print(rho[i], '\n')




        # N: number of point pairs/distance
        # n = distxx.shape[0]



        # rho = np.zeros(nd)
        # # Gaussian kernel
        # for i in range(nd - 1):
        #     for j in range((i + 1), nd):
        #         rho[i] = rho[i] + np.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))
        #         rho[j] = rho[j] + np.exp(-(dist[i, j] / dc) * (dist[i, j] / dc))
        #
        # print('Rho is:\n ', rho, '\n')
        #
        #
        # maxd = dist.max()
        # ordrho = (-rho).argsort()
        # delta = np.zeros(nd)
        # nneigh = np.zeros(nd)
        # delta[ordrho[0]] = -1
        # nneigh[ordrho[0]] = 0
        #
        # for ii in range(1, nd):
        #     delta[ordrho[ii]] = maxd
        #     for jj in range(ii):
        #         if dist[ordrho[ii], ordrho[jj]] < delta[ordrho[ii]]:
        #             delta[ordrho[ii]] = dist[ordrho[ii], ordrho[jj]]
        #             nneigh[ordrho[ii]] = ordrho[jj]
        #
        # delta[ordrho[0]] = delta.max()
        #
        # print('Delta is:\n ', delta, '\n')
        # print('Nneigh is:\n ', nneigh, '\n')
        #
        # return rho, delta, nneigh

    def zbow_3d_plot(self, parent, scale, color, update=False):
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
                    color_data[i] = pseudo_color[self.auto_cluster_idx[i]]
        elif color == 3:
            color_data = self.linear_transformed[['RFP', 'YFP', 'CFP']].as_matrix()

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
        if update:
            self.h_view_3d = self.h_canvas_3d.central_widget.add_view()
            self.h_view_3d.camera = 'turntable'
            self.h_view_3d.camera.set_state(options)

        else:
            self.h_view_3d = self.h_canvas_3d.central_widget.add_view()
            self.h_view_3d.camera = 'turntable'
            self.h_view_3d.camera.fov = default_options['fov']
            self.h_view_3d.camera.distance = default_options['distance']
            self.h_view_3d.camera.elevation = default_options['elevation']
            self.h_view_3d.camera.azimuth = default_options['azimuth']
            self.h_view_3d.camera.scale_factor = default_options['scale_factor']

        # plot 3D RGB axis
        scene.visuals.XYZAxis(parent=self.h_view_3d.scene)

        h_scatter = scatter(parent=self.h_view_3d.scene)
        h_scatter.set_gl_state('translucent')
        # h_scatter.set_gl_state(blend=False, depth_test=True)

        cell_color = ColorArray(color=color_data, alpha=1)
        # @BUG I want to use a different alpha here, but Vispy has a bug where you can see through the main canvas with alpha

        h_scatter.set_data(pos=scale_data,
                           symbol='o',
                           size=5,
                           edge_width=0,
                           face_color=cell_color)

        h_scatter.symbol = visuals.marker_types[10]

        parent.move(new_window_position.x(), new_window_position.y())
        parent.show()


    def zbow_2d_plot(self, parent, scale, color, update=False):
        from vispy import app, visuals, scene
        from vispy.color import Color, ColorArray
        import helper
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
                    color_data[i] = pseudo_color[self.auto_cluster_idx[i]]
        elif color == 3:
            color_data = self.linear_transformed[['RFP', 'YFP', 'CFP']].as_matrix()

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
        if update:
            self.h_view_2d = self.h_canvas_2d.central_widget.add_view()
            self.h_view_2d.camera = 'panzoom'
            self.h_view_2d.camera.set_state(options)
        else:
            self.h_view_2d = self.h_canvas_2d.central_widget.add_view()
            self.h_view_2d.camera = 'panzoom'

        h_scatter = scatter(parent=self.h_view_2d.scene)
        # p1.set_gl_state('translucent', blend=True, depth_test=True)
        h_scatter.set_gl_state('translucent', blend=True, depth_test=False)

        cell_color = ColorArray(color=color_data, alpha=1)
        # @BUG I want to use a different alpha here, but Vispy has a bug where you can see through the main canvas with alpha

        h_scatter.set_data(pos=scale_data,
                           symbol='o',
                           size=5,
                           edge_width=0,
                           face_color=cell_color)

        h_scatter.symbol = visuals.marker_types[10]

        parent.move(new_window_position.x(), new_window_position.y())
        parent.show()
