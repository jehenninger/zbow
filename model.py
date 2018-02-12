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
        self.default_transformed = pd.DataFrame
        self.linear_transformed = pd.DataFrame
        self.default_ternary = pd.DataFrame
        self.custom_transformed = pd.DataFrame
        self.custom_ternary = pd.DataFrame
        self.linear_ternary = pd.DataFrame
        self.param_combo_box_list = list

        self.cluster_data = pd.DataFrame
        self.auto_cluster_data = []

        self.h_canvas_3d = None
        self.h_view_3d = None
        self.h_canvas_2d = None
        self.h_view_2d = None

    # methods

    def fcs_read(self):
        # Old method to call command line
        # command = '/Users/jon/PycharmProjects/zbow/fcs_read.py %s' % self.file_name
        #
        # process = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        # return_code = process.returncode
        # output = process.stdout
        # print('Python2 finished with return code %d\n' % return_code)
        # return output

        import fcsparser

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

    def transform_data(self):
        import logicle
        # initialize outputs

        param_idx = self.parse_params()
        default_param_idx = param_idx[0:12]  # only want first 12 indices for default transform
        custom_param_idx = param_idx[6:9]  # only want RFP, YFP, and CFP for custom transform
        linear_param_idx = param_idx[0:12] # only want first 12 indices for linear

        default_params = [self.params[i] for i in default_param_idx]
        custom_params = [self.params[j] for j in custom_param_idx]
        linear_params = [self.params[k] for k in linear_param_idx]

        self.default_transformed = logicle.default_transform_data(self.raw, default_params)
        self.default_transformed = self.normalize_transform(self.default_transformed)

        self.custom_transformed = logicle.custom_transform_data(self.raw, custom_params)
        self.custom_transformed = self.normalize_transform(self.custom_transformed)

        # @START JON - make sure we get linear values in the correct order that we want
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

        y = rgb_data['YFP'].multiply(math.sin(math.pi/3), axis=0)
        x = rgb_data['RFP'].add(y.multiply(1/math.tan(math.pi/3)))

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

    def auto_cluster(self):
        from sklearn.cluster import DBSCAN
        # @START HERE
        self.auto_cluster_data = DBSCAN(eps=0.01, n_jobs=-1).fit_predict(self.custom_ternary.as_matrix())
        print('Number of clusters: ', max(self.auto_cluster_data), '\n')
        print('Cluster assignment: \n', self.auto_cluster_data, '\n')

    def zbow_3d_plot(self, scale, color):
        from vispy import app, visuals, scene
        from vispy.color import Color, ColorArray
        import helper
        # @TODO need to parse user options here for data

        # get scale data: scale_list = ['custom', 'default', 'linear']
        if scale == 0:
            scale_data = self.custom_transformed.as_matrix()
        elif scale == 1:
            scale_data = self.default_transformed.iloc[:, [6, 7, 8]].as_matrix()
        elif scale == 2:
            scale_data = self.linear_transformed.iloc[:, [6, 7, 8]].as_matrix()

        # get color data:color_list = ['custom', 'default', 'cluster color', 'linear']
        if color == 0:
            # color_data = self.custom_transformed.as_matrix()
            pseudo_color = helper.distinguishable_colors(max(self.auto_cluster_data) + 1)
            print('Scale data length is ', scale_data.shape[0], '\n')
            color_data = [None] * scale_data.shape[0]
            for i in range(0, scale_data.shape[0]):
                color_data[i] = pseudo_color[self.auto_cluster_data[i]]

            print('Color is: \n', color_data, '\n')
        elif color == 1:
            color_data = self.default_transformed[['RFP', 'YFP', 'CFP']].as_matrix()
        elif color == 2:
            color_data = self.pseudo_color # @TODO Need to define cluster color
        elif color == 3:
            color_data = self.linear_transformed[['RFP', 'YFP', 'CFP']].as_matrix()

        default_position = (0.5*self.screen_size[0], 0.1*self.screen_size[1])

        # build your visuals
        scatter = scene.visuals.create_visual_node(visuals.MarkersVisual)

        # build initial canvas if one doesn't exist
        if not self.h_canvas_3d:
            self.h_canvas_3d = scene.SceneCanvas(title='zbow 3D scatter plot',
                                                 keys='interactive',
                                                 show=True,
                                                 bgcolor=Color([0, 0, 0, 1]),
                                                 position=default_position)

            # Add a ViewBox to let the user zoom/rotate
            self.h_view_3d = self.h_canvas_3d.central_widget.add_view()
            self.h_view_3d.camera = 'turntable'
            self.h_view_3d.camera.fov = 5
            self.h_view_3d.camera.distance = 25
            self.h_view_3d.camera.elevation = 30
            self.h_view_3d.camera.azimuth = 130

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

    def init_zbow_2d_plot(self):
        import scatter2D

        # old way
        rgb_data = self.custom_ternary.as_matrix()
        color_data = self.custom_transformed.as_matrix()

        position = (0.5*self.screen_size[0], 0.5*self.screen_size[1])

        scatter2D.scatter_2d(rgb_data, color_data, position=position)

        # new way with library
        # data_copy = self.custom_transformed[['RFP', 'CFP', 'YFP']]
        # rgb_data = [tuple(x) for x in data_copy.values]
        # color_data = self.custom_transformed.as_matrix()
        # scatter2D.scatter_2d(rgb_data, color_data)

