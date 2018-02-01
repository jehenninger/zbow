import subprocess
import pickle
import pandas as pd
from PyQt5 import QtWidgets
import os
import numpy as np



class SessionData:
    def __init__(self):
        # variables
        self.file_name = str
        self.sample_name = str
        self.path_name = str
        self.params = tuple
        self.raw = pd.DataFrame
        self.data_size = int
        self.default_transformed = pd.DataFrame
        self.custom_transformed = pd.DataFrame
        self.param_combo_box_list = list

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

        self.file_name, _ = QtWidgets.QFileDialog.getOpenFileName(caption='Select flow cytometry file',
                                                                  directory='/Users/jon/Desktop',
                                                                  filter='FCS file ( *.fcs);; Text file (*.tsv)')
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

        default_params = [self.params[i] for i in default_param_idx]
        custom_params = [self.params[j] for j in custom_param_idx]
        self.default_transformed = logicle.default_transform_data(self.raw, default_params)
        self.custom_transformed = logicle.custom_transform_data(self.raw, custom_params)
        print('Transform ended successfully \n')  # @DEBUG

    def parse_params(self):
        # this function will get the indices of the proper params from the GUI for transformation and
        # store them in a list
        # @TODO should this be a dict with param names? I'm assuming that params will always be in the same order here

        # loop through combo boxes of first 12 parameters (we don't need time or events) and get the index of the param
        idx = []
        for c in self.param_combo_box_list:
            idx.append(c.currentIndex())

        print('These are the current parameter indices.. \n')
        print(idx)  # @DEBUG making sure that we get the proper indices
        return idx

    def init_zbow_3D_plot(self):
        import vispy_scatter_3D

        rgb_data = self.default_transformed.iloc[:, [6, 7, 8]].as_matrix()
        print('size of RGB data is %d by %d \n' % rgb_data.shape)
        print(rgb_data)
        vispy_scatter_3D.scatter(rgb_data)


