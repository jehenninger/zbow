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
        self.default_transformed = logicle.default_transform_data(self.raw, self.params)
        # self.custom_transformed = logicle.custom_transform_data(self.raw, self.params)
        print('Transform ended \n')
        print(self.default_transformed)


