import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from window import Ui_MainWindow
import model
import view


class Main(Ui_MainWindow):
    def __init__(self, dialog):
        super(Main, self).__init__()
        # Ui_MainWindow.__init__(self)
        self.setupUi(dialog)
        # @TODO Change the icon of the window to something zebrabow-y!

        # get operating system
        self.OS = sys.platform

        # instance data class and view class
        self.data = model.SessionData()

        # initialize fields and defaults
        self.clusterSampleSize.setText("5000")

        cluster_data_list = ['ternary', 'rgb']
        self.clusterOnData.insertItems(0, cluster_data_list)
        self.clusterOnData.setCurrentIndex(0)

        cluster_method_list = ['fast peak', 'kmeans', 'blah blah']
        self.clusterMethod.insertItems(0, cluster_method_list)
        self.clusterMethod.setCurrentIndex(0)

        color_list = ['custom', 'default', 'cluster color', 'linear']
        self.scatterColorOption.insertItems(0, color_list)
        self.scatterColorOption.setCurrentIndex(0)

        self.ternColorOption.insertItems(0, color_list)
        self.ternColorOption.setCurrentIndex(0)

        scale_list = ['custom', 'default', 'linear']
        self.scatterScaleOption.insertItems(0, scale_list)
        self.scatterScaleOption.setCurrentIndex(1)

        self.ternScaleOption.insertItems(0, scale_list)
        self.ternScaleOption.setCurrentIndex(0)

        # initialize table views
        parameter_column_header = QtGui.QStandardItemModel()
        parameter_column_header.setHorizontalHeaderLabels(['parameter', 'matching variable'])
        self.parameterTable.setModel(parameter_column_header)

        cluster_info_column_header = QtGui.QStandardItemModel()
        cluster_info_column_header.setHorizontalHeaderLabels(['color', 'id', '# of cells', '% total', 'mean sil'])
        self.clusterInformationTable.setModel(cluster_info_column_header)
        # @TODO stretch the header sections so that they fill up the whole space when resizing the gui

        # connect menu items
        self.actionLoad.triggered.connect(self.load_data)
        self.actionClear.triggered.connect(self.clear_data)
        self.actionSave.triggered.connect(self.save_data)
        self.actionRestore.triggered.connect(self.restore_data)

        # connect buttons
        self.removeOutliers.clicked.connect(self.remove_outliers)
        self.clusterPushButton.clicked.connect(self.cluster_data)
        self.addCenterPushButton.clicked.connect(self.add_center)
        self.removeCenterPushButton.clicked.connect(self.remove_center)
        self.highlightClusterPushButton.clicked.connect(self.highlight_cluster)
        self.drawGatePushButton.clicked.connect(self.draw_gate)


    def load_data(self):
        # load fcs file
        self.data.fcs_read()

        # transform data
        self.data.transform_data()

        # print successful load and display number of cells
        self.fileLabel.setText(self.data.sample_name + '\n'+'sample size')
        # @TODO update 'sample size' to the actual sample size that we get after loading the data





    def clear_data(self):
        foo

    def save_data(self):
        foo

    def restore_data(self):
         foo

    def remove_outliers(self):
        foo

    def cluster_data(self):
        foo

    def add_center(self):
        foo

    def remove_center(self):
        foo

    def highlight_cluster(self):
        foo

    def draw_gate(self):
        foo


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dialog = QtWidgets.QMainWindow()
    prog = Main(dialog)
    dialog.show()
    sys.exit(app.exec_())
