import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from window import Ui_MainWindow
import model
import view


class scatterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(scatterWindow, self).__init__()


class Main(Ui_MainWindow):
    def __init__(self, dialog):
        super(Main, self).__init__()
        # Ui_MainWindow.__init__(self)
        self.setupUi(dialog)
        # @TODO Change the icon of the window to something zebrabow-y!

        # get operating system
        self.OS = sys.platform

        # init screen size
        self.screen_size = []

        # instance data class and other windows
        self.data = model.SessionData()
        self.scatter3DWindow = scatterWindow()
        self.tern2DWindow = scatterWindow()

        # initialize fields and defaults
        self.clusterSampleSize.setText("1000")

        cluster_data_list = ['custom ternary', 'custom rgb', 'default ternary',
                             'default rgb', 'linear ternary', 'linear rgb']
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
        self.parameterTable.setRowCount(1)
        self.parameterTable.setColumnCount(2)
        parameter_column_header = ['variable', 'matching parameter']
        self.parameterTable.setHorizontalHeaderLabels(parameter_column_header)

        self.clusterInformationTable.setRowCount(1)
        self.clusterInformationTable.setColumnCount(5)
        cluster_info_column_header = ['color', 'id', '# of cells', '% total', 'mean sil']
        self.clusterInformationTable.setHorizontalHeaderLabels(cluster_info_column_header)

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
        self.updateParams.clicked.connect(self.update_params)

        # connect options
        self.scatterColorOption.activated.connect(self.update_plots)
        self.scatterScaleOption.activated.connect(self.update_plots)

        self.ternColorOption.activated.connect(self.update_plots)
        self.ternScaleOption.activated.connect(self.update_plots)

    def load_data(self):
        # @TODO Add loading timer dialog box
        # @TODO Add shortcut for menu items, like loading data
        import pandas as pd
        self.data.screen_size = self.screen_size
        self.data.OS = self.OS

        # reinitialize auto_cluster data
        self.data.tab_cluster_data = pd.Series
        self.data.auto_cluster_idx = []

        # load fcs file
        sample_size = self.clusterSampleSize.text()
        sample_size = int(sample_size)
        self.data.fcs_read(sample_size)

        # fill parameter table
        self.data.param_combo_box_list = view.initParamTable(self.parameterTable, self.data.params)

        # transform data
        self.data.transform_data()

        # print successful load and display number of cells
        self.fileLabel.setText(self.data.sample_name + '\n' + self.data.data_size.__str__() + ' cells')


        # initialize 2D and 3D zbow graph
        default_position = [0.5 * self.screen_size[0], 0.05 * self.screen_size[1]]
        self.scatter3DWindow.move(default_position[0], default_position[1])
        # @TODO set default size of the 2D and 3D graphs so that they don't overlap
        self.data.zbow_3d_plot(self.scatter3DWindow,
                               scale=self.scatterScaleOption.currentIndex(),
                               color=self.scatterColorOption.currentIndex())

        default_position = [0.5 * self.screen_size[0], 0.5 * self.screen_size[1]]
        self.tern2DWindow.move(default_position[0], default_position[1])
        self.data.zbow_2d_plot(self.tern2DWindow,
                               scale=self.ternScaleOption.currentIndex(),
                               color=self.ternColorOption.currentIndex())


    def update_params(self):
        print('not done yet')

    def clear_data(self):
        print('not done yet')

    def save_data(self):
        print('not done yet')

    def restore_data(self):
        print('not done yet')

    def remove_outliers(self):
        print('not done yet')

    def cluster_data(self):
        # auto cluster the data
        self.data.auto_cluster(self.clusterOnData.currentIndex())
        self.data.decision_graph(self.clusterOnData.currentIndex())

        self.update_plots()

    def add_center(self):
        print('not done yet')

    def remove_center(self):
        print('not done yet')

    def highlight_cluster(self):
        print('not done yet')

    def draw_gate(self):
        print('not done yet')

    def update_plots(self):
        self.data.zbow_3d_plot(self.scatter3DWindow,
                               scale=self.scatterScaleOption.currentIndex(),
                               color=self.scatterColorOption.currentIndex(),
                               update=True)

        self.data.zbow_2d_plot(self.tern2DWindow,
                               scale=self.ternScaleOption.currentIndex(),
                               color=self.ternColorOption.currentIndex(),
                               update=True)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen()
    size = screen.size()
    height = size.height()
    width = size.width()

    dialog = QtWidgets.QMainWindow()
    dialog.move(0.05*width, 0.1*height)
    prog = Main(dialog)
    prog.screen_size = [size.width(), size.height()]
    dialog.show()
    sys.exit(app.exec_())
