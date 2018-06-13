import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from window import Ui_MainWindow
import model
import view
import pandas as pd


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
        self.clusterSampleSize.setText("10000")

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
        self.clusterInformationTable.setColumnCount(4)
        cluster_info_column_header = ['id', '# of cells', '% total', 'mean sil']
        self.clusterInformationTable.setHorizontalHeaderLabels(cluster_info_column_header)
        self.clusterInformationTable.cellDoubleClicked.connect(self.highlight_cluster)
        self.clusterInformationTable.cellClicked.connect(self.highlight_cluster_in_table)


        # connect menu items
        self.actionLoad.triggered.connect(self.load_data)
        self.actionClear.triggered.connect(self.clear_data)
        self.actionSave.triggered.connect(self.save_data)
        self.actionRestore.triggered.connect(self.restore_data)

        # connect buttons
        self.removeOutliers.clicked.connect(self.remove_outliers)
        self.clusterPushButton.clicked.connect(self.cluster_data)
        self.joinClusterPushButton.clicked.connect(self.join_cluster)
        self.highlightClusterPushButton.clicked.connect(self.highlight_cluster)
        self.updateParams.clicked.connect(self.update_params)
        self.splitCluster.clicked.connect(self.split_cluster)
        self.viewOutliersButton.clicked.connect(self.view_outliers)

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
        self.data.param_combo_box_list = view.init_param_table(self.parameterTable, self.data.params)

        # transform data
        self.data.transform_data()

        # print successful load and display number of cells
        self.fileLabel.setText(self.data.sample_name + '\n' + self.data.data_size.__str__() + ' total cells (' +
                               self.data.raw.shape[0].__str__() + ' sampled cells)')


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

        self.cluster_data()


    def update_params(self):
        print('not done yet')

    def clear_data(self):
        print('not done yet')

    def save_data(self):

        # get directory to save to
        self.data.save_folder = QtWidgets.QFileDialog.getExistingDirectory(caption='Select directory to save output',
                                                                           directory=os.path.dirname(self.data.file_name))

        self.data.tab_cluster_data.to_pickle(path=os.path.join(self.data.save_folder,
                                                               self.data.sample_name + '_Summary.pkl'),
                                             compression=None)
        self.data.tab_cluster_data.to_excel(os.path.join(self.data.save_folder,
                                                         self.data.sample_name + '_Summary.xlsx'),
                                            index=False)

        pd.Series(self.data.cluster_data_idx).to_pickle(path=os.path.join(self.data.save_folder,
                                                                          self.data.sample_name + '_cluster_solution.pkl'),
                                                        compression=None)

        pd.Series(self.data.cluster_data_idx).to_csv(path=os.path.join(self.data.save_folder,
                                                                      self.data.sample_name + '_cluster_solution.csv'),
                                                     index=False, header=False)

    def restore_data(self):
        print('not done yet')

    def view_outliers(self):
        outliers = self.data.get_outliers()

        self.data.zbow_3d_plot(self.scatter3DWindow,
                               scale=self.scatterScaleOption.currentIndex(),
                               color=4,
                               update=True,
                               highlight_cells=outliers,
                               highlight_color=True)

        self.data.zbow_2d_plot(self.tern2DWindow,
                               scale=self.ternScaleOption.currentIndex(),
                               color=4,
                               update=True,
                               highlight_cells=outliers,
                               highlight_color=True)

    def remove_outliers(self):
        outliers = self.data.get_outliers()
        outliers = pd.Series(outliers, name='bools')

        if self.data.raw_filtered.empty: # this will make sure that you can do multiple rounds of removing outliers
            self.data.raw_filtered = self.data.raw

        self.data.raw_filtered = self.data.raw_filtered[~outliers.values] # want to remove outliers, not keep them!

        self.data.transform_data(outliers_removed=True)

        self.fileLabel.setText(self.data.sample_name + '\n' + self.data.data_size.__str__() + ' total cells (' +
                               self.data.raw_filtered.shape[0].__str__() + ' sampled cells)')

        self.cluster_data()

        self.update_plots()

    def cluster_data(self):
        # auto cluster the data
        self.data.auto_cluster(self.clusterOnData.currentIndex())
        view.update_cluster_table(self.clusterInformationTable, self.data.tab_cluster_data)

        # self.data.decision_graph(self.clusterOnData.currentIndex())

        self.update_plots()

    def join_cluster(self):
        table_object = self.clusterInformationTable.selectedItems()
        clusters_to_join = []
        # TODO handle situations where nothing is selected!
        for i in range(0, len(table_object)):
            temp_table_object = table_object[i].text()

            if 'noise' in str(temp_table_object):
                print('Can not join noise cluster')  # TODO change this to a dialog message box
                con = False
                break
            else:
                clusters_to_join.append(temp_table_object)
                clusters_to_join[i] = int(clusters_to_join[i])
                con = True

        if con:
            self.data.join_clusters_together(clusters_to_join, self.clusterOnData.currentIndex())

            view.update_cluster_table(self.clusterInformationTable, self.data.tab_cluster_data)
            self.update_plots()

    def highlight_cluster(self):
        table_object = self.clusterInformationTable.selectedItems()
        clusters_to_highlight = []

        for i in range(0, len(table_object)):
            temp_table_object = table_object[i].text()

            if 'noise' in str(temp_table_object):
                clusters_to_highlight.append(int(self.data.noise_cluster_idx))
            else:
                clusters_to_highlight.append(temp_table_object)
                clusters_to_highlight[i] = int(clusters_to_highlight[i])

        highlight_cells = [x in clusters_to_highlight for x in self.data.cluster_data_idx]

        self.data.zbow_3d_plot(self.scatter3DWindow,
                               scale=self.scatterScaleOption.currentIndex(),
                               color=4,
                               update=True,
                               highlight_cells=highlight_cells)

        self.data.zbow_2d_plot(self.tern2DWindow,
                               scale=self.ternScaleOption.currentIndex(),
                               color=4,
                               update=True,
                               highlight_cells=highlight_cells)

    def highlight_cluster_in_table(self):
        current_cell = self.clusterInformationTable.selectedItems()

        current_cell = current_cell[0]
        cell_color = current_cell.background()
        cell_color = cell_color.color()
        r = cell_color.red()
        g = cell_color.green()
        b = cell_color.blue()

        self.clusterInformationTable.setStyleSheet('selection-background-color: rgba(' +
                                                   str(r) + ', ' + str(g) + ', ' + str(b) + ', 65)')

    def split_cluster(self):
        cluster_to_split = self.clusterInformationTable.selectedItems()
        cluster_to_split = cluster_to_split[0].text()
        print('cluster to split \n', cluster_to_split)

        if 'noise' in str(cluster_to_split):
            print('Can not split noise cluster')  # TODO change this to a dialog message box

        else:
            cluster_to_split = int(cluster_to_split)
            self.data.split_cluster_in_two(cluster_to_split, self.clusterOnData.currentIndex())

            view.update_cluster_table(self.clusterInformationTable, self.data.tab_cluster_data)
            self.update_plots()

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
