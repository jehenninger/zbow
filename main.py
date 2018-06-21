import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from window import Ui_MainWindow
import model
import view
import helper


class ScatterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ScatterWindow, self).__init__()


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
        self.scatter3DWindow = ScatterWindow()
        self.tern2DWindow = ScatterWindow()

        # initialize fields and defaults
        self.clusterSampleSize.setText("10000")
        self.clusterMinClusterSize.setText("25")
        self.clusterMinSamples.setText("1")

        self.progressBar.setValue(0)

        self.giniCoeff.setText('NA')
        self.shannonEntropy.setText('NA')

        cluster_data_list = ['custom ternary', 'custom rgb', 'default ternary',
                             'default rgb', 'linear ternary', 'linear rgb']
        self.clusterOnData.insertItems(0, cluster_data_list)
        self.clusterOnData.setCurrentIndex(0)

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
        self.evaluateClusteringCheckBox.clicked.connect(self.evaluate_clusters)

        # connect options
        self.scatterColorOption.activated.connect(self.update_plots)
        self.scatterScaleOption.activated.connect(self.update_plots)

        self.ternColorOption.activated.connect(self.update_plots)
        self.ternScaleOption.activated.connect(self.update_plots)

    def load_data(self):
        # @TODO Add loading timer dialog box
        # @TODO Add shortcut for menu items, like loading data
        import pandas as pd

        view.start_progress_bar(self.progressBar, start=0, stop=4)

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

        view.update_progress_bar(self.progressBar)
        QtWidgets.QApplication.processEvents()
        # transform data
        self.data.transform_data()

        # print successful load and display number of cells
        self.fileLabel.setText(self.data.sample_name + '\n' + self.data.data_size.__str__() + ' total cells (' +
                               self.data.raw.shape[0].__str__() + ' sampled cells)')

        view.update_progress_bar(self.progressBar)
        QtWidgets.QApplication.processEvents()
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

        view.update_progress_bar(self.progressBar)
        QtWidgets.QApplication.processEvents()
        self.cluster_data()

        view.update_progress_bar(self.progressBar)
        QtWidgets.QApplication.processEvents()
        self.data.outliers_removed = False

    def update_params(self):
        print('not done yet')

    def clear_data(self):
        print('not done yet')

    def save_data(self):
        from datetime import datetime

        num_of_progress_bar_steps = 4 + len(self.data.tab_cluster_data['id'])
        save_true = True

        if save_true:
            view.start_progress_bar(self.progressBar, start=0, stop=num_of_progress_bar_steps)

            # get directory to save to
            self.data.save_folder = QtWidgets.QFileDialog.getExistingDirectory(caption='Select directory to save output',
                                                                               directory=os.path.dirname(self.data.file_name))

            # make subdirectories if they don't exist

            if not os.path.isdir(os.path.join(self.data.save_folder, 'ternary_plots')):
                os.makedirs(os.path.join(self.data.save_folder, 'ternary_plots'))

            if not os.path.isdir(os.path.join(self.data.save_folder, 'cluster_backgates')):
                os.makedirs(os.path.join(self.data.save_folder, 'cluster_backgates'))

            if not os.path.isdir(os.path.join(self.data.save_folder, 'bar_graphs_and_cluster_plots')):
                os.makedirs(os.path.join(self.data.save_folder, 'bar_graphs_and_cluster_plots'))

            if not os.path.isdir(os.path.join(self.data.save_folder, 'cluster_solutions')):
                os.makedirs(os.path.join(self.data.save_folder, 'cluster_solutions'))

            if not os.path.isdir(os.path.join(self.data.save_folder, 'cluster_summaries')):
                os.makedirs(os.path.join(self.data.save_folder, 'cluster_summaries'))

            # self.data.tab_cluster_data.to_pickle(path=os.path.join(self.data.save_folder,
            #                                                        self.data.sample_name + '_Summary.pkl'),
            #                                      compression=None)
            self.data.tab_cluster_data.to_csv(os.path.join(self.data.save_folder, 'cluster_summaries',
                                                           self.data.sample_name + '_Summary.csv'),
                                              index=False, header=True)

            if self.data.outliers_removed:
                cluster_solution = self.data.raw_filtered
                cluster_solution.insert(0, 'clusterID', self.data.cluster_data_idx)
            else:
                cluster_solution = self.data.raw
                cluster_solution.insert(loc=0, column='clusterID', value=pd.Series(self.data.cluster_data_idx))

            # cluster_solution.to_pickle(path=os.path.join(self.data.save_folder,
            #                                              self.data.sample_name + '_cluster_solution.pkl'),
            #                            compression=None)

            cluster_solution.to_csv(path_or_buf=os.path.join(self.data.save_folder, 'cluster_solutions',
                                                             self.data.sample_name + '_cluster_solution.csv'),
                                    index=False, header=True)

            metadata_output = {'fcs_file': self.data.file_name,
                               'date_and_time': datetime.now(),
                               'original_sample_size': self.data.data_size,
                               'sample_size': cluster_solution.shape[0],
                               'HDBSCAN_min_cluster_size': self.clusterMinClusterSize.text(),
                               'HDBSCAN_min_samples': self.clusterMinSamples.text(),
                               'noise_cluster_idx': self.data.noise_cluster_idx,
                               'red_only_idx': self.data.red_only_cluster_idx,
                               'gini_coefficient': self.data.gini,
                               'shannon_entropy': self.data.shannon
                               }

            del cluster_solution

            metadata_output = pd.DataFrame.from_dict(metadata_output, orient='index')
            metadata_output.to_csv(path_or_buf=os.path.join(self.data.save_folder, 'cluster_solutions',
                                                            self.data.sample_name + '_metadata.csv'),
                                   index=True, header=False)

            view.update_progress_bar(self.progressBar)
            QtWidgets.QApplication.processEvents()

        make_graph_output = True

        if make_graph_output:
            self.data.make_output_plots(scale=self.ternScaleOption.currentIndex(),
                                        color=self.ternColorOption.currentIndex(),
                                        progress_bar=self.progressBar)

    def restore_data(self):
        # reinitialize auto_cluster data
        self.data.tab_cluster_data = pd.Series
        self.data.auto_cluster_idx = []

        # load fcs file
        sample_size = self.clusterSampleSize.text()
        sample_size = int(sample_size)
        self.data.fcs_read(sample_size, reload=True)

        # fill parameter table
        self.data.param_combo_box_list = view.init_param_table(self.parameterTable, self.data.params)

        # transform data
        self.data.transform_data()


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

        # print successful load and display number of cells
        self.fileLabel.setText(self.data.sample_name + '\n' + self.data.data_size.__str__() + ' total cells (' +
                               self.data.raw.shape[0].__str__() + ' sampled cells) - RELOADED')

        self.cluster_data()
        self.giniCoeff.setText(str(self.data.gini))
        self.shannonEntropy.setText(str(self.data.shannon))
        self.data.outliers_removed = False

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
        self.data.outliers_removed = True

        self.fileLabel.setText(self.data.sample_name + '\n' + self.data.data_size.__str__() + ' total cells (' +
                               self.data.raw_filtered.shape[0].__str__() + ' sampled cells)')

        self.cluster_data()
        self.giniCoeff.setText(str(self.data.gini))
        self.shannonEntropy.setText(str(self.data.shannon))

        self.update_plots()

    def cluster_data(self):
        # auto cluster the data
        eval_cluster_bool = self.evaluateClusteringCheckBox.isChecked()

        self.data.auto_cluster(self.clusterOnData.currentIndex(), int(self.clusterMinClusterSize.text()),
                               int(self.clusterMinSamples.text()), evaluate_cluster=eval_cluster_bool)
        self.giniCoeff.setText(str(self.data.gini))
        self.shannonEntropy.setText(str(self.data.shannon))

        view.update_cluster_table(self.clusterInformationTable, self.data.tab_cluster_data)

        # self.data.decision_graph(self.clusterOnData.currentIndex())

        self.update_plots()

    def join_cluster(self):
        eval_cluster_bool = self.evaluateClusteringCheckBox.isChecked()

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
                if 'red only' in str(temp_table_object):
                    temp_table_object = self.data.red_only_cluster_idx
                clusters_to_join.append(temp_table_object)
                clusters_to_join[i] = int(clusters_to_join[i])
                con = True

        if con:
            self.data.join_clusters_together(clusters_to_join, self.clusterOnData.currentIndex(),
                                             evaluate_cluster=eval_cluster_bool)
            self.giniCoeff.setText(str(self.data.gini))
            self.shannonEntropy.setText(str(self.data.shannon))

            view.update_cluster_table(self.clusterInformationTable, self.data.tab_cluster_data)
            self.update_plots()

    def highlight_cluster(self):
        table_object = self.clusterInformationTable.selectedItems()
        clusters_to_highlight = []

        for i in range(0, len(table_object)):
            temp_table_object = table_object[i].text()

            if 'noise' in str(temp_table_object):
                clusters_to_highlight.append(int(self.data.noise_cluster_idx))
            elif 'red only' in str(temp_table_object):
                clusters_to_highlight.append(int(self.data.red_only_cluster_idx))
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
        eval_cluster_bool = self.evaluateClusteringCheckBox.isChecked()

        cluster_to_split = self.clusterInformationTable.selectedItems()
        if len(cluster_to_split) > 1:
            print('Warning: This function can only split one cluster at a time and chose the first selected cluster')
            #TODO change this to a dialog message box

        cluster_to_split = cluster_to_split[0].text()
        print('cluster to split \n', cluster_to_split)

        if 'noise' in str(cluster_to_split):
            print('Can not split noise cluster')  # TODO change this to a dialog message box

        else:
            if 'red only' in str(cluster_to_split):
                cluster_to_split = self.data.red_only_cluster_idx
            cluster_to_split = int(cluster_to_split)
            self.data.split_cluster_in_two(cluster_to_split, self.clusterOnData.currentIndex(),
                                           evaluate_cluster=eval_cluster_bool)
            self.giniCoeff.setText(str(self.data.gini))
            self.shannonEntropy.setText(str(self.data.shannon))

            view.update_cluster_table(self.clusterInformationTable, self.data.tab_cluster_data)
            self.update_plots()

    def evaluate_clusters(self):
        eval_cluster_bool = self.evaluateClusteringCheckBox.isChecked()

        if eval_cluster_bool:
            data = self.data.get_data_to_cluster_on(self.clusterOnData.currentIndex())

            self.data.evaluate_cluster_solution(data)
            self.data.make_tabulated_cluster_data()
            view.update_cluster_table(self.clusterInformationTable, self.data.tab_cluster_data)

    def update_plots(self):
        self.data.zbow_3d_plot(self.scatter3DWindow,
                               scale=self.scatterScaleOption.currentIndex(),
                               color=self.scatterColorOption.currentIndex(),
                               update=True)

        self.data.zbow_2d_plot(self.tern2DWindow,
                               scale=self.ternScaleOption.currentIndex(),
                               color=self.ternColorOption.currentIndex(),
                               update=True)

    # def closeEvent(self, event):
    #
    #     self.save_pref()
    #     can_exit = True
    #
    #     if can_exit:
    #         event.accept()  # let the window close
    #     else:
    #         event.ignore()

    def save_pref(self):
        new_pref = {'sample_size': str(self.clusterSampleSize.text()),
                    'HDBSCAN_min_cluster_size': str(self.clusterMinClusterSize.text()),
                    'HDBSCAN_min_samples': str(self.clusterMinSamples.text()),
                    'cluster_data_list': str(self.clusterOnData.currentText()),
                    'scatter_color_list': str(self.scatterColorOption.currentText()),
                    'scatter_scale_list': str(self.scatterScaleOption.currentText()),
                    'tern_color_list': str(self.ternColorOption.currentText()),
                    'tern_scale_list': str(self.ternScaleOption.currentText())
                    }

        new_pref_output = pd.DataFrame.from_dict(new_pref, orient='index')

        new_pref_output.to_csv(path_or_buf='bin/pref.csv', index=True, header=False)


if __name__ == "__main__":
    import os
    import pandas as pd

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon('bin/logo.png'))

    screen = app.primaryScreen()
    size = screen.size()
    height = size.height()
    width = size.width()

    dialog = QtWidgets.QMainWindow()
    dialog.move(0.05*width, 0.1*height)
    prog = Main(dialog)
    prog.screen_size = [size.width(), size.height()]

    # write/read preferences file
    pref_file_exists = os.path.isfile('bin/pref.csv')

    if pref_file_exists:
        pref = pd.read_csv('bin/pref.csv', index_col=0, header=None)

        cluster_sample_size = pref.loc['sample_size']
        cluster_sample_size = cluster_sample_size.iloc[0]

        cluster_min_cluster_size = pref.loc['HDBSCAN_min_cluster_size']
        cluster_min_cluster_size = cluster_min_cluster_size.iloc[0]

        cluster_min_samples = pref.loc['HDBSCAN_min_samples']
        cluster_min_samples = cluster_min_samples.iloc[0]

        cluster_data_list = str(pref.loc['cluster_data_list'][1])
        if cluster_data_list == 'custom ternary':
            prog.clusterOnData.setCurrentIndex(0)
        elif cluster_data_list == 'custom rgb':
            prog.clusterOnData.setCurrentIndex(1)
        elif cluster_data_list == 'default ternary':
            prog.clusterOnData.setCurrentIndex(2)
        elif cluster_data_list == 'default rgb':
            prog.clusterOnData.setCurrentIndex(3)
        elif cluster_data_list == 'linear ternary':
            prog.clusterOnData.setCurrentIndex(4)
        elif cluster_data_list == 'linear rgb':
            prog.clusterOnData.setCurrentIndex(5)

        tern_color_list = str(pref.loc['tern_color_list'][1])
        if tern_color_list == 'custom':
            prog.ternColorOption.setCurrentIndex(0)
        elif tern_color_list == 'default':
            prog.ternColorOption.setCurrentIndex(1)
        elif tern_color_list == 'cluster color':
            prog.ternColorOption.setCurrentIndex(2)
        elif tern_color_list == 'linear':
            prog.ternColorOption.setCurrentIndex(3)

        tern_scale_list = str(pref.loc['tern_scale_list'][1])
        if tern_scale_list == 'custom':
                prog.ternScaleOption.setCurrentIndex(0)
        elif tern_scale_list == 'default':
                prog.ternScaleOption.setCurrentIndex(1)
        elif tern_scale_list == 'cluster color':
                prog.ternScaleOption.setCurrentIndex(2)
        elif tern_scale_list == 'linear':
                prog.ternScaleOption.setCurrentIndex(3)

        scatter_color_list = str(pref.loc['scatter_color_list'][1])
        if scatter_color_list == 'custom':
            prog.scatterColorOption.setCurrentIndex(0)
        elif scatter_color_list == 'default':
            prog.scatterColorOption.setCurrentIndex(1)
        elif scatter_color_list == 'cluster color':
            prog.scatterColorOption.setCurrentIndex(2)
        elif scatter_color_list == 'linear':
            prog.scatterColorOption.setCurrentIndex(3)

        scatter_scale_list = str(pref.loc['scatter_scale_list'][1])
        if scatter_scale_list == 'custom':
            prog.scatterScaleOption.setCurrentIndex(0)
        elif scatter_scale_list == 'default':
            prog.scatterScaleOption.setCurrentIndex(1)
        elif scatter_scale_list == 'cluster color':
            prog.scatterScaleOption.setCurrentIndex(2)
        elif scatter_scale_list == 'linear':
            prog.scatterScaleOption.setCurrentIndex(3)

        prog.clusterSampleSize.setText(str(cluster_sample_size))
        prog.clusterMinClusterSize.setText(str(cluster_min_cluster_size))
        prog.clusterMinSamples.setText(str(cluster_min_samples))

    else:
        pref = {'sample_size': '20000',
                'HDBSCAN_min_cluster_size': '25',
                'HDBSCAN_min_samples': '1',
                'cluster_data_list': 'custom ternary',
                'scatter_color_list': 'custom',
                'scatter_scale_list': 'default',
                'tern_color_list': 'custom',
                'tern_scale_list': 'custom'
                }

        pref_output = pd.DataFrame.from_dict(pref, orient='index')

        pref_output.to_csv(path_or_buf='bin/pref.csv', index=True, header=False)

    app.aboutToQuit.connect(prog.save_pref)
    dialog.show()
    sys.exit(app.exec_())
