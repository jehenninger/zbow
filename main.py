import sys
from window import Ui_MainWindow
import model
import view
import helper
import pandas as pd
import os
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from matplotlib import pyplot as plt
import numpy as np
from random import uniform
from matplotlib import ticker

from PyQt5 import QtOpenGL  # necessary for pyinstaller
from PyQt5 import QtTest  # necessary for pyinstaller


class ScatterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ScatterWindow, self).__init__()
        self.was_closed = False

    def closeEvent(self, event):
        self.was_closed = True


class ErrorDialog(QtWidgets.QMessageBox):
    def __init__(self):
        super(ErrorDialog, self).__init__()


class Main(Ui_MainWindow):
    def __init__(self, MainWindow, size):
        super(Main, self).__init__()
        # Ui_MainWindow.__init__(self)
        self.setupUi(MainWindow)

        # get operating system
        self.OS = sys.platform

        # init screen size
        self.screen_size = size
        width, height = size.width(), size.height()
        MainWindow.resize(0.50 * width, 0.7 * height)
        MainWindow.move(0.02 * width, 0.02 * height)

        # instance data class and other windows
        self.data = model.SessionData()

        self.scatter3DWindow = ScatterWindow()
        self.tern2DWindow = ScatterWindow()
        self.error_dialog = ErrorDialog()

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
        self.actionMake_cluster_plots.triggered.connect(self.make_cluster_plots)

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

        self.scatter3DWindow.move(0.60 * self.screen_size.width(), 0.05 * self.screen_size.height())
        self.scatter3DWindow.resize(0.20 * self.screen_size.width(), 0.20*self.screen_size.width())

        self.tern2DWindow.move(0.60 * self.screen_size.width(), 0.45 * self.screen_size.height())
        self.tern2DWindow.resize(0.20 * self.screen_size.width(), 0.20*self.screen_size.width())

        view.start_progress_bar(self.progressBar, start=0, stop=4)

        self.data.screen_size = self.screen_size
        self.data.OS = self.OS

        # reinitialize auto_cluster data
        self.data.tab_cluster_data = pd.Series
        self.data.auto_cluster_idx = []

        # load fcs file
        sample_size = self.clusterSampleSize.text()
        sample_size = int(sample_size)
        successful_load = self.data.fcs_read(sample_size)

        if successful_load:
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

            self.data.zbow_3d_plot(self.scatter3DWindow,
                                   scale=self.scatterScaleOption.currentIndex(),
                                   color=self.scatterColorOption.currentIndex())

            self.data.zbow_2d_plot(self.tern2DWindow,
                                   scale=self.ternScaleOption.currentIndex(),
                                   color=self.ternColorOption.currentIndex())

            view.update_progress_bar(self.progressBar)
            QtWidgets.QApplication.processEvents()
            self.cluster_data()

            view.update_progress_bar(self.progressBar)
            QtWidgets.QApplication.processEvents()
            self.data.outliers_removed = False
        else:
            helper.error_message(self.error_dialog, 'Could not load .fcs or .csv file')

    def update_params(self):

        self.scatter3DWindow.move(0.60 * self.screen_size.width(), 0.05 * self.screen_size.height())
        self.scatter3DWindow.resize(0.20 * self.screen_size.width(), 0.20 * self.screen_size.width())

        self.tern2DWindow.move(0.60 * self.screen_size.width(), 0.45 * self.screen_size.height())
        self.tern2DWindow.resize(0.20 * self.screen_size.width(), 0.20 * self.screen_size.width())

        # reinitialize auto_cluster data
        self.data.tab_cluster_data = pd.Series
        self.data.auto_cluster_idx = []

        # transform data
        self.data.transform_data()

        # initialize 2D and 3D zbow graph
        self.data.zbow_3d_plot(self.scatter3DWindow,
                               scale=self.scatterScaleOption.currentIndex(),
                               color=self.scatterColorOption.currentIndex())

        self.data.zbow_2d_plot(self.tern2DWindow,
                               scale=self.ternScaleOption.currentIndex(),
                               color=self.ternColorOption.currentIndex())

        # print successful load and display number of cells
        self.fileLabel.setText(self.data.sample_name + '\n' + self.data.data_size.__str__() + ' total cells (' +
                               self.data.raw.shape[0].__str__() + ' sampled cells) - PARAMETERS UPDATED')

        self.cluster_data()
        self.giniCoeff.setText(str(self.data.gini))
        self.shannonEntropy.setText(str(self.data.shannon))
        self.data.outliers_removed = False

    def clear_data(self):
        print('not done yet')

    def save_data(self):
        from datetime import datetime

        num_of_progress_bar_steps = 4 + len(self.data.tab_cluster_data['id'])

        view.start_progress_bar(self.progressBar, start=0, stop=num_of_progress_bar_steps)

        # get directory to save to
        self.data.save_folder = QtWidgets.QFileDialog.getExistingDirectory(caption='Select directory to save output',
                                                                           directory=os.path.dirname(self.data.file_name))

        if self.data.save_folder:
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

            self.data.make_output_plots(scale=self.ternScaleOption.currentIndex(),
                                        color=self.ternColorOption.currentIndex(),
                                        progress_bar=self.progressBar)
        else:
            helper.error_message(self.error_dialog, 'Could not retrieve directory to save to')

    def restore_data(self):

        self.scatter3DWindow.move(0.60 * self.screen_size.width(), 0.05 * self.screen_size.height())
        self.scatter3DWindow.resize(0.20 * self.screen_size.width(), 0.20 * self.screen_size.width())

        self.tern2DWindow.move(0.60 * self.screen_size.width(), 0.45 * self.screen_size.height())
        self.tern2DWindow.resize(0.20 * self.screen_size.width(), 0.20 * self.screen_size.width())

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
        self.data.zbow_3d_plot(self.scatter3DWindow,
                               scale=self.scatterScaleOption.currentIndex(),
                               color=self.scatterColorOption.currentIndex())

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

        if table_object:
            for i in range(0, len(table_object)):
                temp_table_object = table_object[i].text()

                if 'noise' in str(temp_table_object):
                    helper.error_message(self.error_dialog, 'Can not join noise cluster')
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
        else:
            helper.error_message(self.error_dialog, 'No clusters selected')

    def highlight_cluster(self):
        table_object = self.clusterInformationTable.selectedItems()
        clusters_to_highlight = []

        if table_object:
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
        else:
            helper.error_message(self.error_dialog, 'No clusters selected')

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
        if cluster_to_split:
            if len(cluster_to_split) > 1:
                helper.error_message(self.error_dialog,
                                     'Warning: This function can only split one cluster at '
                                     'a time and chose the first selected cluster')

            cluster_to_split = cluster_to_split[0].text()

            if 'noise' in str(cluster_to_split):
                helper.error_message(self.error_dialog, 'Can not split noise cluster')
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
        else:
            helper.error_message(self.error_dialog, 'No cluster selected')

    def evaluate_clusters(self):
        eval_cluster_bool = self.evaluateClusteringCheckBox.isChecked()

        if eval_cluster_bool:
            data = self.data.get_data_to_cluster_on(self.clusterOnData.currentIndex())

            self.data.evaluate_cluster_solution(data)
            self.data.make_tabulated_cluster_data()
            view.update_cluster_table(self.clusterInformationTable, self.data.tab_cluster_data)

    def update_plots(self):
        if self.data.file_name:
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

    def make_cluster_plots(self):
        if isinstance(self.data.path_name, str):
            default_path = self.data.path_name
        else:
            default_path = '~/'

        file_list = QtWidgets.QFileDialog.getOpenFileNames(caption='Select cluster summary files',
                                                           directory=default_path,
                                                           filter='*Summary.csv')

        QtWidgets.QApplication.processEvents()

        if file_list[0]:
            view.start_progress_bar(self.progressBar, start=0, stop=len(file_list[0])-1)
            path_name = os.path.dirname(os.path.abspath(file_list[0][0]))
            cluster_figure, cluster_ax = plt.subplots()
            cluster_figure.set_dpi(300)
            sample_name = list()
            for i, file in enumerate(file_list[0]):
                sample_name.append(os.path.splitext(os.path.basename(file))[0])
                tab_data = pd.read_csv(file)
                tab_data = tab_data[tab_data['id'] != 'noise']
                bar_data = tab_data['percentage']
                bar_color = tab_data[['mean R', 'mean G', 'mean B']]
                cluster_ax.boxplot(bar_data, positions=[i+1], sym='', vert=True, medianprops=dict(color='k'))

                x_coord = [i+1] * len(bar_data)

                bar_data_square = [j ** 2 for j in bar_data]

                x_fudge_factor = np.divide(x_coord, bar_data_square)
                x_fudge_factor[x_fudge_factor > 0.2] = 0.2
                x_fudge_factor[x_fudge_factor < 0.02] = 0.02

                x_fudge_choice = [uniform(-x_fudge_factor[k], x_fudge_factor[k]) for k, val in enumerate(x_fudge_factor)]

                x_coord = np.array(x_coord) + np.array(x_fudge_choice)

                bar_color['alpha'] = [0.7] * len(bar_color)
                bar_color = [tuple(x) for x in bar_color.values]

                cluster_ax.scatter(x_coord, bar_data, s=100, c=bar_color)

                view.update_progress_bar(self.progressBar)
                QtWidgets.QApplication.processEvents()

            plt.ylim(0, 100)
            plt.xlim(0.5, len(file_list[0])+0.5)

            sample_name = [x.replace('_Summary', '') for x in sample_name]
            sample_name = [x[0:20] for x in sample_name]

            plt.xticks(range(1, len(file_list[0])+1), sample_name)

            cluster_ax.tick_params(axis='x', labelsize='x-small')
            cluster_ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

            plt.savefig(os.path.join(path_name, 'combined_cluster_graph.png'),
                        dpi=300, transparent=True, pad_inches=0, Bbox='tight')

            plt.savefig(os.path.join(path_name, 'combined_cluster_graph.eps'),
                        dpi=300, transparent=True, pad_inches=0, Bbox='tight')

            plt.close(cluster_figure)
        else:
            helper.error_message(self.error_dialog, 'No files selected')

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

        if getattr(sys, 'frozen', False):
            # running in a bundle
            bundle_dir = sys._MEIPASS
            pref_path = os.path.join(bundle_dir, 'bin/pref.csv')
        else:
            # running live
            pref_path = 'bin/pref.csv'

        new_pref_output.to_csv(path_or_buf=pref_path, index=True, header=False)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    if getattr(sys, 'frozen', False):
        # running in a bundle
        bundle_dir = sys._MEIPASS
        app.setWindowIcon(QtGui.QIcon(os.path.join(bundle_dir, 'bin/logo.png')))
    else:
        # running live
        app.setWindowIcon(QtGui.QIcon('bin/logo.png'))

    screen = app.primaryScreen()
    size = screen.size()

    MainWindow = QtWidgets.QMainWindow()

    ui = Main(MainWindow, size)

    # write/read preferences file
    if getattr(sys, 'frozen', False):
        # running in a bundle
        bundle_dir = sys._MEIPASS
        pref_file_exists = os.path.isfile(os.path.join(bundle_dir, 'bin/pref.csv'))
        pref_path = os.path.join(bundle_dir, 'bin/pref.csv')
    else:
        # running live
        pref_file_exists = os.path.isfile('bin/pref.csv')
        pref_path = 'bin/pref.csv'

    if pref_file_exists:
        pref = pd.read_csv(pref_path, index_col=0, header=None)

        cluster_sample_size = pref.loc['sample_size']
        cluster_sample_size = cluster_sample_size.iloc[0]

        cluster_min_cluster_size = pref.loc['HDBSCAN_min_cluster_size']
        cluster_min_cluster_size = cluster_min_cluster_size.iloc[0]

        cluster_min_samples = pref.loc['HDBSCAN_min_samples']
        cluster_min_samples = cluster_min_samples.iloc[0]

        cluster_data_list = str(pref.loc['cluster_data_list'][1])
        if cluster_data_list == 'custom ternary':
            ui.clusterOnData.setCurrentIndex(0)
        elif cluster_data_list == 'custom rgb':
            ui.clusterOnData.setCurrentIndex(1)
        elif cluster_data_list == 'default ternary':
            ui.clusterOnData.setCurrentIndex(2)
        elif cluster_data_list == 'default rgb':
            ui.clusterOnData.setCurrentIndex(3)
        elif cluster_data_list == 'linear ternary':
            ui.clusterOnData.setCurrentIndex(4)
        elif cluster_data_list == 'linear rgb':
            ui.clusterOnData.setCurrentIndex(5)

        tern_color_list = str(pref.loc['tern_color_list'][1])
        if tern_color_list == 'custom':
            ui.ternColorOption.setCurrentIndex(0)
        elif tern_color_list == 'default':
            ui.ternColorOption.setCurrentIndex(1)
        elif tern_color_list == 'cluster color':
            ui.ternColorOption.setCurrentIndex(2)
        elif tern_color_list == 'linear':
            ui.ternColorOption.setCurrentIndex(3)

        tern_scale_list = str(pref.loc['tern_scale_list'][1])
        if tern_scale_list == 'custom':
                ui.ternScaleOption.setCurrentIndex(0)
        elif tern_scale_list == 'default':
                ui.ternScaleOption.setCurrentIndex(1)
        elif tern_scale_list == 'cluster color':
                ui.ternScaleOption.setCurrentIndex(2)
        elif tern_scale_list == 'linear':
                ui.ternScaleOption.setCurrentIndex(3)

        scatter_color_list = str(pref.loc['scatter_color_list'][1])
        if scatter_color_list == 'custom':
            ui.scatterColorOption.setCurrentIndex(0)
        elif scatter_color_list == 'default':
            ui.scatterColorOption.setCurrentIndex(1)
        elif scatter_color_list == 'cluster color':
            ui.scatterColorOption.setCurrentIndex(2)
        elif scatter_color_list == 'linear':
            ui.scatterColorOption.setCurrentIndex(3)

        scatter_scale_list = str(pref.loc['scatter_scale_list'][1])
        if scatter_scale_list == 'custom':
            ui.scatterScaleOption.setCurrentIndex(0)
        elif scatter_scale_list == 'default':
            ui.scatterScaleOption.setCurrentIndex(1)
        elif scatter_scale_list == 'cluster color':
            ui.scatterScaleOption.setCurrentIndex(2)
        elif scatter_scale_list == 'linear':
            ui.scatterScaleOption.setCurrentIndex(3)

        ui.clusterSampleSize.setText(str(cluster_sample_size))
        ui.clusterMinClusterSize.setText(str(cluster_min_cluster_size))
        ui.clusterMinSamples.setText(str(cluster_min_samples))

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

        if getattr(sys, 'frozen', False):
            # running in a bundle
            bundle_dir = sys._MEIPASS
            pref_output.to_csv(path_or_buf=os.path.join(bundle_dir, 'bin/pref.csv'), index=True, header=False)
        else:
            # running live
            pref_output.to_csv(path_or_buf='bin/pref.csv', index=True, header=False)

    app.aboutToQuit.connect(ui.save_pref)
    MainWindow.show()
    sys.exit(app.exec_())
