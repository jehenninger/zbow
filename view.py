# methods
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
import re
from itertools import chain


def init_param_table(parameter_table, params):

    expected_params_labels = ['FSC-A',
                              'FSC-H',
                              'FSC-W',
                              'SSC-A',
                              'SSC-H',
                              'SSC-W',
                              'RFP',
                              'YFP',
                              'CFP',
                              'RFP AutoF',
                              'CFP AutoF',
                              'YFP AutoF',
                              'Time',
                              'Event #']

    expected_params = ['FSC-A',
                       'FSC-H',
                       'FSC-W',
                       'SSC-A',
                       'SSC-H',
                       'SSC-W',
                       '(?=Comp).*PE|(?=Comp).*Red',
                       '(?=Comp).*FITC|(?=Comp).*GFP',
                       '(?=Comp).*CFP|(?=Comp).*DAPI|(?=Comp).*Blue',
                       '.*mCherry|.*PE-Cy7',
                       '.*Hoechst',
                       '.*PerCP|.*YFP',
                       'Time',
                       'Event']

    num_of_params = expected_params.__len__()

    parameter_table.setRowCount(num_of_params)

    # get index where params equals expected_params
    param_index = []
    count = 0
    for s in expected_params:
        param_index.append([i for i, item in enumerate(params) if re.search(s, item)])

        if not param_index[count]:
            param_index[count] = [0]  # put first index in cases where we can't find the parameter

        count = count + 1

    param_index = list(chain.from_iterable(param_index))

    # loop through expected params, print them to table, and print their counterparts in second column combo box
    combo_box_list = []
    for p in range(0, num_of_params):
        item = QtWidgets.QTableWidgetItem(expected_params_labels[p])
        parameter_table.setItem(p, 0, item)
        combo_box_list.append(QtWidgets.QComboBox())
        combo_box_list[p].insertItems(0, params)
        parameter_table.setCellWidget(p, 1, combo_box_list[p])

        combo_box_list[p].setCurrentIndex(param_index[p])

    return combo_box_list


def update_cluster_table(cluster_table, tab_cluster_data):
    num_of_clusters = tab_cluster_data.shape
    num_of_clusters = num_of_clusters[0]

    cluster_table.setRowCount(num_of_clusters)

    for c in range(0, num_of_clusters):
        cluster_id = QtWidgets.QTableWidgetItem(str(tab_cluster_data.iloc[c]['id']))
        #cluster_id.setFlags(QtCore.Qt.ItemIsSelectable)
        cluster_id.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

        num_of_cells = QtWidgets.QTableWidgetItem(str(tab_cluster_data.iloc[c]['num of cells']))

        percentage = QtWidgets.QTableWidgetItem(str(tab_cluster_data.iloc[c]['percentage']))

        if tab_cluster_data.iloc[c]['mean sil'] == -2:
            mean_sil = QtWidgets.QTableWidgetItem()
        else:
            mean_sil = QtWidgets.QTableWidgetItem(str(tab_cluster_data.iloc[c]['mean sil']))

        cluster_color_r = int(tab_cluster_data.iloc[c]['mean R'] * 255)
        cluster_color_g = int(tab_cluster_data.iloc[c]['mean G'] * 255)
        cluster_color_b = int(tab_cluster_data.iloc[c]['mean B'] * 255)

        cluster_table.setItem(c, 0, cluster_id)
        cluster_id.setBackground(QtGui.QColor(cluster_color_r, cluster_color_g, cluster_color_b))
        cluster_id.setForeground(QtGui.QColor('white'))
        cluster_id.setTextAlignment(QtCore.Qt.AlignCenter)

        cluster_table.setItem(c, 1, num_of_cells)
        num_of_cells.setTextAlignment(QtCore.Qt.AlignRight)

        cluster_table.setItem(c, 2, percentage)
        percentage.setTextAlignment(QtCore.Qt.AlignRight)

        cluster_table.setItem(c, 3, mean_sil)
        mean_sil.setTextAlignment(QtCore.Qt.AlignRight)

        # color background of silhouette values if they are below a threshold of 0.6
        if tab_cluster_data.iloc[c]['mean sil'] > 0.6:
            mean_sil.setBackground(QtGui.QColor(0, 200, 0, 50))
        elif tab_cluster_data.iloc[c]['mean sil'] == -2:
            pass
        else:
            mean_sil.setBackground(QtGui.QColor(200, 0, 0, 50))

    # cluster_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers & QtWidgets.QAbstractItemView.DoubleClicked)


def start_progress_bar(progress_bar, start, stop):
    progress_bar.setMinimum(start)
    progress_bar.setMaximum(stop)
    progress_bar.setValue(start)


def update_progress_bar(progress_bar, steps=1):
    progress_bar.setValue(progress_bar.value() + steps)

    if progress_bar.value() == progress_bar.maximum():
        progress_bar.reset()