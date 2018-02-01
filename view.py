# methods
from PyQt5 import QtWidgets
import re
from itertools import chain


def initParamTable(parameter_table, params):

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
        parameter_table.setItem(p, 0, QtWidgets.QTableWidgetItem(expected_params_labels[p]))
        combo_box_list.append(QtWidgets.QComboBox())
        combo_box_list[p].insertItems(0, params)
        parameter_table.setCellWidget(p, 1, combo_box_list[p])

        combo_box_list[p].setCurrentIndex(param_index[p])

    return combo_box_list


