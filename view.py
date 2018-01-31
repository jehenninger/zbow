# methods
from PyQt5 import QtWidgets

def updateParamTable(parameter_table, params):
    num_of_params = params.__len__()

    parameter_table.setRowCount(num_of_params)

    for p in range(0,num_of_params):
        parameter_table.setItem(p, 0, QtWidgets.QTableWidgetItem(params[p]))
