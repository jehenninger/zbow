# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/jon/qt_projects/zbow/mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
# Here is the tool to convert the .ui > .py conversion: pyuic5

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(952, 734)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_15 = QtWidgets.QLabel(self.groupBox)
        self.label_15.setObjectName("label_15")
        self.verticalLayout.addWidget(self.label_15)
        self.fileLabel = QtWidgets.QLabel(self.groupBox)
        self.fileLabel.setText("")
        self.fileLabel.setObjectName("fileLabel")
        self.verticalLayout.addWidget(self.fileLabel)
        self.removeOutliers = QtWidgets.QPushButton(self.groupBox)
        self.removeOutliers.setObjectName("removeOutliers")
        self.verticalLayout.addWidget(self.removeOutliers)
        self.parameterTable = QtWidgets.QTableView(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.parameterTable.sizePolicy().hasHeightForWidth())
        self.parameterTable.setSizePolicy(sizePolicy)
        self.parameterTable.setObjectName("parameterTable")
        self.verticalLayout.addWidget(self.parameterTable)
        self.horizontalLayout.addWidget(self.groupBox)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_4)
        self.groupBox_2.setAutoFillBackground(True)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.clusterOptionsBox = QtWidgets.QGroupBox(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clusterOptionsBox.sizePolicy().hasHeightForWidth())
        self.clusterOptionsBox.setSizePolicy(sizePolicy)
        self.clusterOptionsBox.setObjectName("clusterOptionsBox")
        self.formLayout = QtWidgets.QFormLayout(self.clusterOptionsBox)
        self.formLayout.setObjectName("formLayout")
        self.label_2 = QtWidgets.QLabel(self.clusterOptionsBox)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.label_3 = QtWidgets.QLabel(self.clusterOptionsBox)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.clusterOnData = QtWidgets.QComboBox(self.clusterOptionsBox)
        self.clusterOnData.setObjectName("clusterOnData")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.clusterOnData)
        self.label_4 = QtWidgets.QLabel(self.clusterOptionsBox)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.clusterPushButton = QtWidgets.QPushButton(self.clusterOptionsBox)
        self.clusterPushButton.setObjectName("clusterPushButton")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.clusterPushButton)
        self.label_5 = QtWidgets.QLabel(self.clusterOptionsBox)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.addCenterPushButton = QtWidgets.QPushButton(self.clusterOptionsBox)
        self.addCenterPushButton.setObjectName("addCenterPushButton")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.addCenterPushButton)
        self.clusterSampleSize = QtWidgets.QLineEdit(self.clusterOptionsBox)
        self.clusterSampleSize.setObjectName("clusterSampleSize")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.clusterSampleSize)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout.setItem(5, QtWidgets.QFormLayout.FieldRole, spacerItem)
        self.label_13 = QtWidgets.QLabel(self.clusterOptionsBox)
        self.label_13.setObjectName("label_13")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.removeCenterPushButton = QtWidgets.QPushButton(self.clusterOptionsBox)
        self.removeCenterPushButton.setObjectName("removeCenterPushButton")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.removeCenterPushButton)
        self.highlightClusterPushButton = QtWidgets.QPushButton(self.clusterOptionsBox)
        self.highlightClusterPushButton.setObjectName("highlightClusterPushButton")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.highlightClusterPushButton)
        self.label_14 = QtWidgets.QLabel(self.clusterOptionsBox)
        self.label_14.setObjectName("label_14")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.label = QtWidgets.QLabel(self.clusterOptionsBox)
        self.label.setObjectName("label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout.setItem(3, QtWidgets.QFormLayout.FieldRole, spacerItem1)
        self.clusterMethod = QtWidgets.QComboBox(self.clusterOptionsBox)
        self.clusterMethod.setObjectName("clusterMethod")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.clusterMethod)
        self.horizontalLayout_2.addWidget(self.clusterOptionsBox)
        self.plotOptionsBox = QtWidgets.QGroupBox(self.groupBox_2)
        self.plotOptionsBox.setObjectName("plotOptionsBox")
        self.formLayout_2 = QtWidgets.QFormLayout(self.plotOptionsBox)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_6 = QtWidgets.QLabel(self.plotOptionsBox)
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.label_7 = QtWidgets.QLabel(self.plotOptionsBox)
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.scatterColorOption = QtWidgets.QComboBox(self.plotOptionsBox)
        self.scatterColorOption.setObjectName("scatterColorOption")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.scatterColorOption)
        self.label_8 = QtWidgets.QLabel(self.plotOptionsBox)
        self.label_8.setObjectName("label_8")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.scatterScaleOption = QtWidgets.QComboBox(self.plotOptionsBox)
        self.scatterScaleOption.setObjectName("scatterScaleOption")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.scatterScaleOption)
        self.label_9 = QtWidgets.QLabel(self.plotOptionsBox)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_9)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout_2.setItem(3, QtWidgets.QFormLayout.LabelRole, spacerItem2)
        self.label_10 = QtWidgets.QLabel(self.plotOptionsBox)
        self.label_10.setObjectName("label_10")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.ternColorOption = QtWidgets.QComboBox(self.plotOptionsBox)
        self.ternColorOption.setObjectName("ternColorOption")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.ternColorOption)
        self.label_11 = QtWidgets.QLabel(self.plotOptionsBox)
        self.label_11.setObjectName("label_11")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.ternScaleOption = QtWidgets.QComboBox(self.plotOptionsBox)
        self.ternScaleOption.setObjectName("ternScaleOption")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.ternScaleOption)
        self.drawGatePushButton = QtWidgets.QPushButton(self.plotOptionsBox)
        self.drawGatePushButton.setObjectName("drawGatePushButton")
        self.formLayout_2.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.drawGatePushButton)
        self.label_12 = QtWidgets.QLabel(self.plotOptionsBox)
        self.label_12.setObjectName("label_12")
        self.formLayout_2.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_12)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.formLayout_2.setItem(7, QtWidgets.QFormLayout.LabelRole, spacerItem3)
        self.horizontalLayout_2.addWidget(self.plotOptionsBox)
        self.verticalLayout_2.addWidget(self.groupBox_2)
        self.clusterInformationBox = QtWidgets.QGroupBox(self.groupBox_4)
        self.clusterInformationBox.setObjectName("clusterInformationBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.clusterInformationBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.clusterInformationTable = QtWidgets.QTableView(self.clusterInformationBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clusterInformationTable.sizePolicy().hasHeightForWidth())
        self.clusterInformationTable.setSizePolicy(sizePolicy)
        self.clusterInformationTable.setObjectName("clusterInformationTable")
        self.verticalLayout_3.addWidget(self.clusterInformationTable)
        self.verticalLayout_2.addWidget(self.clusterInformationBox)
        self.horizontalLayout.addWidget(self.groupBox_4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 952, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuData_processing = QtWidgets.QMenu(self.menubar)
        self.menuData_processing.setObjectName("menuData_processing")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad = QtWidgets.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionClear = QtWidgets.QAction(MainWindow)
        self.actionClear.setObjectName("actionClear")
        self.actionRestore = QtWidgets.QAction(MainWindow)
        self.actionRestore.setObjectName("actionRestore")
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionClear)
        self.menuData_processing.addAction(self.actionRestore)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuData_processing.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "zbow analysis"))
        self.groupBox.setTitle(_translate("MainWindow", "Data info"))
        self.label_15.setText(_translate("MainWindow", "Loaded file"))
        self.removeOutliers.setText(_translate("MainWindow", "remove outliers"))
        self.clusterOptionsBox.setTitle(_translate("MainWindow", "cluster options"))
        self.label_2.setText(_translate("MainWindow", "sample size"))
        self.label_3.setText(_translate("MainWindow", "cluster data"))
        self.label_4.setText(_translate("MainWindow", "cluster"))
        self.clusterPushButton.setText(_translate("MainWindow", "OK"))
        self.label_5.setText(_translate("MainWindow", "add center"))
        self.addCenterPushButton.setText(_translate("MainWindow", "OK"))
        self.label_13.setText(_translate("MainWindow", "remove center"))
        self.removeCenterPushButton.setText(_translate("MainWindow", "OK"))
        self.highlightClusterPushButton.setText(_translate("MainWindow", "OK"))
        self.label_14.setText(_translate("MainWindow", "highlight cluster"))
        self.label.setText(_translate("MainWindow", "cluster method"))
        self.plotOptionsBox.setTitle(_translate("MainWindow", "plot options"))
        self.label_6.setText(_translate("MainWindow", "3D scatter options"))
        self.label_7.setText(_translate("MainWindow", "color"))
        self.label_8.setText(_translate("MainWindow", "scale"))
        self.label_9.setText(_translate("MainWindow", "ternary plot options"))
        self.label_10.setText(_translate("MainWindow", "color"))
        self.label_11.setText(_translate("MainWindow", "scale"))
        self.drawGatePushButton.setText(_translate("MainWindow", "OK"))
        self.label_12.setText(_translate("MainWindow", "draw ternary gate"))
        self.clusterInformationBox.setTitle(_translate("MainWindow", "cluster information"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuData_processing.setTitle(_translate("MainWindow", "Data processing"))
        self.actionLoad.setText(_translate("MainWindow", "Load data"))
        self.actionSave.setText(_translate("MainWindow", "Save data/images"))
        self.actionClear.setText(_translate("MainWindow", "Clear session"))
        self.actionRestore.setText(_translate("MainWindow", "Restore original data"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

