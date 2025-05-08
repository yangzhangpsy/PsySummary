# -*- coding: utf-8 -*-
import csv
import platform
import sys
import traceback
import re
import os

import numpy as np
import pandas as pd
from scipy.io import matlab

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QApplication, QFileDialog, \
    QHBoxLayout, QGridLayout, QLabel, QVBoxLayout, QPushButton, QMenu, QWidget, QMainWindow, QMessageBox, QAction
from PyQt5.QtGui import QKeySequence

from app.lib import DecodingFiles, DraggableListWidget, VariableDraggableListWidget, MainFilterListWidget
from app.lib import ImportMatThread
from app.lib import MessageBox
from app.lib import FilterWindow
from app.lib import DataFrameTableWidget
from app.lib import PivotedDataWidget
from app.lib import ScriptDock

from app.psyDataFunc import PsyDataFunc
from app.psyDataInfo import PsyDataInfo

from app.tool import StatisticTool, FlashMessageBox
from app.variableCompute import VariableCompute


def setListWidgetData(widget, items):
    if hasattr(widget, 'contentList'):
        widget.contentList = items
        # False to keep the content list untouched
        widget.clear(False)
    else:
        widget.clear()

    widget.addItems(items)


def getListWidgetData(widget):
    """Returns a list of all item texts from the given QListWidget."""
    return [widget.item(i).text() for i in range(widget.count())]


def parseStringToList(string):
    start_index = string.find(": ")

    if start_index != -1:
        list_string = string[start_index + 2:]

        filter_list = eval(list_string)

        return filter_list
    else:
        return None


def fixColumnName(name, shouldStartWithLetter: bool = False):

    fixed_name = ''.join(re.findall(r'[a-zA-Z0-9_\-.]', name))

    if shouldStartWithLetter:
        if not fixed_name or not re.match(r'^[a-zA-Z]', fixed_name):
            fixed_name = 'col' + fixed_name  # 添加前缀以满足规则
    return fixed_name


def validateName(data):
    if data is not None:
        if isinstance(data, pd.DataFrame):
            name_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9_\-.]*$')
            return all(data.columns.str.contains(name_pattern.pattern, regex=True))
        else:
            raise TypeError("Invalid data type. Expected a DataFrame.")


def checkMatVersion(fileList: list):
    # Expand the result to match the original list length, marking others as False
    return np.array(
        [(matlab.matfile_version(file)[0] == 2) if (file.endswith('.mat') and os.path.isfile(file)) else False for file
         in fileList], dtype=bool)


def readPsyDataFiles(files):
    try:
        all_dfs = []
        for file in files:
            df = pd.read_csv(file, sep='|', quoting=csv.QUOTE_NONNUMERIC, index_col=False)
            # quoting{0 or csv.QUOTE_MINIMAL, 1 or csv.QUOTE_ALL, 2 or csv.QUOTE_NONNUMERIC, 3 or csv.QUOTE_NONE}, default csv.QUOTE_MINIMAL
            # combined_df = pd.concat([combined_df, df], ignore_index=True)
            all_dfs.append(df)

            PsyDataInfo.PsyData.printLogInfo(f"Reading file: {file}", 0)

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)

            PsyDataFunc.list2Script(files, "fileList")
            PsyDataFunc.genScript(PsyDataFunc.list2Script(files, "fileList"))
            PsyDataFunc.genScript(f"aggData.readPsyDataFiles(fileList)")
            return combined_df
    except Exception as e:
        raise IOError(f"File reading Error: {e}")


class PsyData(QMainWindow):
    def __init__(self):
        super().__init__()

        # self.plugin_mode = not __name__ == "__main__"
        self.files = None
        self.plugin_mode = False
        self.readMatThreads = dict()
        self.pivotTableWindow = None
        self.filterWindow = None
        self.tableFrame = None
        self.variablesNameList = None
        self.import_file = None
        self.data = pd.DataFrame()
        self.dataReadStart = False

        self.is_windows = platform.system() == "Windows"

        self.lst = [None, ' ']
        self.analysisScript = []
        self.FILE_DIRECTORY = ''

        PsyDataInfo.PsyData = self

        if self.plugin_mode:
            self.resize(800, 700)
        else:
            self.resize(800, 700)

        self.setWindowTitle('Data Summary')
        self.setWindowIcon(PsyDataFunc.getImageObject("icon.png", type=1))
        # set the central widget
        self.central_widget = QWidget()
        self.computationVariableGui = VariableCompute(self.data)

        # self.setStyleSheet(default_qss)

        """
        init top menu-bar
        """
        menubar = self.menuBar()
        file_menu: QMenu = menubar.addMenu("&File")
        tool_menu: QMenu = menubar.addMenu("&Toolbox")

        file_menu.addAction("Load Data", self.loadDataFile, QKeySequence(QKeySequence.Open))
        file_menu.addAction("View Data", self.showDataTable, QKeySequence(QKeySequence.WhatsThis))
        file_menu.addAction("Save Data", self.savePsyData, QKeySequence(QKeySequence.Save))
        file_menu.addAction("Save Filtered Data", self.saveFilteredData, QKeySequence(QKeySequence.SaveAs))
        tool_menu.addAction("Transform Variable", self.computationVariable)

        """
        # script dock and action
        """
        self.script_dock = ScriptDock()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.script_dock)
        self.script_dock.realVisibleChanged.connect(self.setActionIcon)

        view_menu: QMenu = menubar.addMenu("&View")

        self.script_action = QAction("&Script", self)
        self.script_action.setData("script")

        if self.is_windows:
            checked_icon = PsyDataFunc.getImageObject("checked", 1)
            self.script_action.setIcon(checked_icon)
            self.script_action.setIconVisibleInMenu(True)
        else:
            self.script_action.setCheckable(True)
            self.script_action.setChecked(True)

        self.script_action.triggered.connect(self.setDockView)

        """
        # output dock and action
        """
        """
        # output dock and action
        """
        if not self.plugin_mode:
            from app.output import Output
            self.output = Output(True)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.output)
            self.output.realVisibleChanged.connect(self.setActionIcon)

            self.output_action = QAction("&Output", self)
            self.output_action.setData("output")

            if self.is_windows:
                checked_icon = PsyDataFunc.getImageObject("checked", 1)
                self.output_action.setIcon(checked_icon)

                self.output_action.setIconVisibleInMenu(True)
            else:
                self.output_action.setCheckable(True)
                self.output_action.setChecked(True)

            self.output_action.triggered.connect(self.setDockView)

            self.tabifyDockWidget(self.output, self.script_dock)

            self.output.setVisible(True)
            self.output.raise_()
            self.output.setFocus()

            view_menu.addAction(self.output_action)
        view_menu.addAction(self.script_action)

        #  lists
        self.rows_list = DraggableListWidget()
        self.columns_list = DraggableListWidget(1)
        self.data_list = DraggableListWidget(2)

        self.variables_list = VariableDraggableListWidget(3)

        self.filter_list = MainFilterListWidget()
        # self.filter_list.setDragDropMode(QListWidget.InternalMove)
        # self.filterWindow = FilterWindow(self.data, self.filter_list)

        # buttons
        self.filter_button = QPushButton('Define Filters')
        self.save_filter_button = QPushButton('Save Filter')
        self.load_filter_button = QPushButton('Load Filter')
        self.run_button = QPushButton('Run')
        self.close_button = QPushButton('Close')

        self.filter_button.clicked.connect(self.defineFilterEvent)
        self.save_filter_button.clicked.connect(self.saveFilterEvent)
        self.load_filter_button.clicked.connect(self.loadFilterEvent)
        self.run_button.clicked.connect(self.runSummary)
        self.close_button.clicked.connect(self.clickCloseEvent)

        self.computationVariableGui.transformFinished.connect(self.transformVariable)

        self.instruct_lab = QLabel()

        self.instruct_lab.setText("""
To summarize data, drag the variable names
from the Variables list on the right into
the Rows/Columns/Data list on the left.
To remove a variable from a list, press
Del key on the keyboard, right-click on
and select 'Delete' from the menu, or
drag the variable back to the variable list.
""")

        # Layout for bottom buttons
        buttons_layout = QHBoxLayout()

        buttons_layout.addWidget(self.filter_button, 1)
        buttons_layout.addWidget(self.save_filter_button, 1)
        buttons_layout.addWidget(self.load_filter_button, 1)
        buttons_layout.addWidget(self.run_button, 1)
        buttons_layout.addWidget(self.close_button, 1)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Add groups to main layout
        main_layout = QGridLayout()
        main_layout.setColumnMinimumWidth(1, 10)
        main_layout.setColumnMinimumWidth(3, 10)

        main_layout.setRowMinimumHeight(3, 20)

        main_layout.addWidget(self.instruct_lab, 0, 0, 2, 1)

        main_layout.addWidget(QLabel("Columns:"), 0, 2, 1, 1)
        main_layout.addWidget(QLabel("Variables:"), 0, 4, 1, 1)
        main_layout.addWidget(self.columns_list, 1, 2, 1, 1)
        main_layout.addWidget(self.variables_list, 1, 4, 3, 1)
        main_layout.addWidget(QLabel("Rows:"), 2, 0, 1, 1)
        main_layout.addWidget(QLabel("Data:"), 2, 2, 1, 1)

        main_layout.addWidget(self.rows_list, 3, 0, 1, 1)
        main_layout.addWidget(self.data_list, 3, 2, 1, 1)

        main_layout.addWidget(QLabel("Filters:"), 5, 0, 1, 1)

        main_layout.addWidget(self.filter_list, 6, 0, 2, 5)

        all_layout = QVBoxLayout()
        all_layout.addLayout(main_layout)
        # all_layout.addStretch(2)
        all_layout.addLayout(buttons_layout)

        # Widget for main layout
        # self.setLayout(all_layout)
        self.central_widget.setLayout(all_layout)
        self.setCentralWidget(self.central_widget)

        # self.statusBar = QStatusBar()
        # self.setStatusBar(self.statusBar)
        self.installEventFilter(self)

    def setDockView(self, checked):
        if self.sender() is self.output_action:
            target_status = self.output.isHidden()
            self.output.setVisible(target_status)

            if self.is_windows:
                self.output_action.setIconVisibleInMenu(target_status)
            else:
                self.output_action.setChecked(target_status)

        elif self.sender() is self.script_action:
            target_status = self.script_dock.isHidden()
            self.script_dock.setVisible(target_status)

            if self.is_windows:
                self.script_action.setIconVisibleInMenu(target_status)
            else:
                self.script_action.setChecked(target_status)

    def setActionIcon(self, isVisible):
        if self.is_windows:
            if self.sender() is self.output:
                self.output_action.setIconVisibleInMenu(self.output.isVisible())
            elif self.sender() is self.script_dock:
                self.script_action.setIconVisibleInMenu(self.script_dock.isVisible())
        else:
            if self.sender() is self.output:
                self.output_action.setChecked(self.output.isVisible())
            elif self.sender() is self.script_dock:
                self.script_action.setChecked(self.script_dock.isVisible())

    def loadDataFile(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog

        if self.FILE_DIRECTORY:
            default_dir = self.FILE_DIRECTORY
        else:
            default_dir = os.path.expanduser('~')

        files, _ = QFileDialog.getOpenFileNames(self, "Select File(s)", default_dir,
                                                "Matlab Files (*.mat);;Text Files (*.txt);;Text Files (*.csv);;Dat Files (*.dat);;psyData Files (*.psydata)",
                                                options=options)

        if files:
            try:
                self.files = files
                _, file_extension = os.path.splitext(files[0])

                if file_extension == '.txt' or file_extension == '.dat' or file_extension == '.csv':
                    self.import_file = DecodingFiles(files)
                    self.import_file.ok_btn.clicked.connect(self.decodingFileOKPressedEvent)
                    self.import_file.show()
                elif file_extension == '.mat':
                    # clear it first
                    # self.data = pd.DataFrame()
                    self.readMatlabFilesMThread(files)
                    # self.clearAllListAndSetData()
                elif file_extension == '.psydata':
                    self.data = readPsyDataFiles(files)
                    self.clearAllListAndSetData()

                self.FILE_DIRECTORY = os.path.dirname(files[0])

            except Exception as e:
                msg_box = FlashMessageBox('Flash Message', str(e))
                msg_box.show()

    def readMatlabFilesMThread(self, fileList):
        """
        Batch read Matlab files and select an appropriate reading method based on the file format version.

        This function first checks if the provided list of Matlab files contains any v7.3 version files. If it does,
        it prompts the user that reading v7.3 files can be very slow and asks if they want to proceed. If the user agrees,
        it uses a special method to read the v7.3 files while using the regular method for other versions. If no v7.3 files
        are found in the list, it directly reads all files using the regular method.

        Parameters:
        - fileList: List containing paths to Matlab files.
        """
        # Check the version of the Mat files in the list to determine if any are v7.3
        isV7 = checkMatVersion(fileList)

        # If there are v7.3 version files
        if any(isV7):
            # Display a warning message to inform the user that reading v7.3 files can be very slow and ask if they want to proceed
            ans = MessageBox.information(self, "Warning",
                                         f"At least one mat file's format is v7.3, we prefer to support V7"
                                         f"\nStrongly suggest to save the data via save(filename, '-v7');\n\n"
                                         f"Are you sure to load the mat v7.3 files via a very very very slow way?",
                                         QMessageBox.Ok,
                                         QMessageBox.Close)
            # If the user chooses to proceed
            if ans == QMessageBox.Ok:
                # Print a warning message indicating that the current loading operation may be very slow
                self.printLogInfo("Warning, try to load the file via mat73, which is very slow...", 4)
                # Filter out the v7.3 version files and read them using a special method
                v7Files = np.array(fileList)[isV7]
                self.readMatFilesThread(v7Files.tolist(), 2)

            # Filter out non-v7.3 version files
            no_v7files = np.array(fileList)[np.logical_not(isV7)]
            # If there are non-v7.3 version files, read them using the regular method
            if no_v7files.size > 0:
                self.readMatFilesThread(no_v7files, 1, True)
        else:
            # If no v7.3 version files are found, read all files using the regular method
            self.readMatFilesThread(fileList)

    def readMatFilesThread(self, fileList: list, matType: int = 1, appendDataModel: bool = False):
        readMatThread = ImportMatThread(fileList, matType, appendDataModel)
        self.readMatThreads.update({matType: readMatThread})

        readMatThread.readStatus.connect(self.handleThreadSignal)
        readMatThread.finished.connect(self.handleReadDataFinished)
        readMatThread.start()

    def handleReadDataFinished(self, data: pd.DataFrame, fileType: int, fileList: list, appendDataModel: bool):
        # Use DataFrame.append for potentially better performance in some cases
        if data.size > 0:
            if not self.dataReadStart:
                self.data = pd.DataFrame()
                self.dataReadStart = True

            self.data = pd.concat([self.data, data], ignore_index=True)

        self.readMatThreads[fileType].wait()
        PsyDataFunc.genScript(PsyDataFunc.list2Script(fileList, 'fileList'))
        if fileType == 1:
            PsyDataFunc.genScript(f"aggData.readMatlabFiles(fileList, {appendDataModel})")
        else:
            PsyDataFunc.genScript(f"aggData.readMatlabFiles73(fileList, {appendDataModel})")

        self.readMatThreads.pop(fileType)

        if not self.readMatThreads:
            self.clearAllListAndSetData()

    def decodingFileOKPressedEvent(self):
        self.data = self.import_file.readFinalData()

        self.import_file.acceptEvent()
        self.clearAllListAndSetData()

        text_format, delimiter = self.import_file.getFormatAndDelimiter()
        PsyDataFunc.genScript(PsyDataFunc.list2Script(self.import_file.files, 'fileList'))
        PsyDataFunc.genScript(
            f"aggData.readDatFiles(fileList, {self.import_file.getContainHeadStatus()},'{text_format}', '{delimiter}')")

    def clearAllListAndSetData(self):
        self.dataReadStart = False
        self.clearAllList()

        if not validateName(self.data):
            self.cleanColumnNames()
            MessageBox.information(self, 'Warning', "At least one of the variable names is illegal.")

        self.setData()

    def cleanColumnNames(self):
        seen_names = {}
        cleaned_columns = []

        for col in self.data.columns:
            new_name = fixColumnName(col, True)  # 清理后的列名

            # 检查是否重复，如果重复则添加后缀 _1, _2, ...
            original_name = new_name
            count = 1
            while new_name in seen_names:
                new_name = f"{original_name}_{count}"
                count += 1

            # 标记这个名字已经被使用
            seen_names[new_name] = True
            cleaned_columns.append(new_name)

        # 重命名 DataFrame 列
        self.data.columns = cleaned_columns
        return None

    def clearAllList(self):
        self.variables_list.clear()
        self.columns_list.clear()
        self.rows_list.clear()
        self.data_list.clear()
        self.filter_list.clear()

    # 读取单个文件
    def readFile(self, file_path):
        try:
            tmp = self.lst
            with open(file_path, 'r', encoding=tmp[0]) as file:
                lines = file.readlines()
                variable_names = lines[0].strip().split(tmp[1])  # 第一行为变量名
                data = [line.strip().split(tmp[1]) for line in lines[1:]]  # 以分隔符分隔的变量值
                df = pd.DataFrame(data, columns=variable_names)
                return df
        except Exception as e:
            self.printLogInfo(f"Error reading file {file_path}: {e}", 3)
            return None

    # 读取多个文件
    def readMultipleFiles(self, fileList):
        all_dfs = []
        try:
            for file in fileList:
                df = self.readFile(file)
                fileName = os.path.basename(file)
                df = df.assign(fileName=fileName)
                if df is not None:
                    all_dfs.append(df)
            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True)
            else:
                return None
        except Exception as e:
            self.printLogInfo(f"Error in reading file:{e}", 3)
            return None

    # 读取matlab文件

    # 定义一个函数，将二维数组或整数值转换为单个值

    def setData(self):
        if self.data is not None:
            # Use fillna with None directly for speed improvement
            # self.data = self.data.fillna(None)
            # self.data = self.data.where(pd.notna(self.data), None)

            self.variablesNameList = self.data.columns.tolist()
            self.variables_list.addItems(self.variablesNameList)
            self.variables_list.sortItems(Qt.AscendingOrder)

    # 显示打开文件的table
    def showDataTable(self):
        try:
            if self.data is None:
                MessageBox.information(self, 'Warning', "No data exist, please load the data first.")
                return False

            df = self.getFilteredDataFrame()
            self.tableFrame = DataFrameTableWidget(df)
            self.tableFrame.show()
        except Exception as e:
            MessageBox.warning(self, "Show Filtered Data Error", f"{e}")
            return None

    # filter 触发事件
    def defineFilterEvent(self):
        if self.data is None or self.data.size == 0:
            MessageBox.information(self, 'Warning', "No data exist, please load data first.")
            return False
        # try:
        self.filterWindow = FilterWindow(self.data, self.filter_list)
        self.filterWindow.show()

    # 运行分析程序
    def runSummary(self):
        if self.data is None:
            MessageBox.information(self, 'Warning', "No data exist, please load data first.")
            return None

        try:
            rowList = getListWidgetData(self.rows_list)
            columnList = getListWidgetData(self.columns_list)
            dataList = getListWidgetData(self.data_list)

            items = self.getFilterList()

            self.pivotTableWindow = PivotedDataWidget(self.data, rowList, columnList, dataList, items)

            main_gui_topLeft = self.getGlobalPosition()
            self.pivotTableWindow.move(main_gui_topLeft.x() + self.frameGeometry().width(), main_gui_topLeft.y())

            self.pivotTableWindow.show()
        except Exception as e:
            MessageBox.information(self, 'Warning', f"{e}")
            traceback.print_exc()
            return None

    def contingentPivotTableWindow(self):
        if self.pivotTableWindow and self.pivotTableWindow.isVisible():
            main_gui_topLeft = self.getGlobalPosition()
            self.pivotTableWindow.move(main_gui_topLeft.x() + self.frameGeometry().width(), main_gui_topLeft.y())

    def eventFilter(self, source, event):
        # 当主窗口移动时，实时同步移动另一个窗口
        if source == self and event.type() == QEvent.Move:
            self.contingentPivotTableWindow()
        return super().eventFilter(source, event)

    def closeEvent(self, event):
        if self.pivotTableWindow:
            self.pivotTableWindow.close()
        super().closeEvent(event)

    def getGlobalPosition(self):
        # Get the frame geometry, which includes the toolbar and window decorations
        global_pos = self.frameGeometry().topLeft()

        return global_pos

    def transformVariable(self, newVariableName):
        self.variablesNameList.append(newVariableName)

        self.variables_list.addItem(newVariableName)
        self.variables_list.sortItems(Qt.AscendingOrder)

    def clickCloseEvent(self):
        self.close()

    def loadFilterEvent(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'PsySum Files (*.psysum)')
            if file_path:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    rowListString = lines[0].strip()
                    columnListString = lines[1].strip()
                    dataListString = lines[2].strip()
                    filterListString = lines[3].strip()

                    rowList = parseStringToList(rowListString)
                    columnList = parseStringToList(columnListString)
                    dataList = parseStringToList(dataListString)
                    filterList = parseStringToList(filterListString)

                    setListWidgetData(self.rows_list, rowList)
                    setListWidgetData(self.columns_list, columnList)
                    setListWidgetData(self.data_list, dataList)
                    setListWidgetData(self.filter_list, filterList)

        except Exception as e:
            self.printLogInfo(f"Error in reading file:{e}", 3)
            return None

    def computationVariable(self):
        self.computationVariableGui.updateData(self.data)
        self.computationVariableGui.show()

    def getFilterList(self):
        items = []

        for index in range(self.filter_list.count()):
            item = self.filter_list.item(index)
            items.append(item.text())
        return items

    def getFilteredDataFrame(self):
        df = self.data

        items = self.getFilterList()

        if len(items) != 0:
            rowList = getListWidgetData(self.rows_list)
            columnList = getListWidgetData(self.columns_list)

            df = StatisticTool.filterData(rowList, columnList, self.data, items)
        return df

    # 保存预设的.psydata文件
    def saveFilteredData(self):
        df = self.getFilteredDataFrame()

        try:
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'psyData Files (*.psydata)')
            if file_path:
                df.to_csv(file_path, sep='|', quoting=csv.QUOTE_NONNUMERIC, index=False, header=True)
                # filterDataOnly(self, row_vars, col_vars, ruleList):
                PsyDataFunc.genScript(
                    f"filteredDataFrame = aggData.filterData(rowVariables, colVariables, ruleList, cdfPoolingOmegas)")
                PsyDataFunc.genScript(
                    f"filteredDataFrame.to_csv('{file_path}', sep='|', quoting=csv.QUOTE_NONNUMERIC, index=False, header=True)")
        except Exception as e:
            self.printLogInfo(f"Error in saving filtered data:{e}", 3)
            return None

    def savePsyData(self):
        if self.data is None:
            MessageBox.information(self, 'Warning', "No data exist, please load data first.")
            return False
        try:
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'psyData Files (*.psydata)')
            if file_path:
                # 将数组数据保存到文件中
                self.data.to_csv(file_path, sep='|', quoting=csv.QUOTE_NONNUMERIC, index=False, header=True)

                PsyDataFunc.genScript(
                    f"aggData.data.to_csv('{file_path}', sep='|', quoting=csv.QUOTE_NONNUMERIC, index=False, header=True)")
        except Exception as e:
            self.printLogInfo(f"Error in saving file:{e}", 3)
            return None

    # 保存预设文件
    def saveFilterEvent(self):
        columnList = getListWidgetData(self.columns_list)
        rowList = getListWidgetData(self.rows_list)
        dataList = getListWidgetData(self.data_list)
        filterList = getListWidgetData(self.filter_list)

        if self.data is None:
            MessageBox.information(self, 'Warning', "No data exist, please load data first.")
            return False
        try:
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'PsySum Files (*.psysum)')
            if file_path:
                # 将数组数据保存到文件中
                with open(file_path, 'w') as file:
                    file.write(f'rowList: {rowList}\n')
                    file.write(f'columnList: {columnList}\n')
                    file.write(f'dataList: {dataList}\n')
                    file.write(f'filterList: {filterList}\n')
        except Exception as e:
            MessageBox.warning(self, "Save file error", f"{e}")
            return None

    # @staticmethod
    def printLogInfo(self, infoText, infoType: int = 0):
        PsyDataFunc.printOut(infoText, infoType)

    def handleThreadSignal(self, infoType: int, infoText: str):
        self.printLogInfo(infoText, infoType)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PsyData()
    window.show()
    app.exec()
