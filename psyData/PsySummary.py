# -*- coding: utf-8 -*-
import csv
import ast
import sys
import traceback
import re
import os

import numpy as np
import pandas as pd
from scipy.io import matlab

from PyQt5.QtCore import Qt, QEvent, QSettings
from PyQt5.QtWidgets import QApplication, QFileDialog, \
    QHBoxLayout, QGridLayout, QLabel, QVBoxLayout, QPushButton, QMenu, QWidget, QMainWindow, QMessageBox, QAction, \
    QActionGroup, QDockWidget
from PyQt5.QtGui import QKeySequence
from app.func import Func
from app.info import Info
from app.lib import MessageBox, Settings
from app.lib.decoding_files import DecodingFiles
from app.lib.import_mat_thread import ImportMatThread
from app.lib.source_file import addSourceFileColumn
from app.lib.draggablelistwidget import DraggableListWidget, MainFilterListWidget, \
    VariableDraggableListWidget, MODEL_SPEC_ROLE
from app.lib.filterWindow import FilterWindow
from app.lib.dataFrameTableWidget import DataFrameTableWidget
from app.lib.distributionPreview import DistributionPreviewDialog
from app.lib.pivotedDataWidget import PivotedDataWidget
from app.psyDataFunc import PsyDataFunc
from app.psyDataInfo import PsyDataInfo
from app.lib.scriptDock import ScriptDock
from app.tool import StatisticTool, FlashMessageBox
from app.variableCompute import VariableCompute
from app.output import Output


def setListWidgetData(widget, items):
    if hasattr(widget, 'contentList'):
        widget.contentList = list(items)
        # False to keep the content list untouched
        widget.clear(False)
    else:
        widget.clear()

    widget.addItems(items)


def getListWidgetData(widget):
    """Returns a list of all item texts from the given QListWidget."""
    return [widget.item(i).text() for i in range(widget.count())]


def getDataListEntries(widget):
    """Return Data targets while preserving structured cognitive-model settings."""
    entries = []
    for index in range(widget.count()):
        item = widget.item(index)
        specification = item.data(MODEL_SPEC_ROLE)
        entries.append({'model_specification': specification} if specification else item.text())
    return entries


def parseStringToList(string):
    start_index = string.find(": ")

    if start_index != -1:
        list_string = string[start_index + 2:]  # 提取包含列表的部分
        # 使用 eval() 函数解析字符串并转换为列表
        filter_list = ast.literal_eval(list_string)
        # 输出转换后的列表
        return filter_list
    else:
        return None


def fixColumnName(name, shouldStartWithLetter: bool = False):
    # 去掉不符合规则的字符，仅保留合法字符
    fixed_name = ''.join(re.findall(r'[a-zA-Z0-9_\-.]', name))
    # 确保列名以字母开头
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
        self.plugin_mode = False
        self.readMatThreads = dict()
        self.pivotTableWindow = None
        self.filterWindow = None
        self.distributionPreviewWindow = None
        self.tableFrame = None
        self.variablesNameList = None
        self.import_file = None
        self.data = pd.DataFrame()
        self.dataReadStart = False
        self.is_windows = Info.OS_TYPE == 0
        self.files = None
        self.lst = [None, ' ']
        self.analysisScript = []

        PsyDataInfo.PsyData = self

        if self.plugin_mode:
            self.resize(980, 700)
        else:
            self.resize(980, 700)

        self.setWindowTitle('Data Summary')
        self.setWindowIcon(Func.getImageObject("common/icon.png", type=1))
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
        self.open_recent_menu = file_menu.addMenu("Open Recent")
        self.open_recent_menu.aboutToShow.connect(self.refreshOpenRecentMenu)
        self.refreshOpenRecentMenu()
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
            checked_icon = Func.getImageObject("menu/checked", 1)
            self.script_action.setIcon(checked_icon)
            self.script_action.setIconVisibleInMenu(True)
        else:
            self.script_action.setCheckable(True)
            self.script_action.setChecked(True)

        self.script_action.triggered.connect(self.setDockView)

        """
        # output dock and action
        """
        if not self.plugin_mode:
            self.output = Output(True)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.output)
            self.output.realVisibleChanged.connect(self.setActionIcon)

            self.output_action = QAction("&Output", self)
            self.output_action.setData("output")

            if self.is_windows:
                checked_icon = Func.getImageObject("menu/checked", 1)
                self.output_action.setIcon(checked_icon)
                # self.output_action.setCheckable(True)
                # self.output_action.setChecked(True)
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
        self.filter_button.setToolTip(
            'Create or edit the filtering rules used for preview, export, and analysis.')
        self.distribution_preview_button = QPushButton('Preview Filter Effects')
        self.distribution_preview_button.setToolTip(
            'Compare distributions before and after applying the current filters.')
        self.export_filtered_button = QPushButton('Export Filtered Data')
        self.export_filtered_button.setToolTip(
            'Save retained rows without changing the currently loaded data.')
        self.save_filter_button = QPushButton('Save Setup')
        self.save_filter_button.setToolTip(
            'Save the current Rows, Columns, Data, and Filters configuration.')
        self.load_filter_button = QPushButton('Load Setup')
        self.load_filter_button.setToolTip(
            'Load a previously saved PsySummary configuration.')
        # A doubled ampersand renders one literal ampersand instead of a Qt mnemonic.
        self.run_button = QPushButton('Apply Filters && Run')
        self.run_button.setToolTip(
            'Apply the current filters to the original data and run the selected analysis.')
        self.close_button = QPushButton('Close')

        self.filter_button.clicked.connect(self.defineFilterEvent)
        self.distribution_preview_button.clicked.connect(self.showDistributionPreview)
        self.export_filtered_button.clicked.connect(self.saveFilteredData)
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
        buttons_layout.addSpacing(10)
        buttons_layout.addWidget(self.distribution_preview_button, 1)
        buttons_layout.addWidget(self.export_filtered_button, 1)
        buttons_layout.addWidget(self.run_button, 1)
        buttons_layout.addSpacing(10)
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

        if Info.FILE_DIRECTORY:
            default_dir = Info.FILE_DIRECTORY
        else:
            default_dir = Info.UserPath

        files, _ = QFileDialog.getOpenFileNames(self, "Select File(s)", default_dir,
                                                "Matlab Files (*.mat);;Text Files (*.txt);;Text Files (*.csv);;Dat Files (*.dat);;psyData Files (*.psydata)",
                                                options=options)

        if files:
            self.openDataFiles(files)

    @staticmethod
    def normalizeRecentFilePaths(file_paths):
        """Normalize QSettings and dialog file-path values to a string list."""
        if isinstance(file_paths, (str, bytes)):
            file_paths = [file_paths]
        elif not isinstance(file_paths, list):
            file_paths = list(file_paths) if file_paths else []
        return [
            os.path.abspath(os.path.expanduser(os.fsdecode(file_path)))
            for file_path in file_paths
            if isinstance(file_path, (str, bytes)) and file_path
        ]

    def openDataFiles(self, files):
        """Open one or more supported data files through the normal import workflow."""
        files = self.normalizeRecentFilePaths(files)
        if not files:
            return
        try:
            self.files = files
            _, file_extension = os.path.splitext(files[0])
            file_extension = file_extension.lower()

            if file_extension in {'.txt', '.dat', '.csv'}:
                self.import_file = DecodingFiles(files, add_source_file=True)
                self.import_file.ok_btn.clicked.connect(self.decodingFileOKPressedEvent)
                self.import_file.finalDataReady.connect(self.decodingFileDataReady)
                self.import_file.show()
            elif file_extension == '.mat':
                self.readMatlabFilesMThread(files)
            elif file_extension == '.psydata':
                self.data = readPsyDataFiles(files)
                self.clearAllListAndSetData()
                self.updateRecentFiles(files)
            else:
                raise ValueError(f"Unsupported data file type: {file_extension or 'no extension'}")
        except Exception as e:
            msg_box = FlashMessageBox('Flash Message', str(e))
            msg_box.show()

    def updateRecentFiles(self, file_paths):
        """Add successfully opened data files to PsySummary's recent-file history."""
        new_paths = self.normalizeRecentFilePaths(file_paths)
        settings = Settings(Info.ConfigFile, QSettings.IniFormat)
        recent_paths = self.normalizeRecentFilePaths(
            settings.value('psysummary_recent_files', []))
        for file_path in reversed(new_paths):
            if file_path in recent_paths:
                recent_paths.remove(file_path)
            recent_paths.insert(0, file_path)
        settings.setValue('psysummary_recent_files', recent_paths[:20])
        if hasattr(self, 'open_recent_menu'):
            self.refreshOpenRecentMenu()

    def refreshOpenRecentMenu(self):
        """Rebuild PsySummary's Open Recent submenu from its data-file history."""
        self.open_recent_menu.clear()
        settings = Settings(Info.ConfigFile, QSettings.IniFormat)
        recent_paths = self.normalizeRecentFilePaths(
            settings.value('psysummary_recent_files', []))

        if recent_paths:
            for file_path in recent_paths[:20]:
                action = self.open_recent_menu.addAction(file_path)
                action.setToolTip(file_path)
                action.setEnabled(os.path.isfile(file_path))
                if action.isEnabled():
                    action.triggered.connect(
                        lambda checked=False, recent_file=file_path: self.openDataFiles([recent_file]))
            self.open_recent_menu.addSeparator()
        else:
            empty_action = self.open_recent_menu.addAction('No Recent Files')
            empty_action.setEnabled(False)
            self.open_recent_menu.addSeparator()

        clear_action = self.open_recent_menu.addAction('Clear Items')
        clear_action.setEnabled(bool(recent_paths))
        clear_action.triggered.connect(self.clearRecentFiles)

    def clearRecentFiles(self):
        """Clear PsySummary's recent data-file history."""
        Settings(Info.ConfigFile, QSettings.IniFormat).setValue('psysummary_recent_files', [])
        self.refreshOpenRecentMenu()

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
            self.updateRecentFiles(fileList)

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
        self.import_file.readFinalData()

    def decodingFileDataReady(self, data):
        """Apply imported data after background file reading."""
        self.data = data
        for file_path in self.import_file.files:
            self.printLogInfo(f"Reading file: {file_path}", 0)
        self.updateRecentFiles(self.import_file.files)
        self.import_file.acceptEvent()
        self.clearAllListAndSetData()

        text_format, delimiter = self.import_file.getFormatAndDelimiter()
        PsyDataFunc.genScript(PsyDataFunc.list2Script(self.import_file.files, 'fileList'))
        PsyDataFunc.genScript(f"aggData.readDatFiles(fileList, {self.import_file.getContainHeadStatus()},'{text_format}', '{delimiter}')")

    def clearAllListAndSetData(self):
        if self.data is None:
            return

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
        loaded_files = []
        try:
            for file in fileList:
                df = self.readFile(file)
                if df is not None:
                    all_dfs.append(df)
                    loaded_files.append(file)
            if all_dfs:
                all_dfs, _source_column = addSourceFileColumn(all_dfs, loaded_files)
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
            self.data_list.setModelContext(self.data, self.getFilteredDataFrame)

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
        self.filterWindow.previewRequested.connect(self.showDistributionPreview)
        self.filterWindow.show()

    def showDistributionPreview(self):
        """Open a before/after visualization using the active PsySummary filters."""
        if self.data is None or self.data.empty:
            MessageBox.information(self, 'Warning', 'No data exist, please load data first.')
            return False
        try:
            row_variables = getListWidgetData(self.rows_list)
            column_variables = getListWidgetData(self.columns_list)
            data_items = getListWidgetData(self.data_list)
            target_variables = []
            for item in data_items:
                variable = item.split('@', 1)[0]
                if (variable not in target_variables and variable in self.data.columns
                        and pd.to_numeric(self.data[variable], errors='coerce').notna().any()):
                    target_variables.append(variable)
            if not target_variables:
                MessageBox.information(
                    self,
                    'Warning',
                    'No numeric Data variable is defined. Drag at least one numeric variable into '
                    'the Data area before opening Distribution Preview.')
                return False

            rules = self.getFilterList()
            marker = '__psysummary_preview_row_id__'
            while marker in self.data.columns:
                marker += '_'

            preview_source = self.data.copy()
            preview_source[marker] = np.arange(len(preview_source), dtype=int)
            retained_data = StatisticTool.filterData(
                row_variables, column_variables, preview_source, rules)
            retained_ids = set(retained_data[marker].astype(int).tolist())
            retained_mask = preview_source[marker].isin(retained_ids).to_numpy(dtype=bool)
            preview_source = preview_source.drop(columns=[marker])

            self.distributionPreviewWindow = DistributionPreviewDialog(
                preview_source,
                retained_mask,
                target_variables=target_variables,
                row_facets=row_variables,
                column_facets=column_variables,
                parent=self,
            )
            self.distributionPreviewWindow.show()
            return True
        except Exception as error:
            MessageBox.warning(self, 'Distribution Preview Error', str(error))
            return False

    # 运行分析程序
    def runSummary(self):
        if self.data is None:
            MessageBox.information(self, 'Warning', "No data exist, please load data first.")
            return None

        try:
            rowList = getListWidgetData(self.rows_list)
            columnList = getListWidgetData(self.columns_list)
            dataList = getDataListEntries(self.data_list)

            if not dataList:
                MessageBox.information(
                    self,
                    'Warning',
                    'No Data variable is defined. Drag at least one variable into the Data area '
                    'before running Data Summary.')
                return None

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
        if self.filterWindow:
            self.filterWindow.close()
        if self.computationVariableGui:
            self.computationVariableGui.close()
        if self.distributionPreviewWindow:
            self.distributionPreviewWindow.close()
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
                    modelSpecifications = {}
                    for line in lines[4:]:
                        if line.startswith('modelSpecifications:'):
                            modelSpecifications = parseStringToList(line) or {}
                            break

                    setListWidgetData(self.rows_list, rowList)
                    setListWidgetData(self.columns_list, columnList)
                    setListWidgetData(self.data_list, dataList)
                    self.data_list.restoreModelSpecifications(modelSpecifications)
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
        """Return retained rows without replacing the currently loaded data."""
        items = self.getFilterList()

        if items:
            rowList = getListWidgetData(self.rows_list)
            columnList = getListWidgetData(self.columns_list)
            # filterData owns the one defensive working copy needed to keep self.data unchanged.
            return StatisticTool.filterData(rowList, columnList, self.data, items)
        # Export and the read-only data viewer do not mutate the loaded DataFrame.
        return self.data

    # 保存预设的.psydata文件
    def saveFilteredData(self):
        filtered_copy = self.getFilteredDataFrame()

        try:
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                'Save Filtered Data',
                '',
                'CSV Files (*.csv);;psyData Files (*.psydata)')
            if file_path:
                extension = os.path.splitext(file_path)[1].lower()
                if extension not in {'.csv', '.psydata'}:
                    extension = '.psydata' if 'psyData' in selected_filter else '.csv'
                    file_path += extension

                PsyDataFunc.genScript(f"filteredDataFrame = aggData.filterData(rowVariables, colVariables, ruleList, cdfPoolingOmegas)")
                if extension == '.csv':
                    filtered_copy.to_csv(file_path, index=False, header=True)
                    PsyDataFunc.genScript(
                        f"filteredDataFrame.to_csv({file_path!r}, index=False, header=True)")
                else:
                    filtered_copy.to_csv(
                        file_path, sep='|', quoting=csv.QUOTE_NONNUMERIC, index=False, header=True)
                    PsyDataFunc.genScript(
                        f"filteredDataFrame.to_csv({file_path!r}, sep='|', quoting=csv.QUOTE_NONNUMERIC, "
                        f"index=False, header=True)")
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

                PsyDataFunc.genScript(f"aggData.data.to_csv('{file_path}', sep='|', quoting=csv.QUOTE_NONNUMERIC, index=False, header=True)")
        except Exception as e:
            self.printLogInfo(f"Error in saving file:{e}", 3)
            return None

    # 保存预设文件
    def saveFilterEvent(self):
        columnList = getListWidgetData(self.columns_list)
        rowList = getListWidgetData(self.rows_list)
        dataList = getListWidgetData(self.data_list)
        modelSpecifications = self.data_list.modelSpecifications()
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
                    file.write(f'modelSpecifications: {modelSpecifications!r}\n')
        except Exception as e:
            MessageBox.warning(self, "Save file error", f"{e}")
            return None

    # @staticmethod
    def printLogInfo(self, infoText, infoType: int = 0):
        if self.plugin_mode:
            Func.printOut(infoText, infoType)
        else:
            PsyDataFunc.printOut(infoText, infoType)

    def handleThreadSignal(self, infoType: int, infoText: str):
        self.printLogInfo(infoText, infoType)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PsyData()
    window.show()
    app.exec()
