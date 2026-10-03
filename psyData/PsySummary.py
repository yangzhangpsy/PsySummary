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

from PyQt5.QtCore import QEvent, QSettings, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, \
    QHBoxLayout, QGridLayout, QLabel, QVBoxLayout, QPushButton, QMenu, QWidget, QMainWindow, QMessageBox, QAction, \
    QActionGroup, QDockWidget, QStyle, QToolButton
from PyQt5.QtGui import QKeySequence, QPainter, QPalette
from app.psyDataFunc import PsyDataFunc as Func
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
from app.lib.modelFitOverlay import ModelFitOverlay
from app.lib.pivotedDataWidget import MODEL_FIT_METHODS, PivotedDataWidget
from app.psyDataFunc import PsyDataFunc
from app.psyDataInfo import PsyDataInfo
from app.lib.scriptDock import ScriptDock
from app.tool import StatisticTool, FlashMessageBox
from app.variableCompute import VariableCompute
from app.output import Output
from app.cognitiveModelSpec import (
    COGNITIVE_MODEL_NAMES, split_target, validate_model_data,
)


PSYSUMMARY_SETUP_DIRECTORY_KEY = 'psysummary_setup_directory'


class ResultsToggleButton(QToolButton):
    """Draw a clickable Results chevron without a native button frame."""

    def paintEvent(self, _event):
        """Paint only the current chevron text using the appropriate palette color."""
        painter = QPainter(self)
        color_group = QPalette.Active if self.isEnabled() else QPalette.Disabled
        painter.setPen(self.palette().color(color_group, QPalette.ButtonText))
        font = painter.font()
        font.setPointSize(16)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())


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
        self._mat_import_results = {}
        self._mat_import_job_id = 0
        self._mat_import_append = False
        QApplication.instance().aboutToQuit.connect(self._shutdownDataImports)
        self.pivotTableWindow = None
        self._pending_result_widget = None
        self._model_fit_message_box = None
        self._results_dock_width = 400
        self._results_collapsed_width = None
        self._results_geometry_expanded = False
        self._results_transition = False
        self._results_window_resizing = False
        self._closing = False
        self.filterWindow = None
        self.distributionPreviewWindow = None
        self.tableFrame = None
        self.variablesNameList = None
        self.import_file = None
        self.data = pd.DataFrame()
        self.dataReadStart = False
        self.model_fit_running = False
        self._closing_after_model_cancel = False
        self._closing_after_variable_compute = False
        self._menu_enabled_before_model_fit = True
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
        self.setWindowIcon(Func.getImageObject("icon.png", type=1))
        # set the central widget
        self.central_widget = QWidget()
        self.computationVariableGui = VariableCompute(self.data, self)

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

        self.results_dock = QDockWidget('Aggregation Results', self)
        self.results_dock.setObjectName('aggregationResultsDock')
        self.results_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.results_dock.setFeatures(QDockWidget.DockWidgetClosable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.results_dock)
        self.setCorner(Qt.TopRightCorner, Qt.RightDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self.results_dock.hide()
        self.results_dock.visibilityChanged.connect(
            self._aggregationResultsVisibilityChanged)
        self.results_action = QAction('&Aggregation Results', self)
        self.results_action.setCheckable(True)
        self.results_action.setEnabled(False)
        self.results_action.triggered.connect(self._setAggregationResultsVisible)
        view_menu.addAction(self.results_action)

        self.script_action = QAction("&Script", self)
        self.script_action.setData("script")

        if self.is_windows:
            checked_icon = Func.getImageObject("checked", 1)
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
            self.output = Output(
                True, export_default_filename='psySummaryLog.txt')
            self.addDockWidget(Qt.BottomDockWidgetArea, self.output)
            self.output.realVisibleChanged.connect(self.setActionIcon)

            self.output_action = QAction("&Output", self)
            self.output_action.setData("output")

            if self.is_windows:
                checked_icon = Func.getImageObject("checked", 1)
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
        self.results_toggle_button = ResultsToggleButton()
        self.results_toggle_button.setText('》')
        self.results_toggle_button.setFixedSize(22, 24)
        self.results_toggle_button.setCursor(Qt.PointingHandCursor)
        self.results_toggle_button.setFocusPolicy(Qt.NoFocus)
        self.results_toggle_button.setEnabled(False)
        self.results_toggle_button.hide()
        self.results_toggle_button.setToolTip(
            'Run an analysis to create Aggregation Results.')

        self.filter_button.clicked.connect(self.defineFilterEvent)
        self.distribution_preview_button.clicked.connect(self.showDistributionPreview)
        self.export_filtered_button.clicked.connect(self.saveFilteredData)
        self.save_filter_button.clicked.connect(self.saveFilterEvent)
        self.load_filter_button.clicked.connect(self.loadFilterEvent)
        self.run_button.clicked.connect(self.runSummary)
        self.close_button.clicked.connect(self.clickCloseEvent)
        self.results_toggle_button.clicked.connect(self._toggleAggregationResults)

        self.computationVariableGui.transformFinished.connect(self.transformVariable)
        self.computationVariableGui.computationRunningChanged.connect(self._variableComputationStateChanged)

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
        instruction_font = self.instruct_lab.font()
        instruction_font.setPointSizeF(instruction_font.pointSizeF() + 2.0)
        self.instruct_lab.setFont(instruction_font)

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
        self.variables_header = QWidget()
        variables_header_layout = QHBoxLayout(self.variables_header)
        variables_header_layout.setContentsMargins(0, 0, 0, 0)
        variables_header_layout.setSpacing(4)
        variables_header_layout.addWidget(QLabel("Variables:"))
        variables_header_layout.addStretch(1)
        variables_header_layout.addWidget(self.results_toggle_button)
        main_layout.addWidget(self.variables_header, 0, 4, 1, 1)
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
        self.model_fit_overlay = ModelFitOverlay(self)
        self.model_fit_overlay.syncGeometry()

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
        if self._variableCalculationBusy('loading data'):
            return False
        if self.model_fit_running:
            self._showModelFitBusyMessage()
            return False
        files = self.normalizeRecentFilePaths(files)
        if not files:
            return
        try:
            self.files = files
            _, file_extension = os.path.splitext(files[0])
            file_extension = file_extension.lower()

            if file_extension in {'.txt', '.dat', '.csv'}:
                self.import_file = DecodingFiles(files, add_source_file=True, parent=self)
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
        """Start an owned, uniquely identified import job in the current batch."""
        if not self.readMatThreads:
            self._mat_import_results = {}
            self._mat_import_append = appendDataModel
        self._mat_import_job_id += 1
        worker = ImportMatThread(fileList, matType, appendDataModel, self)
        worker.job_id = self._mat_import_job_id
        self.readMatThreads[worker.job_id] = worker
        worker.readStatus.connect(self.handleThreadSignal)
        worker.dataReady.connect(self.handleReadDataFinished)
        worker.finished.connect(self._finishMatImport)
        worker.start()

    def handleReadDataFinished(self, data: pd.DataFrame, fileType: int, fileList: list, appendDataModel: bool):
        """Stage successful data; native thread completion owns cleanup and commit."""
        worker = self.sender()
        if worker is not None and data.size > 0:
            self._mat_import_results[worker.job_id] = (data, fileType, list(fileList))

    def _finishMatImport(self):
        """Release every completed job, including failures, and commit a finished batch."""
        worker = self.sender()
        if worker is None:
            return
        self.readMatThreads.pop(worker.job_id, None)
        worker.deleteLater()
        if self.readMatThreads:
            return
        results = [self._mat_import_results[key] for key in sorted(self._mat_import_results)]
        self._mat_import_results = {}
        if not results or self._closing:
            return
        previous_data = self.data
        previous_script = self.script_dock.text_edit.toPlainText()
        try:
            frames = ([self.data] if self._mat_import_append else []) + [row[0] for row in results]
            combined = pd.concat(frames, ignore_index=True)
            # Record the same stable order used for the GUI, not thread finish order.
            for index, (_data, file_type, files) in enumerate(results):
                PsyDataFunc.genScript(PsyDataFunc.list2Script(files, 'fileList'))
                method = 'readMatlabFiles' if file_type == 1 else 'readMatlabFiles73'
                append = self._mat_import_append or index > 0
                PsyDataFunc.genScript(f"aggData.{method}(fileList, {append})")
            self.data = combined
            self.clearAllListAndSetData()
            self.updateRecentFiles([path for row in results for path in row[2]])
        except Exception as error:
            self.data = previous_data
            self.script_dock.text_edit.setPlainText(previous_script)
            self.printLogInfo(f'MAT import failed: {error}', 2)

    def _dataImportBusy(self):
        """Keep source-dependent actions blocked until import cleanup has completed."""
        if self.readMatThreads or getattr(self.import_file, 'final_read_thread', None) is not None:
            MessageBox.information(self, 'Data Import in Progress', 'Please wait for data import to finish.')
            return True
        return False

    def _shutdownDataImports(self):
        """Wait for owned import threads only during final application shutdown."""
        self._closing = True
        workers = list(self.readMatThreads.values())
        for worker in workers:
            worker.requestInterruption()
        for worker in workers:
            worker.wait()
        delimited = getattr(self.import_file, 'final_read_thread', None)
        if delimited is not None:
            delimited.requestInterruption()
            delimited.quit()
            delimited.wait()

    def decodingFileOKPressedEvent(self):
        self.import_file.readFinalData()

    def decodingFileDataReady(self, data):
        """Apply imported data after background file reading."""
        self.data = data
        for file_path in self.import_file.files:
            self.printLogInfo(f"Reading file: {file_path}", 0)
        self.updateRecentFiles(self.import_file.files)
        self.import_file.acceptEvent()

        text_format, delimiter = self.import_file.getFormatAndDelimiter()
        PsyDataFunc.genScript(PsyDataFunc.list2Script(self.import_file.files, 'fileList'))
        PsyDataFunc.genScript(
            f"aggData.readDatFiles(fileList, {self.import_file.getContainHeadStatus()}, "
            f"{text_format!r}, {delimiter!r})")
        self.clearAllListAndSetData()

    def clearAllListAndSetData(self):
        if self.data is None:
            return

        self.dataReadStart = False
        self.clearAllList()

        if not validateName(self.data) or not self.data.columns.is_unique:
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
        # The exported importer reads the original headers; replay the GUI's exact rename.
        PsyDataFunc.genScript(f"aggData.data.columns = {cleaned_columns!r}")
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
        if self._dataImportBusy():
            return False
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
        if self._variableCalculationBusy('defining filters'):
            return False
        if self.data is None or self.data.size == 0:
            MessageBox.information(self, 'Warning', "No data exist, please load data first.")
            return False
        # try:
        self.filterWindow = FilterWindow(self.data, self.filter_list)
        self.filterWindow.previewRequested.connect(self.showDistributionPreview)
        self.filterWindow.show()

    def showDistributionPreview(self):
        """Open a before/after visualization using the active PsySummary filters."""
        if self._variableCalculationBusy('previewing filter effects'):
            return False
        if self.model_fit_running:
            self._showModelFitBusyMessage('previewing filter effects')
            return False
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

            # Filtering needs only rule/group columns, not unrelated experiment data.
            from app.dataPreparation import split_filter_rule
            filter_columns = list(dict.fromkeys(
                row_variables + column_variables + [split_filter_rule(rule)[0] for rule in rules]))
            preview_source = self.data.loc[:, filter_columns].copy(deep=False)
            preview_source[marker] = np.arange(len(preview_source), dtype=int)
            retained_data = StatisticTool.filterData(
                row_variables, column_variables, preview_source, rules,
                record_script=False)
            retained_mask = np.zeros(len(self.data), dtype=bool)
            retained_mask[retained_data[marker].to_numpy(dtype=np.intp)] = True
            del retained_data, preview_source

            self.distributionPreviewWindow = DistributionPreviewDialog(
                self.data,
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
        if self._variableCalculationBusy('running an analysis'):
            return None
        if self.model_fit_running:
            self._showModelFitBusyMessage()
            return None
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
            contains_model_fit = any(
                split_target(target)[1] in MODEL_FIT_METHODS
                for target in dataList)

            model_targets = [split_target(target) for target in dataList
                             if split_target(target)[1] in COGNITIVE_MODEL_NAMES]
            if model_targets:
                filtered = self.getFilteredDataFrame()
                problems = []
                for variable, model, specification in model_targets:
                    try:
                        if not specification:
                            raise ValueError('Open Model Settings to configure this model.')
                        validate_model_data(specification, filtered, rowList + columnList)
                    except (TypeError, ValueError, KeyError) as error:
                        problems.append(f'{variable}@{model}:\n{error}')
                if problems:
                    MessageBox.warning(self, 'Invalid Model Settings', '\n\n'.join(problems))
                    return None

            if contains_model_fit:
                self.model_fit_running = True
                self._startModelFitOverlay()
            result_widget = PivotedDataWidget(
                self.data, rowList, columnList, dataList, items)
            if contains_model_fit:
                self._pending_result_widget = result_widget
                result_widget.analysisFinished.connect(
                    lambda widget=result_widget: self._modelFitFinished(widget))
                result_widget.analysisFailed.connect(
                    lambda message, widget=result_widget:
                    self._modelFitFailed(widget, message))
                if hasattr(result_widget, 'analysisCancelled'):
                    result_widget.analysisCancelled.connect(
                        lambda widget=result_widget: self._modelFitCancelled(widget))
                if hasattr(result_widget, 'analysisProgress'):
                    result_widget.analysisProgress.connect(self._modelFitProgress)
            else:
                self._showAggregationResults(result_widget)
        except Exception as e:
            self.model_fit_running = False
            self._pending_result_widget = None
            self._stopModelFitOverlay()
            MessageBox.information(self, 'Warning', f"{e}")
            traceback.print_exc()
            return None

    def _showModelFitBusyMessage(self, action='starting another analysis'):
        """Show the active-fit notice at a stable two-line width."""
        dialog = MessageBox(self)
        dialog.setIcon(QMessageBox.Information)
        dialog.setWindowTitle('Model Fitting in Progress')
        dialog.setText(
            'A model is currently being fitted in the background.\n'
            f'Please wait for it to finish before {action}.')
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.setMinimumWidth(540)
        dialog.setStyleSheet('QLabel { min-width: 500px; }')
        self._model_fit_message_box = dialog
        dialog.exec_()

    def _startModelFitOverlay(self):
        """Cover PsySummary with a localized spinner while a model queue runs."""
        self._menu_enabled_before_model_fit = self.menuBar().isEnabled()
        self.menuBar().setEnabled(False)
        self.model_fit_overlay.start()

    def _modelFitProgress(self, current, total, label):
        """Update the overlay only when the fitting queue starts another model."""
        if self.model_fit_running:
            self.model_fit_overlay.setProgress(current, total, label)

    def _stopModelFitOverlay(self):
        """Restore normal interaction after a model queue reaches a terminal state."""
        if (not hasattr(self, 'model_fit_overlay')
                or (self.model_fit_overlay.isHidden()
                    and not self.model_fit_overlay.animation_timer.isActive())):
            return
        self.model_fit_overlay.stop()
        self.menuBar().setEnabled(self._menu_enabled_before_model_fit)

    def _modelFitFinished(self, result_widget):
        """Reveal a completed background result and clear the run guard."""
        if self._pending_result_widget is not result_widget:
            return
        self.model_fit_running = False
        self._pending_result_widget = None
        self._stopModelFitOverlay()
        self._showAggregationResults(result_widget)

    def _modelFitFailed(self, result_widget, message):
        """Discard a failed pending result while preserving any previous result."""
        if self._pending_result_widget is not result_widget:
            return
        self.model_fit_running = False
        self._pending_result_widget = None
        self._stopModelFitOverlay()
        result_widget.deleteLater()
        MessageBox.warning(self, 'Model Fitting Error', message)

    def _modelFitCancelled(self, result_widget):
        """Discard a cancelled pending fit and finish a requested window close."""
        if self._pending_result_widget is not result_widget:
            return
        self.model_fit_running = False
        self._pending_result_widget = None
        self._stopModelFitOverlay()
        PsyDataFunc.printOut('Model fitting cancelled.', 4, False)
        result_widget.deleteLater()
        if self._closing_after_model_cancel:
            self._closing_after_model_cancel = False
            QTimer.singleShot(0, self.close)

    def _showAggregationResults(self, result_widget):
        """Replace and reveal the fixed right-side aggregation-results dock."""
        previous_widget = self.results_dock.widget()
        if previous_widget is not None and previous_widget is not result_widget:
            diagnostics = getattr(previous_widget, 'fit_diagnostics_dialog', None)
            if diagnostics is not None:
                diagnostics.close()
            previous_widget.setParent(None)
            previous_widget.deleteLater()

        self.pivotTableWindow = result_widget
        self.results_dock.setWidget(result_widget)
        self.results_action.setEnabled(True)
        self.results_toggle_button.setEnabled(True)
        self._setAggregationResultsVisible(True)

    def _toggleAggregationResults(self):
        """Toggle the aggregation-results drawer from the central boundary button."""
        self._setAggregationResultsVisible(self.results_dock.isHidden())

    def _setAggregationResultsVisible(self, visible):
        """Show or hide results while preserving the Data Summary layout width."""
        if visible and self.pivotTableWindow is None:
            self._syncAggregationResultsControls()
            return
        if bool(visible) == (not self.results_dock.isHidden()):
            self._syncAggregationResultsControls()
            return

        if visible:
            self._resizeWindowForAggregationResults(True)
            self._results_transition = True
            try:
                self.results_dock.show()
                self.results_dock.raise_()
                self.resizeDocks(
                    [self.results_dock], [self._results_dock_width], Qt.Horizontal)
            finally:
                self._results_transition = False
        else:
            self._rememberAggregationResultsWidth()
            self._results_transition = True
            try:
                self.results_dock.hide()
            finally:
                self._results_transition = False
            self._deferAggregationResultsCollapse()
        self._syncAggregationResultsControls()

    def _aggregationResultsVisibilityChanged(self, _visible):
        """Handle title-bar closes and keep all Results visibility controls synchronized."""
        if self._closing or self._results_transition:
            return
        explicitly_visible = not self.results_dock.isHidden()
        if explicitly_visible:
            self._resizeWindowForAggregationResults(True)
            self.resizeDocks(
                [self.results_dock], [self._results_dock_width], Qt.Horizontal)
        else:
            self._rememberAggregationResultsWidth()
            self._deferAggregationResultsCollapse()
        self._syncAggregationResultsControls()

    def _deferAggregationResultsCollapse(self):
        """Shrink after Qt removes the hidden dock from the main-window layout."""
        QTimer.singleShot(0, self._completeAggregationResultsCollapse)

    def _completeAggregationResultsCollapse(self):
        """Finish a pending collapse only if Results is still explicitly hidden."""
        if self._closing or not self.results_dock.isHidden():
            return
        self.layout().activate()
        self._resizeWindowForAggregationResults(False)
        self._syncAggregationResultsControls()

    def _rememberAggregationResultsWidth(self):
        """Remember the user's latest dock width for the next expansion."""
        if self.results_dock.width() > 0:
            self._results_dock_width = self.results_dock.width()

    def _aggregationResultsExtent(self):
        """Return the horizontal space occupied by the Results dock and its separator."""
        separator = self.style().pixelMetric(QStyle.PM_DockWidgetSeparatorExtent)
        return max(1, self._results_dock_width) + max(0, separator)

    def _resizeWindowForAggregationResults(self, expanded):
        """Pair drawer visibility with an outer-window resize in normal window mode."""
        if self.isMaximized() or self.isFullScreen():
            return
        if expanded == self._results_geometry_expanded:
            return

        extent = self._aggregationResultsExtent()
        available = QApplication.desktop().availableGeometry(self)
        if expanded:
            self._results_collapsed_width = self.width()
            target_width = min(self.width() + extent, available.width())
            target_x = min(self.x(), available.right() - target_width + 1)
            self._results_window_resizing = True
            try:
                self.resize(target_width, self.height())
                self.move(max(available.left(), target_x), self.y())
            finally:
                self._results_window_resizing = False
        else:
            target_width = (
                self._results_collapsed_width
                if self._results_collapsed_width is not None
                else self.width() - extent
            )
            self._results_window_resizing = True
            try:
                self.resize(max(self.minimumWidth(), target_width), self.height())
            finally:
                self._results_window_resizing = False
        self._results_geometry_expanded = expanded

    def resizeEvent(self, event):
        """Track user resizing so the collapsed Data Summary width remains current."""
        super().resizeEvent(event)
        if hasattr(self, 'model_fit_overlay'):
            self.model_fit_overlay.syncGeometry()
        if (self._closing or self._results_transition or self._results_window_resizing
                or self.isMaximized() or self.isFullScreen()
                or not hasattr(self, 'results_dock')):
            return
        if self.pivotTableWindow is None or self.results_dock.isHidden():
            self._results_collapsed_width = event.size().width()
        elif self._results_geometry_expanded:
            self._results_collapsed_width = max(
                self.minimumWidth(),
                event.size().width() - self._aggregationResultsExtent(),
            )

    def _syncAggregationResultsControls(self):
        """Update arrow direction and View-menu state without recursive signals."""
        available = self.pivotTableWindow is not None
        expanded = available and not self.results_dock.isHidden()
        self.results_toggle_button.setVisible(available)
        self.results_toggle_button.setEnabled(available)
        self.results_toggle_button.setText('《' if expanded else '》')
        self.results_toggle_button.setToolTip(
            'Collapse Aggregation Results.' if expanded else
            'Expand Aggregation Results.' if available else
            'Run an analysis to create Aggregation Results.')
        self.results_action.setEnabled(available)
        blocked = self.results_action.blockSignals(True)
        self.results_action.setChecked(expanded)
        self.results_action.blockSignals(blocked)

    def changeEvent(self, event):
        """Reconcile paired Results geometry after leaving maximized or full-screen mode."""
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            QTimer.singleShot(0, self._reconcileAggregationResultsGeometry)

    def _reconcileAggregationResultsGeometry(self):
        """Apply a deferred drawer resize after the window returns to normal mode."""
        if self._closing or self.isMaximized() or self.isFullScreen():
            return
        expanded = self.pivotTableWindow is not None and not self.results_dock.isHidden()
        self._resizeWindowForAggregationResults(expanded)
        self._syncAggregationResultsControls()

    def closeEvent(self, event):
        if self._dataImportBusy():
            event.ignore()
            return
        if self.computationVariableGui.computation_running:
            if self.computationVariableGui.requestCancelAndClose(close_parent=True):
                self._closing_after_variable_compute = True
            event.ignore()
            return
        if self.model_fit_running:
            if self._closing_after_model_cancel:
                event.ignore()
                return
            if not self._confirmStopModelFit():
                event.ignore()
                return
            if self.model_fit_running:
                pending_result = self._pending_result_widget
                if pending_result is not None:
                    self._closing_after_model_cancel = True
                    pending_result.cancelAnalysis()
                    event.ignore()
                    return
                self.model_fit_running = False
        self._closing = True
        if self.pivotTableWindow:
            diagnostics = getattr(self.pivotTableWindow, 'fit_diagnostics_dialog', None)
            if diagnostics is not None:
                diagnostics.close()
        if self.filterWindow:
            self.filterWindow.close()
        if self.computationVariableGui:
            self.computationVariableGui.close()
        if self.distributionPreviewWindow:
            self.distributionPreviewWindow.close()
        super().closeEvent(event)

    def _confirmStopModelFit(self):
        """Ask whether an active background fit should be cancelled before closing."""
        dialog = MessageBox(self)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setWindowTitle('Stop Model Fitting?')
        dialog.setText(
            'Model fitting is in progress.\n'
            'Stop fitting and close PsySummary?')
        stop_button = dialog.addButton(
            'Stop and Close', QMessageBox.DestructiveRole)
        continue_button = dialog.addButton(
            'Keep Fitting', QMessageBox.RejectRole)
        dialog.setDefaultButton(continue_button)
        dialog.setEscapeButton(continue_button)
        dialog.setStyleSheet(
            'QLabel#qt_msgbox_label { min-width: 320px; max-width: 380px; }')
        self._model_fit_close_message_box = dialog
        dialog.exec_()
        return dialog.clickedButton() is stop_button

    def transformVariable(self, newVariableName):
        self.variablesNameList.append(newVariableName)

        self.variables_list.addItem(newVariableName)
        self.variables_list.sortItems(Qt.AscendingOrder)

    def _variableCalculationBusy(self, action):
        """Prevent competing source-data operations during import or calculation."""
        if self._dataImportBusy():
            return True
        if not self.computationVariableGui.computation_running:
            return False
        MessageBox.information(
            self.computationVariableGui, 'Variable Calculation in Progress',
            f'A variable is being calculated in the background.\nPlease wait before {action}.')
        return True

    def _variableComputationStateChanged(self, running):
        """Finish a deferred host-window close only after worker cleanup."""
        if not running and self._closing_after_variable_compute:
            QTimer.singleShot(0, self.close)

    def clickCloseEvent(self):
        self.close()

    def _setupDirectory(self):
        """Return the last valid PsySummary Setup directory or a stable fallback."""
        saved_directory = Settings(Info.ConfigFile, QSettings.IniFormat).value(
            PSYSUMMARY_SETUP_DIRECTORY_KEY, '')
        if isinstance(saved_directory, str) and saved_directory:
            saved_directory = os.path.abspath(os.path.expanduser(saved_directory))
            if os.path.isdir(saved_directory):
                return saved_directory
        if Info.FILE_DIRECTORY and os.path.isdir(Info.FILE_DIRECTORY):
            return Info.FILE_DIRECTORY
        return Info.UserPath

    @staticmethod
    def _rememberSetupDirectory(file_path):
        """Persist the directory containing a successfully saved or loaded Setup file."""
        directory = os.path.dirname(os.path.abspath(file_path))
        settings = Settings(Info.ConfigFile, QSettings.IniFormat)
        settings.setValue(PSYSUMMARY_SETUP_DIRECTORY_KEY, directory)
        settings.sync()

    def loadFilterEvent(self):
        if self._dataImportBusy():
            return False
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 'Load Setup', self._setupDirectory(),
                'PsySum Files (*.psysum)')
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as file:
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
                    self._rememberSetupDirectory(file_path)

        except Exception as e:
            self.printLogInfo(f"Error in reading file:{e}", 3)
            return None

    def computationVariable(self):
        if self.model_fit_running:
            self._showModelFitBusyMessage('computing a variable')
            return
        if self._variableCalculationBusy('starting another calculation'):
            return
        if self.readMatThreads:
            MessageBox.information(self, 'Data Import in Progress', 'Please wait for data import to finish.')
            return
        self.computationVariableGui.updateData(self.data)
        self.computationVariableGui.show()

    def getFilterList(self):
        items = []

        for index in range(self.filter_list.count()):
            item = self.filter_list.item(index)
            items.append(item.text())
        return items

    def getFilteredDataFrame(self, record_script=False):
        """Return retained rows, optionally recording reusable filter parameters."""
        items = self.getFilterList()

        if items:
            rowList = getListWidgetData(self.rows_list)
            columnList = getListWidgetData(self.columns_list)
            # filterData owns the one defensive working copy needed to keep self.data unchanged.
            return StatisticTool.filterData(
                rowList, columnList, self.data, items,
                record_script=record_script)
        if record_script:
            PsyDataFunc.genScript('cdfPoolingOmegas = []')
        # Export and the read-only data viewer do not mutate the loaded DataFrame.
        return self.data

    # 保存预设的.psydata文件
    def saveFilteredData(self):
        if self._variableCalculationBusy('exporting filtered data'):
            return False
        filtered_copy = self.getFilteredDataFrame(record_script=True)

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

                PsyDataFunc.genScript(
                    f"aggData.data.to_csv({file_path!r}, sep='|', quoting=csv.QUOTE_NONNUMERIC, "
                    f"index=False, header=True)")
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
            file_path, _ = QFileDialog.getSaveFileName(
                self, 'Save Setup', self._setupDirectory(),
                'PsySum Files (*.psysum)')
            if file_path:
                if os.path.splitext(file_path)[1].lower() != '.psysum':
                    file_path += '.psysum'
                # 将数组数据保存到文件中
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(f'rowList: {rowList}\n')
                    file.write(f'columnList: {columnList}\n')
                    file.write(f'dataList: {dataList}\n')
                    file.write(f'filterList: {filterList}\n')
                    file.write(f'modelSpecifications: {modelSpecifications!r}\n')
                self._rememberSetupDirectory(file_path)
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
