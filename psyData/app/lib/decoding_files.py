# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QComboBox, QTableWidget, QPushButton, QGridLayout, \
    QLabel, QHBoxLayout, QVBoxLayout, QCheckBox

from app.psyDataFunc import PsyDataFunc as Func
from app.lib import MessageBox, Dialog, VarComboBox
from app.lib.source_file import addSourceFileColumn


def readDelimitedFile(file_path, encoding_code, delimiter, contains_header, num_rows=None):
    """Read one delimited file without accessing GUI state."""
    header_line = 0 if contains_header else None
    data = pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding_code,
        nrows=num_rows,
        header=header_line,
        engine="python",
    )

    if not contains_header:
        data.columns = [f"Column{i + 1}" for i in range(data.shape[1])]

    return data


class FinalDataReadWorker(QObject):
    """Read and combine complete data files outside the GUI thread."""

    dataReady = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, files, encoding_code, delimiter, contains_header, add_source_file=False):
        super().__init__()
        self.files = list(files)
        self.encoding_code = encoding_code
        self.delimiter = delimiter
        self.contains_header = contains_header
        self.add_source_file = add_source_file

    @pyqtSlot()
    def run(self):
        """Read every selected file and emit either the complete data or an error."""
        data_frames = []
        read_errors = []

        for file_path in self.files:
            try:
                data_frames.append(
                    readDelimitedFile(
                        file_path,
                        self.encoding_code,
                        self.delimiter,
                        self.contains_header,
                    )
                )
            except Exception as error:
                read_errors.append(f"{file_path}: {error}")

        if read_errors:
            self.failed.emit("\n".join(read_errors))
        elif not data_frames:
            self.failed.emit("No readable files were selected.")
        else:
            try:
                if self.add_source_file:
                    data_frames, _source_column = addSourceFileColumn(data_frames, self.files)
                if len(data_frames) == 1:
                    self.dataReady.emit(data_frames[0])
                else:
                    self.dataReady.emit(pd.concat(data_frames, ignore_index=True, copy=False))
            except Exception as error:
                self.failed.emit(f"Error while combining decoded files: {error}")

        self.finished.emit()


class DecodingFiles(Dialog):
    finalDataReady = pyqtSignal(object)

    def __init__(self, files=None, add_source_file=False, parent=None):
        super().__init__(parent)
        if files is None:
            files = []

        self.files = files
        self.add_source_file = add_source_file
        self.data = pd.DataFrame()
        self.last_error = ""
        self.final_read_thread = None
        self.final_read_worker = None
        self.pending_final_data = None
        self.pending_final_error = ""

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # 去除问号按钮
        self.setWindowModality(Qt.WindowModal)
        # self.setWindowIcon(Func.getImageObject("icon.png", type=1))

        self.default_properties = {
            "text format": "utf-8",
            "delimiter": "WhiteSpace"
        }

        self.initUI()

    def setFiles(self, files: list):
        self.files = files
        self.setData()

    def initUI(self):
        self.setupUI()
        self.connectSignals()
        self.setData()

    def connectSignals(self):
        self.text_format_comboBox.currentIndexChanged.connect(self.reloadShortData)
        self.delimiter_comboBox.setEditable(True)
        self.delimiter_comboBox.currentTextChanged.connect(self.delimiterChanged)
        # self.ok_btn.clicked.connect(self.acceptEvent)
        self.cancel_btn.clicked.connect(self.rejectEvent)
        self.contains_header_check.clicked.connect(self.reloadShortData)
        self.merge_delimiters_check.clicked.connect(self.reloadShortData)

    def setupUI(self):
        self.text_format_comboBox = QComboBox()
        self.delimiter_comboBox = VarComboBox()
        self.contains_header_check = QCheckBox()
        self.contains_header_check.setChecked(True)
        self.contains_header_check.setText("Contains Header")
        self.merge_delimiters_check = QCheckBox()
        self.merge_delimiters_check.setChecked(True)
        self.merge_delimiters_check.setText("Merge Delimiters")

        self.view_table = QTableWidget()

        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")

        self.setObjectName("Dialog")
        self.resize(900, 282)
        self.setWindowTitle("Data Format")

        self.text_format_comboBox.setObjectName("DataFormatComboBox")
        self.text_format_comboBox.addItems(["utf-8", "gbk"])

        self.delimiter_comboBox.setObjectName("DelimiterComboBox")
        self.delimiter_comboBox.setMinimumWidth(120)
        self.delimiter_comboBox.addItems(["WhiteSpace", ",", "."])

        self.view_table.setObjectName("ViewTable")
        self.view_table.setColumnCount(0)
        self.view_table.setRowCount(0)

        gridLayout = QGridLayout()
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Text Encoding:"))
        format_layout.addWidget(self.text_format_comboBox)
        format_layout.addWidget(QLabel("Text Delimiter:"))
        format_layout.addWidget(self.delimiter_comboBox)
        format_layout.addWidget(self.contains_header_check)
        format_layout.addWidget(self.merge_delimiters_check)
        format_layout.addStretch(1)
        gridLayout.addLayout(format_layout, 0, 0)
        gridLayout.addWidget(self.view_table, 1, 0, 8, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch(3)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(gridLayout)
        main_layout.addStretch(1)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def acceptEvent(self):
        self.default_properties.update({"text format": self.text_format_comboBox.currentText()})
        self.default_properties.update({"delimiter": self.delimiter_comboBox.currentText()})
        self.close()

    def reloadShortData(self):
        self.clearPreview()
        self.setData()

    def clearPreview(self):
        """Clear all rows, columns, and headers from the preview table."""
        self.view_table.clear()
        self.view_table.setRowCount(0)
        self.view_table.setColumnCount(0)

    def delimiterChanged(self, delimiter):
        """Apply a safe default for the selected delimiter and refresh the preview."""
        self.merge_delimiters_check.blockSignals(True)
        self.merge_delimiters_check.setChecked(delimiter == "WhiteSpace")
        self.merge_delimiters_check.blockSignals(False)
        self.reloadShortData()

    def readFinalData(self):
        """Start reading the complete selected files in a worker thread."""
        if self.final_read_thread is not None and self.final_read_thread.isRunning():
            return False

        try:
            encoding_code, delimiter = self.getFormatAndDelimiter()
        except ValueError as error:
            self.showFinalDataError(str(error))
            return False

        self.pending_final_data = None
        self.pending_final_error = ""
        self.setFinalReadBusy(True)
        self.final_read_thread = QThread(self)
        self.final_read_worker = FinalDataReadWorker(
            self.files,
            encoding_code,
            delimiter,
            self.contains_header_check.isChecked(),
            self.add_source_file,
        )
        self.final_read_worker.moveToThread(self.final_read_thread)
        self.final_read_thread.started.connect(self.final_read_worker.run)
        self.final_read_worker.dataReady.connect(self.handleFinalDataReady)
        self.final_read_worker.failed.connect(self.handleFinalDataFailed)
        self.final_read_worker.finished.connect(self.final_read_thread.quit)
        self.final_read_worker.finished.connect(self.final_read_worker.deleteLater)
        self.final_read_thread.finished.connect(self.finalReadThreadFinished)
        self.final_read_thread.finished.connect(self.final_read_thread.deleteLater)
        self.final_read_thread.start()
        return True

    def setFinalReadBusy(self, is_busy):
        """Update controls while a complete file read is running."""
        self.text_format_comboBox.setEnabled(not is_busy)
        self.delimiter_comboBox.setEnabled(not is_busy)
        self.contains_header_check.setEnabled(not is_busy)
        self.merge_delimiters_check.setEnabled(not is_busy)
        self.ok_btn.setEnabled(not is_busy)
        self.cancel_btn.setEnabled(not is_busy)
        self.ok_btn.setText("Loading..." if is_busy else "OK")

    def handleFinalDataReady(self, data):
        """Store complete worker data until its thread has fully stopped."""
        self.pending_final_data = data

    def handleFinalDataFailed(self, error):
        """Store a complete-read failure until its thread has fully stopped."""
        self.pending_final_error = error

    def showFinalDataError(self, error):
        """Display a complete-read failure in the GUI thread."""
        self.reportReadError(error)
        msg = MessageBox(
            QMessageBox.Warning,
            "Warning",
            f"Unable to decode the selected file(s).\n{error}\n\n"
            "Please choose a different encoding or delimiter.",
        )
        msg.exec_()

    def finalReadThreadFinished(self):
        """Release the worker and publish its result after the thread exits."""
        self.final_read_worker = None
        self.final_read_thread = None
        self.setFinalReadBusy(False)

        if self.pending_final_error:
            error = self.pending_final_error
            self.pending_final_error = ""
            self.pending_final_data = None
            self.showFinalDataError(error)
            return

        if self.pending_final_data is not None:
            data = self.pending_final_data
            self.pending_final_data = None
            self.last_error = ""
            self.finalDataReady.emit(data)

    def setData(self):
        if not self.files:
            self.clearPreview()
            self.ok_btn.setEnabled(False)
            return

        try:
            data = self.readMultipleFiles(self.files)
            if data is None:
                self.clearPreview()
                self.ok_btn.setEnabled(False)
                error = self.last_error or "No readable data was found."
                msg = MessageBox(
                    QMessageBox.Warning,
                    "Warning",
                    f"Unable to decode the selected file(s).\n{error}\n\n"
                    "Please choose a different encoding or delimiter.",
                )
                msg.exec_()
                return

            self.setTable(data)
            self.ok_btn.setEnabled(True)
        except Exception as e:
            self.clearPreview()
            self.ok_btn.setEnabled(False)
            msg = MessageBox(QMessageBox.Warning, "Warning",
                             f"Unable to display the decoded data.\n{e}")
            msg.exec_()

    def setTable(self, data):
        rowNum = min(data.shape[0], 10)
        self.view_table.setRowCount(rowNum)
        self.view_table.setColumnCount(data.shape[1])
        self.view_table.setHorizontalHeaderLabels(data.columns)

        for i in range(rowNum):
            for j in range(data.shape[1]):
                item = QTableWidgetItem(str(data.iat[i, j]))
                self.view_table.setItem(i, j, item)

    def getFormatAndDelimiter(self):
        # text_format = self.get("text format", "utf-8")
        # delimiter = self.get("delimiter", "WhiteSpace")

        text_format = self.text_format_comboBox.currentText()
        delimiter = self.delimiter_comboBox.currentText()

        if delimiter == "WhiteSpace":
            delimiter = r"\s"
        elif delimiter:
            delimiter = re.escape(delimiter)
        else:
            raise ValueError("The delimiter cannot be empty.")

        if self.merge_delimiters_check.isChecked():
            delimiter = f"(?:{delimiter})+"

        return text_format, delimiter

    # def readFile(self, file_path, readAllRows=False):
    #     if not file_path:
    #         raise ValueError("File path is empty or not provided.")
    #
    #     try:
    #         df = None
    #
    #         [code, splitCode] = self.getFormatAndDelimiter()
    #
    #         with open(file_path, 'r', encoding=code) as file:
    #             lines = file.readlines()
    #
    #             if readAllRows:
    #                 max_rows = len(lines)
    #             else:
    #                 max_rows = min(10, len(lines))
    #
    #             if self.contains_header_check.isChecked():
    #                 variable_names = re.split(splitCode, lines[0].strip())  # 第一行为变量名
    #                 data = [re.split(splitCode, line.strip()) for line in lines[1:max_rows]]  # 以分隔符分隔的变量值
    #             else:
    #                 data = [re.split(splitCode, line.strip()) for line in lines[0:max_rows]]
    #                 variable_names = [f"Column{i + 1}" for i in range(len(data[0]))]
    #
    #             df = pd.DataFrame(data)
    #
    #             if len(variable_names) >= df.shape[1]:
    #                 df.columns = variable_names[:df.shape[1]]
    #             else:
    #                 df.columns = variable_names + [f'untitled{iVar}' for iVar in
    #                                                range(df.shape[1] - len(variable_names))]
    #             # return df
    #     except (IOError, OSError, FileNotFoundError) as e:
    #         Func.printOut(f"Error in reading file! File probably changed/moved.:{file_path}:{e}", 3)
    #         # return None
    #
    #     except Exception as e:
    #         Func.printOut(f"Error in reading file! File probably has bad format:{file_path}:{e}", 3)
    #         # return None
    #     finally:
    #         return df

    def readFile(self, file_path, readAllRows=False):
        if not file_path:
            raise ValueError("File path is empty or not provided.")

        try:
            # 1. 获取编码和分隔符
            encoding_code, delimiter = self.getFormatAndDelimiter()

            # 2. 确定读取行数：readAllRows 为 False 时仅读取前 10 行
            num_rows = None if readAllRows else 10

            # 3. 直接使用 pandas 读取，性能与健壮性兼得
            # pd.read_csv 在设置了 nrows 后，不会将整个大文件加载进内存
            df = readDelimitedFile(
                file_path,
                encoding_code,
                delimiter,
                self.contains_header_check.isChecked(),
                num_rows,
            )

            return df  # 成功后直接返回

        except (IOError, OSError, FileNotFoundError) as e:
            self.reportReadError(f"File access error: {file_path} - {e}")
        except Exception as e:
            self.reportReadError(f"Parsing error: {file_path} - {e}")

        return None

    def reportReadError(self, message):
        """Store a read error and send it to the application output when available."""
        self.last_error = message
        try:
            Func.printOut(message, 3)
        except (AttributeError, RuntimeError):
            print(message)

    # 读取多个文件
    def readMultipleFiles(self, fileList, readAllRows=False, addFilenameVariable=None):
        all_dfs = []
        read_errors = []
        self.last_error = ""
        try:
            if isinstance(fileList, list):
                for file in fileList:
                    df = self.readFile(file, readAllRows)

                    if df is None:
                        read_errors.append(self.last_error or f"Unable to read file: {file}")
                        continue

                    all_dfs.append(df)

                if read_errors:
                    self.last_error = "\n".join(read_errors)
                    return None

                if not all_dfs:
                    self.last_error = "No readable files were selected."
                    return None

                add_source_file = self.add_source_file if addFilenameVariable is None else addFilenameVariable
                if add_source_file:
                    all_dfs, _source_column = addSourceFileColumn(all_dfs, fileList)

                if len(all_dfs) == 1:
                    return all_dfs[0]
                return pd.concat(all_dfs, ignore_index=True, copy=False)

            self.last_error = "The file list is invalid."
            return None
        except Exception as e:
            self.reportReadError(f"Error while combining decoded files: {e}")
            return None

    def rejectEvent(self):
        self.close()

    def closeEvent(self, event):
        """Keep the dialog alive until its complete-read worker has stopped."""
        if self.final_read_thread is not None and self.final_read_thread.isRunning():
            event.ignore()
            return
        super().closeEvent(event)

    def getContainHeadStatus(self):
        return self.contains_header_check.isChecked()

    def getFiles(self):
        return self.files
