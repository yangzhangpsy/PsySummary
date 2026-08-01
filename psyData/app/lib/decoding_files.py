# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem, QMessageBox, QComboBox, QTableWidget, QPushButton, QGridLayout, \
    QLabel, QHBoxLayout, QVBoxLayout, QCheckBox

from app.lib import Dialog, VarComboBox, MessageBox
from app.psyDataFunc import PsyDataFunc



class DecodingFiles(Dialog):
    def __init__(self, files=None):
        super().__init__()
        if files is None:
            files = []

        self.files = files
        self.data = pd.DataFrame()
        self.last_error = ""

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # Remove the help button
        self.setWindowModality(Qt.WindowModal)
        # self.setWindowIcon(Func.getImageObject("common/icon.png", type=1))

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
        self.delimiter_comboBox.currentTextChanged.connect(self.reloadShortData)
        # self.ok_btn.clicked.connect(self.acceptEvent)
        self.cancel_btn.clicked.connect(self.rejectEvent)
        self.contains_header_check.clicked.connect(self.reloadShortData)

    def setupUI(self):
        self.text_format_comboBox = QComboBox()
        self.delimiter_comboBox = VarComboBox()
        self.contains_header_check = QCheckBox()
        self.contains_header_check.setChecked(True)
        self.contains_header_check.setText("Contains Header")

        self.view_table = QTableWidget()

        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")

        self.setObjectName("Dialog")
        self.resize(510, 282)
        self.setWindowTitle("Data Format")

        self.text_format_comboBox.setObjectName("DataFormatComboBox")
        self.text_format_comboBox.addItems(["utf-8", "gbk"])

        self.delimiter_comboBox.setObjectName("DelimiterComboBox")
        self.delimiter_comboBox.addItems(["WhiteSpace", ",", "."])

        self.view_table.setObjectName("ViewTable")
        self.view_table.setColumnCount(0)
        self.view_table.setRowCount(0)

        gridLayout = QGridLayout()
        gridLayout.addWidget(QLabel("Text Encoding:"), 0, 0, 1, 1)
        gridLayout.addWidget(self.text_format_comboBox, 0, 1, 1, 1)

        gridLayout.addWidget(QLabel("Text Delimiter:"), 0, 2, 1, 1)
        gridLayout.addWidget(self.delimiter_comboBox, 0, 3, 1, 1)

        gridLayout.addWidget(self.contains_header_check, 0, 4, 1, 1)
        gridLayout.addWidget(self.view_table, 1, 0, 8, 5)

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

    def readFinalData(self):
        data = self.readMultipleFiles(self.files, True, False)
        return data

    def setData(self):
        if not self.files:
            self.clearPreview()
            self.ok_btn.setEnabled(False)
            return

        data = self.readMultipleFiles(self.files)
        if data is None:
            self.clearPreview()
            self.ok_btn.setEnabled(False)
            msg = MessageBox(
                QMessageBox.Warning,
                "Warning",
                f"Unable to decode the selected file(s).\n{self.last_error}\n\n"
                "Please choose a different encoding or delimiter.")
            msg.exec_()
            return

        self.setTable(data)
        self.ok_btn.setEnabled(True)

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

        return text_format, delimiter

    def readFile(self, file_path, readAllRows=False):
        if not file_path:
            raise ValueError("File path is empty or not provided.")

        try:
            encoding_code, delimiter = self.getFormatAndDelimiter()
            data = pd.read_csv(
                file_path,
                sep=delimiter,
                encoding=encoding_code,
                nrows=None if readAllRows else 10,
                header=0 if self.contains_header_check.isChecked() else None,
                engine="python")
            if not self.contains_header_check.isChecked():
                data.columns = [f"Column{i + 1}" for i in range(data.shape[1])]
            return data
        except (IOError, OSError, FileNotFoundError) as e:
            self.reportReadError(f"File access error: {file_path} - {e}")
        except Exception as e:
            self.reportReadError(f"Parsing error: {file_path} - {e}")
        return None

    def reportReadError(self, message):
        """Store a read error and send it to the application output when available."""
        self.last_error = message
        try:
            PsyDataFunc.printOut(message, 3)
        except (AttributeError, RuntimeError):
            print(message)

    # Read multiple files
    def readMultipleFiles(self, fileList, readAllRows=False, addFilenameVariable=False):
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

                    if addFilenameVariable:
                        fileName = os.path.basename(file)
                        if fileName not in df.columns:
                            df = df.assign(fileName=fileName)

                    all_dfs.append(df)

                if read_errors:
                    self.last_error = "\n".join(read_errors)
                    return None

                if not all_dfs:
                    self.last_error = "No readable files were selected."
                    return None

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

    def getContainHeadStatus(self):
        return self.contains_header_check.isChecked()

    def getFiles(self):
        return self.files
