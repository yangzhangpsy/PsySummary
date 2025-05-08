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

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # 去除问号按钮
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
        self.view_table.setRowCount(0)
        self.setData()

    def readFinalData(self):
        data = self.readMultipleFiles(self.files, True, False)
        return data

    def setData(self):
        if self.files:
            try:
                data = self.readMultipleFiles(self.files)
                self.setTable(data)
            except Exception as e:
                msg = MessageBox(QMessageBox.Warning, "Warning",
                                 f"Please choose a different encoding for the error.\n{e}")
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

        specialCharDict = {"WhiteSpace": "\s",
                           '.': '\.',
                           "$": "\$",
                           "^": "\^",
                           "*": "\*",
                           "+": "\+",
                           "|": "\|"}

        if delimiter in specialCharDict:
            delimiter = specialCharDict[delimiter]

        return text_format, delimiter

    def readFile(self, file_path, readAllRows=False):
        if not file_path:
            raise ValueError("File path is empty or not provided.")

        try:
            df = None

            [code, splitCode] = self.getFormatAndDelimiter()

            with open(file_path, 'r', encoding=code) as file:
                lines = file.readlines()

                if readAllRows:
                    max_rows = len(lines)
                else:
                    max_rows = min(10, len(lines))

                if self.contains_header_check.isChecked():
                    variable_names = re.split(splitCode, lines[0].strip())  # 第一行为变量名
                    data = [re.split(splitCode, line.strip()) for line in lines[1:max_rows]]  # 以分隔符分隔的变量值
                else:
                    data = [re.split(splitCode, line.strip()) for line in lines[0:max_rows]]
                    variable_names = [f"Column{i + 1}" for i in range(len(data[0]))]

                df = pd.DataFrame(data)

                if len(variable_names) >= df.shape[1]:
                    df.columns = variable_names[:df.shape[1]]
                else:
                    df.columns = variable_names + [f'untitled{iVar}' for iVar in
                                                   range(df.shape[1] - len(variable_names))]
                # return df
        except (IOError, OSError, FileNotFoundError) as e:
            PsyDataFunc.printOut(f"Error in reading file! File probably changed/moved.:{file_path}:{e}", 3)
            # return None

        except Exception as e:
            PsyDataFunc.printOut(f"Error in reading file! File probably has bad format:{file_path}:{e}", 3)
            # return None
        finally:
            return df

    # 读取多个文件
    def readMultipleFiles(self, fileList, readAllRows=False, addFilenameVariable=False):
        all_dfs = []
        try:
            if isinstance(fileList, list):
                for file in fileList:
                    df = self.readFile(file, readAllRows)

                    if df is not None:
                        if addFilenameVariable:
                            fileName = os.path.basename(file)
                            if fileName not in df.columns:
                                df = df.assign(fileName=fileName)

                        all_dfs.append(df)

                if all_dfs:
                    return pd.concat(all_dfs, ignore_index=True)
                else:
                    return None
            else:
                return None
        except Exception as e:
            PsyDataFunc.printOut(f"Error in reading file:{e}", 3)
            return None

    def rejectEvent(self):
        self.close()

    def getContainHeadStatus(self):
        return self.contains_header_check.isChecked()

    def getFiles(self):
        return self.files
