# -*- coding: utf-8 -*-
import os

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QMessageBox, QComboBox, QTableWidget, QPushButton, QGridLayout, QLabel, QHBoxLayout, QVBoxLayout

from app.lib.message_box import MessageBox
from app.psyDataFunc import PsyDataFunc


class ImportFileUI(QDialog):
    def __init__(self, lst, files):
        super().__init__()
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # 去除问号按钮
        self.setWindowIcon(PsyDataFunc.getImageObject("icon.png", type=1))

        self.text_format_comboBox = QComboBox()
        self.delimiter_comboBox = QComboBox()
        self.view_table = QTableWidget()

        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")

        self.lst = lst
        self.previewLst = ['utf-8', ' ']
        self.files = files

        self.initUI()

    def initUI(self):
        self.setupUI()

        self.text_format_comboBox.currentIndexChanged.connect(self.codeChange)
        self.delimiter_comboBox.setEditable(True)
        self.delimiter_comboBox.currentTextChanged.connect(self.splitChange)
        self.ok_btn.clicked.connect(self.acceptEvent)
        self.cancel_btn.clicked.connect(self.rejectEvent)

        self.setData()

    def setupUI(self):
        self.setObjectName("Dialog")
        self.resize(510, 282)
        self.setWindowTitle("Data Format")

        self.text_format_comboBox.setObjectName("formater")
        self.text_format_comboBox.addItems(["utf-8", "gbk"])

        self.delimiter_comboBox.setObjectName("delimiter")
        self.delimiter_comboBox.addItems([" ", ",", "."])

        self.view_table.setObjectName("ViewTable")
        self.view_table.setColumnCount(0)
        self.view_table.setRowCount(0)

        gridLayout = QGridLayout()
        gridLayout.addWidget(QLabel("Text Encoding:"), 0, 0, 1, 1)
        gridLayout.addWidget(self.text_format_comboBox, 0, 1, 1, 1)
        gridLayout.addWidget(QLabel("Text Delimiter:"), 0, 3, 1, 1)
        gridLayout.addWidget(self.delimiter_comboBox, 0, 4, 1, 1)
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
        code = self.text_format_comboBox.currentText()
        splitCode = self.delimiter_comboBox.currentText()
        self.lst[0] = code
        self.lst[1] = splitCode
        self.close()

    def codeChange(self):
        code = self.text_format_comboBox.currentText()
        self.previewLst[0] = code
        self.view_table.setRowCount(0)
        self.setData()

    def splitChange(self):
        split = self.delimiter_comboBox.currentText()
        self.previewLst[1] = split
        self.view_table.setRowCount(0)
        self.setData()

    def setData(self):
        try:
            data = self.readMultipleFiles(self.files)
            self.setTable(data)
        except Exception as e:
            msg = MessageBox(QMessageBox.Warning, "Warning", f"Please choose a different encoding for the error.\n{e}")
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

    def readFile(self, file_path):
        if not file_path:
            raise ValueError("File path is empty or not provided.")

        try:
            tmp = self.previewLst
            with open(file_path, 'r', encoding=tmp[0]) as file:
                lines = file.readlines()
                variable_names = lines[0].strip().split(tmp[1])  # 第一行为变量名
                data = [line.strip().split(tmp[1]) for line in lines[1:]]  # 以分隔符分隔的变量值
                df = pd.DataFrame(data, columns=variable_names)
                return df
        except Exception as e:
            PsyDataFunc.printOut(f"Error in reading file:{file_path}:{e}", 3)
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
            PsyDataFunc.printOut(f"Error in reading file:{e}", 3)
            return None

    def rejectEvent(self):
        self.close()
