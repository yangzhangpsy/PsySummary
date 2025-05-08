import sys
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QLabel, QLineEdit, QPushButton, QApplication, QListWidget,
                             QGridLayout, QHBoxLayout, QWidget, QListWidgetItem, QSizePolicy,
                             QMessageBox)

from app.lib import MessageBox
from app.lib.list_widget import ListWidget
from app.psyDataFunc import PsyDataFunc


class DroppableLineEdit(QLineEdit):
    def __init__(self, listWidget, *__args):
        super(DroppableLineEdit, self).__init__(*__args)
        self.setAcceptDrops(True)

        self.list_widget = listWidget

    def dragEnterEvent(self, event):
        if event.source() is self.list_widget:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.source() is self.list_widget:
            current_item = event.source().currentItem()
            if current_item:
                text = f"self.dataFrame['{current_item.text()}']"
                cursor = self.cursorPosition()
                self.insert(text)
                self.setCursorPosition(cursor + len(text))
                event.acceptProposedAction()
            else:
                event.ignore()


def runBoxcox(df):
    df, optimal_lambda = boxcox(df)
    return df


class VariableCompute(QWidget):
    transformFinished = pyqtSignal(str)

    def __init__(self, dataFrame: pd.DataFrame = None):
        super(VariableCompute, self).__init__()

        self.variable_list = None
        self.target_input = None
        self.numeric_expression = None
        self.variables = dataFrame.columns.tolist()
        self.dataFrame = dataFrame

        self.setWindowTitle('Compute Variable')
        self.setWindowIcon(PsyDataFunc.getImageObject("icon.png", type=1))
        # self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        # Main container widget
        # main_widget = QWidget()

        # Layouts
        main_layout = QGridLayout()

        # Target Variable Section
        target_label = QLabel('Target Variable:')
        target_label.setAlignment(Qt.AlignRight | Qt.AlignCenter)
        self.target_input = QLineEdit()

        # List View of Variables: 2 for sorting ContextMenu
        self.variable_list = ListWidget(2)

        for variable in self.variables:
            item = QListWidgetItem(variable, self.variable_list)
            item.setData(Qt.UserRole, variable)

        self.variable_list.setDragEnabled(True)
        self.variable_list.setSelectionMode(QListWidget.SingleSelection)
        self.variable_list.setDefaultDropAction(Qt.CopyAction)

        # Numeric Expression Section
        self.numeric_expression = DroppableLineEdit(self.variable_list)
        self.numeric_expression.setFixedHeight(60)

        self.numeric_expression.setAcceptDrops(True)

        # Numeric Buttons and Operators
        operators_layout = QGridLayout()
        buttons = [
            '7', '8', '9', '/', 'log',
            '4', '5', '6', '*', 'exp',
            '1', '2', '3', '-', '1/x',
            '0', '.', '==', '+', 'boxcox',
            '<', '>', '<=', '>=', '(',
            '!=', "&&", '|', 'Del', ')'
        ]

        positions = [(i, j) for i in range(6) for j in range(5)]

        for position, button_text in zip(positions, buttons):
            button = QPushButton(button_text)
            button.setFixedSize(50, 50)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            if button_text == 'Del':
                button.clicked.connect(self.on_delete_button_click)
            else:
                button.clicked.connect(
                    lambda checked, text=button_text.replace('&&', '&'): self.on_operator_button_click(text))
            operators_layout.addWidget(button, *position)

        # Bottom Buttons
        button_layout = QHBoxLayout()

        ok_button = QPushButton('OK')
        reset_button = QPushButton('Reset')
        cancel_button = QPushButton('Cancel')

        ok_button.clicked.connect(self.on_ok_button_click)
        reset_button.clicked.connect(self.on_reset_button_click)
        cancel_button.clicked.connect(self.on_cancel_button_click)

        button_layout.addWidget(reset_button)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)

        target_variable_layout = QHBoxLayout()

        target_variable_layout.addWidget(target_label)
        target_variable_layout.addWidget(self.target_input)
        equal_label = QLabel('=')
        equal_label.setAlignment(Qt.AlignCenter)
        target_variable_layout.addWidget(equal_label)
        target_variable_layout.addWidget(self.numeric_expression)

        main_layout.addLayout(target_variable_layout, 0, 0, 1, 5)
        main_layout.addWidget(self.variable_list, 1, 0, 5, 1)
        main_layout.addLayout(operators_layout, 1, 1, 3, 2)
        main_layout.addLayout(button_layout, 7, 0, 1, 4)

        # Setting layout to main widget
        self.setLayout(main_layout)

        # Enable drag and drop
        self.variable_list.setDragEnabled(True)
        self.variable_list.setDragDropMode(QListWidget.DragOnly)

        self.numeric_expression.setAcceptDrops(True)

    def updateData(self, dataFrame):
        self.dataFrame = dataFrame
        self.variables = self.dataFrame.columns.tolist()

        for variable in self.variables:
            item = QListWidgetItem(variable, self.variable_list)
            item.setData(Qt.UserRole, variable)

    def on_operator_button_click(self, text):
        translateDict = {
            'log': 'np.log()',
            'exp': 'np.exp()',
            '1/x': '1/()',
            '&': 'np.logical_and(,)',
            '|': 'np.logical_or(,)',
            'boxcox': 'runBoxcox()'
        }

        original_text = text
        if text in translateDict:
            text = translateDict[text]

            # 获取当前状态
        cursor_pos = self.numeric_expression.cursorPosition()
        original_length = len(self.numeric_expression.text())
        is_at_end = cursor_pos == original_length

        # 插入表达式
        self.numeric_expression.insert(text)

        # 仅当光标在末尾且是转换后的表达式时处理
        if is_at_end and original_text in translateDict:
            # 直接查找第一个左括号位置
            lparen_pos = text.find('(')
            if lparen_pos != -1:
                # 定位到第一个括号后
                new_pos = cursor_pos + lparen_pos + 1
                self.numeric_expression.setCursorPosition(new_pos)
            else:
                # 无括号则定位到末尾
                self.numeric_expression.setCursorPosition(cursor_pos + len(text))
        else:
            # 非末尾保持原有逻辑
            self.numeric_expression.setCursorPosition(cursor_pos + len(text))
    def on_operator_button_click_old(self, text):
        translateDict = {'log': 'np.log()',
                         'exp': 'np.exp()',
                         '1/x': '1/()',
                         '&': 'np.logical_and( , )',
                         '|': 'np.logical_or( , )',
                         'boxcox': 'runBoxcox()'}

        if text in translateDict:
            text = translateDict[text]

        # buttons = [
        #     '7', '8', '9', '/', 'log',
        #     '4', '5', '6', '*', 'exp',
        #     '1', '2', '3', '-', '1/x',
        #     '0', '.', '=', '+', '(',
        #     '<', '>', '<=', '>=', ')',
        #     '=', '!=', "&&", '|', 'Del'
        # ]
        # Insert the text at the current position in the QLineEdit
        cursor_pos = self.numeric_expression.cursorPosition()
        self.numeric_expression.insert(text)

        # Move the cursor to the end of the inserted text
        if text in translateDict:
            self.numeric_expression.setCursorPosition(cursor_pos + len(text))
        else:
            self.numeric_expression.setCursorPosition(cursor_pos + len(text))

    def on_reset_button_click(self):
        self.target_input.clear()
        self.numeric_expression.clear()

    def on_ok_button_click(self):
        target_variable_name = self.target_input.text()
        calculate_expression = self.numeric_expression.text()

        if target_variable_name not in self.dataFrame.columns:
            try:
                df = eval(calculate_expression)
                self.dataFrame[target_variable_name] = df

                self.variables.append(target_variable_name)
                item = QListWidgetItem(target_variable_name, self.variable_list)
                item.setData(Qt.UserRole, target_variable_name)

                # emit signal
                self.transformFinished.emit(target_variable_name)
                # only generate the script after successfully executing the computation
                PsyDataFunc.genScript(f"aggData.data['{target_variable_name}'] = {calculate_expression}")

                self.close()

            except Exception as e:
                MessageBox.information(self, "Warning",
                                       f"incorrect expression '{calculate_expression}': please check it carefully!\n"
                                       f"Error: {e}", QMessageBox.Close)
        else:
            MessageBox.information(self, "Warning",
                                   f"The target variable {target_variable_name} already in the data!\n"
                                   f"Please change the name before process!",
                                   QMessageBox.Close)

    def on_cancel_button_click(self):
        self.close()

    def on_delete_button_click(self):
        self.numeric_expression.backspace()


# Running the application
if __name__ == '__main__':
    app = QApplication(sys.argv)

    data = {'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['New York', 'Los Angeles', 'Chicago']}
    df = pd.DataFrame(data)

    mainWin = VariableCompute(df)
    mainWin.show()
    sys.exit(app.exec_())
