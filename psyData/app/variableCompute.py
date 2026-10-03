import sys
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QLabel, QLineEdit, QPushButton, QApplication, QListWidget,
                             QGridLayout, QHBoxLayout, QWidget, QListWidgetItem, QSizePolicy,
                             QMessageBox, QProgressBar)
from app.psyDataFunc import PsyDataFunc as Func
from app.lib import MessageBox

from app.lib.list_widget import ListWidget
from app.psyDataFunc import PsyDataFunc
from app.expression import (
    EXPRESSION_HELP, prepare_variable, evaluate_expression, to_aggregate_expression, runBoxcox,
    validate_variable_name,
)


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
                text = f"self.dataFrame[{current_item.text()!r}]"
                cursor = self.cursorPosition()
                self.insert(text)
                self.setCursorPosition(cursor + len(text))
                event.acceptProposedAction()
            else:
                event.ignore()


def evaluateVariableExpression(expression, widget):
    """Evaluate a GUI draft through the shared expression rules."""
    return evaluate_expression(expression, widget.dataFrame)


def convertExpressionToAggregateData(expression):
    """Convert data references without replacing text inside column names."""
    return to_aggregate_expression(expression)


class VariableComputeThread(QThread):
    """Calculate one read-only expression without accessing Qt widgets or logs."""

    def __init__(self, name, expression, data_frame, parent=None):
        super().__init__(parent)
        self.name = name
        self.expression = expression
        self.data_frame = data_frame
        self.result = None
        self.error = ''

    def run(self):
        """Prepare a result for the GUI; cancellation never commits partial data."""
        try:
            if not self.isInterruptionRequested():
                result = prepare_variable(self.name, self.expression, self.data_frame)
                if not self.isInterruptionRequested():
                    self.result = result
        except Exception as error:
            self.error = str(error)
        finally:
            self.data_frame = None


class VariableCompute(QWidget):
    transformFinished = pyqtSignal(str)
    computationRunningChanged = pyqtSignal(bool)
    computationFinished = pyqtSignal()

    def __init__(self, dataFrame: pd.DataFrame = None, parent=None):
        super(VariableCompute, self).__init__(parent, Qt.Window)
        self.setWindowModality(Qt.WindowModal)
        self.computation_running = False
        self._computation_thread = None
        self._computation_source = None
        self._cancel_requested = False
        self._close_after_computation = False
        self._close_on_success = True
        self._enabled_before_computation = []

        self.variable_list = None
        self.target_input = None
        self.numeric_expression = None
        self.variables = dataFrame.columns.tolist()
        self.dataFrame = dataFrame

        self.setWindowTitle('Compute Variable')
        self.setWindowIcon(Func.getImageObject("common/icon.png", type=1))
        # self.setGeometry(100, 100, 800, 600)
        self.initUI()
        QApplication.instance().aboutToQuit.connect(self._finishBeforeApplicationQuit)

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
        self.numeric_expression.setToolTip(EXPRESSION_HELP)
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
            button_width = 70 if position[1] == 4 else 50
            button.setFixedSize(button_width, 50)
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
        run_button = QPushButton('Run')
        self.run_button = run_button
        reset_button = QPushButton('Reset')
        cancel_button = QPushButton('Cancel')
        self.cancel_button = cancel_button

        ok_button.clicked.connect(self.on_ok_button_click)
        run_button.clicked.connect(self.on_run_button_click)
        reset_button.clicked.connect(self.on_reset_button_click)
        cancel_button.clicked.connect(self.on_cancel_button_click)

        button_layout.addWidget(reset_button)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(run_button)
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
        self.computation_status = QLabel('Computing variable…')
        self.computation_progress = QProgressBar()
        self.computation_progress.setRange(0, 0)
        self.computation_progress.setTextVisible(False)
        self.computation_progress.setFixedHeight(5)
        main_layout.addWidget(self.computation_status, 8, 0, 1, 5)
        main_layout.addWidget(self.computation_progress, 9, 0, 1, 5)
        self.computation_status.hide()
        self.computation_progress.hide()

        # Setting layout to main widget
        self.setLayout(main_layout)

        # Enable drag and drop
        self.variable_list.setDragEnabled(True)
        self.variable_list.setDragDropMode(QListWidget.DragOnly)

        self.numeric_expression.setAcceptDrops(True)

    def updateData(self, dataFrame):
        if self.computation_running:
            raise RuntimeError('Wait for the current variable calculation before replacing its data.')
        self.dataFrame = dataFrame
        self.variables = self.dataFrame.columns.tolist()

        self.variable_list.clear()

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

    def on_run_button_click(self):
        """Calculate and commit a variable while keeping this window open."""
        self._startComputation(close_on_success=False)

    def on_ok_button_click(self):
        """Calculate and close the window only after successful commit."""
        self._startComputation(close_on_success=True)

    def _startComputation(self, close_on_success):
        """Validate the draft and start a worker without blocking the Qt event loop."""
        if self.computation_running:
            return
        host = self.parentWidget()
        if host is not None and getattr(host, 'model_fit_running', False):
            MessageBox.information(self, 'Model Fitting in Progress',
                                   'Please wait for model fitting to finish before computing a variable.')
            return
        try:
            name = validate_variable_name(self.target_input.text(), self.dataFrame)
            source = to_aggregate_expression(self.numeric_expression.text())
            self._computation_source = self.dataFrame
            self._cancel_requested = False
            self._close_after_computation = False
            self._close_on_success = close_on_success
            thread = VariableComputeThread(name, source, self.dataFrame, self)
            self._computation_thread = thread
            thread.finished.connect(self._finishComputation)
            self._setComputationRunning(True)
            thread.start()
        except Exception as error:
            self._computation_thread = None
            self._computation_source = None
            self._setComputationRunning(False)
            MessageBox.information(self, 'Compute Variable Error', str(error), QMessageBox.Close)

    def _setComputationRunning(self, running):
        """Freeze edit controls but keep cancellation and ordinary repainting active."""
        self.computation_running = running
        if running:
            self._enabled_before_computation = [
                (widget, widget.isEnabled()) for widget in self.findChildren(QWidget)
                if isinstance(widget, (QPushButton, QLineEdit, QListWidget)) and widget is not self.cancel_button]
            for widget, _enabled in self._enabled_before_computation:
                widget.setEnabled(False)
            self.computation_status.setText('Computing variable…')
        else:
            for widget, enabled in self._enabled_before_computation:
                widget.setEnabled(enabled)
            self._enabled_before_computation = []
            self.cancel_button.setEnabled(True)
        self.computation_status.setVisible(running)
        self.computation_progress.setVisible(running)
        self.computationRunningChanged.emit(running)

    def _finishComputation(self):
        """Commit only a finished, current result on the GUI thread, then release it."""
        thread = self._computation_thread
        if thread is None:
            return
        successful = False
        try:
            if self._cancel_requested:
                return
            if thread.error:
                raise ValueError(thread.error)
            if thread.result is None:
                return
            host = self.parentWidget()
            if (self.dataFrame is not self._computation_source
                    or (host is not None and getattr(host, 'data', self.dataFrame) is not self.dataFrame)):
                raise ValueError('The input data changed during calculation. Reopen Compute Variable and try again.')
            name, column, source, warning = thread.result
            validate_variable_name(name, self.dataFrame)
            if warning and MessageBox.warning(
                    self, 'Non-finite Result', warning + '\n\nCreate this variable anyway?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            if self._cancel_requested:
                return
            acknowledgement = ', allow_nonfinite=True' if warning else ''
            script = f'aggData.calculateVariable({name!r}, {source!r}{acknowledgement})'
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            self.dataFrame[name] = column
            try:
                PsyDataFunc.genScript(script)
            except Exception:
                self.dataFrame.drop(columns=[name], inplace=True)
                raise
            self.variables.append(name)
            self.variable_list.addItem(item)
            self.target_input.setText(name)
            successful = True
            self.transformFinished.emit(name)
        except Exception as error:
            MessageBox.information(self, 'Compute Variable Error', str(error), QMessageBox.Close)
        finally:
            thread.result = None
            self._computation_thread = None
            self._computation_source = None
            thread.deleteLater()
            self._setComputationRunning(False)
            if (successful and self._close_on_success) or self._close_after_computation:
                self.close()
            self.computationFinished.emit()

    def requestCancelAndClose(self, close_parent=False):
        """Confirm discarding a running result and defer closure until thread cleanup."""
        if not self.computation_running or self._close_after_computation:
            return True
        destination = 'Data Summary' if close_parent else 'this window'
        answer = MessageBox.question(
            self, 'Variable Calculation in Progress',
            'A variable is being calculated in the background.\n'
            f'Discard its result and close {destination} after the calculation finishes?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return False
        self._close_after_computation = True
        self._cancelComputation()
        return True

    def _cancelComputation(self):
        """Request cancellation without terminating an active numerical operation."""
        self._cancel_requested = True
        if self._computation_thread is not None:
            self._computation_thread.requestInterruption()
        self.computation_status.setText('Cancelling; waiting for calculation to finish…')
        self.cancel_button.setEnabled(False)

    def closeEvent(self, event):
        """Keep the worker owner alive until an active calculation has ended."""
        if self.computation_running:
            self.requestCancelAndClose()
            event.ignore()
            return
        super().closeEvent(event)

    def _finishBeforeApplicationQuit(self):
        """Wait only on final application exit so Qt never destroys a running thread."""
        if self._computation_thread is not None:
            self._cancel_requested = True
            self._computation_thread.requestInterruption()
            self._computation_thread.wait()

    def on_cancel_button_click(self):
        if self.computation_running:
            self._close_after_computation = True
            self._cancelComputation()
            return
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
