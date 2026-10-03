import sys
import pandas as pd

from app.psyDataFunc import PsyDataFunc as Func
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, \
    QHeaderView, QApplication, QMainWindow, QTableView, QAbstractItemView, QInputDialog, QTabWidget, QMessageBox
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex, pyqtSignal, QThread
from app.lib import MessageBox
from app.expression import validate_variable_name, convert_variable_type


def variable_type(series):
    """Report storage types without scanning or coercing an entire column."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return 'Category'
    if pd.api.types.is_bool_dtype(dtype):
        return 'Boolean'
    if pd.api.types.is_complex_dtype(dtype):
        return 'Complex'
    if pd.api.types.is_numeric_dtype(dtype):
        return 'Numeric'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'Date/Time'
    return 'Text / Mixed' if pd.api.types.is_object_dtype(dtype) else 'Text'


class TypeConversionThread(QThread):
    """Validate and convert only one column off the GUI thread."""

    def __init__(self, series, target):
        super().__init__(QApplication.instance())
        self.original = series
        self.target = target
        self.result = None
        self.error = None
        QApplication.instance().aboutToQuit.connect(self.wait)

    def run(self):
        try:
            self.result = convert_variable_type(self.original, self.target)
        except Exception as error:
            self.error = str(error)


class PandasModel(QAbstractTableModel):
    """Read individual cells from the supplied frame without viewport snapshots."""

    def __init__(self, inputDF, parent=None):
        super(PandasModel, self).__init__(parent)
        self._df = inputDF
        self.decimals = {}

    def rowCount(self, parent=QModelIndex()):
        """Expose source rows only at the root of this flat table model."""
        return 0 if parent.isValid() else self._df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        """Expose source columns only at the root of this flat table model."""
        return 0 if parent.isValid() else self._df.shape[1]

    def displayDecimals(self, column):
        """Default float columns to four decimals while honoring explicit Auto."""
        name = self._df.columns[column]
        if name in self.decimals:
            return self.decimals[name]
        return 4 if pd.api.types.is_float_dtype(self._df[name].dtype) else None

    def data(self, index, role=Qt.DisplayRole):
        """Return one requested cell, including cells outside the current viewport."""
        if role != Qt.DisplayRole or not index.isValid() or index.model() is not self:
            return None
        row, column = index.row(), index.column()
        if 0 <= row < self._df.shape[0] and 0 <= column < self._df.shape[1]:
            value = self._df.iat[row, column]
            decimals = self.displayDecimals(column)
            if decimals is not None and pd.api.types.is_number(value) and not isinstance(value, (bool, complex)) and pd.notna(value):
                return format(value, f'.{decimals}f')
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """Preserve source labels and reject invalid header positions."""
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal and 0 <= section < self._df.shape[1]:
                return str(self._df.columns[section])
            elif orientation == Qt.Vertical and 0 <= section < self._df.shape[0]:
                return str(self._df.index[section])
        return None


class DataFrameTableWidget(QMainWindow):
    COLUMN_SIZE_SAMPLE_ROWS = 100

    def __init__(self, dataframe, parent=None, rename_callback=None, type_callback=None):
        super().__init__(parent)
        self.rename_callback = rename_callback
        self.type_callback = type_callback
        self.conversion_running = False
        self._conversion_worker = None
        self.model = PandasModel(dataframe, parent=self)
        self.view = QTableView()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Data Viewer')
        self.setWindowIcon(Func.getImageObject("common/icon.png", type=1))
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.resize(800, 600)

        self.view.setModel(self.model)
        self.view.horizontalHeader().sectionDoubleClicked.connect(self.renameColumn)
        self.view.horizontalHeader().setToolTip('Double-click a variable name to rename it.')
        self.view.horizontalHeader().setResizeContentsPrecision(self.COLUMN_SIZE_SAMPLE_ROWS)
        self.view.resizeColumnsToContents()
        self.view.setAcceptDrops(False)
        self.view.setSelectionMode(QAbstractItemView.NoSelection)
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.view.setFocusPolicy(Qt.NoFocus)
        self.view.setAlternatingRowColors(True)
        self.view.setVerticalScrollMode(QTableView.ScrollPerPixel)
        self.view.setHorizontalScrollMode(QTableView.ScrollPerPixel)

        # 设置表格视图为主窗口的中央部件
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.South)
        self.tabs.addTab(self.view, 'Data View')
        self.variable_view = QTableWidget()
        self.variable_view.setColumnCount(3)
        self.variable_view.setHorizontalHeaderLabels(['Name', 'Type', 'Decimals'])
        self.variable_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.variable_view.setAlternatingRowColors(True)
        self.variable_view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.variable_view.cellDoubleClicked.connect(self.editVariable)
        self.tabs.addTab(self.variable_view, 'Variable View')
        self.tabs.currentChanged.connect(lambda _index: self.refreshVariables())
        self.model.headerDataChanged.connect(self._headersChanged)
        layout.addWidget(self.tabs)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.refreshVariables()

    def refreshVariables(self):
        """Update lightweight column metadata, never copy the data table."""
        self.variable_view.setRowCount(self.model.columnCount())
        for index, name in enumerate(self.model._df.columns):
            kind = variable_type(self.model._df[name])
            decimals = self.model.displayDecimals(index) if kind == 'Numeric' else '—'
            if decimals is None:
                decimals = 'Auto'
            for column, value in enumerate((name, kind, decimals)):
                item = QTableWidgetItem(str(value))
                item.setToolTip('Double-click to edit.' if column != 2 or kind == 'Numeric' else 'Not applicable to non-numeric variables.')
                self.variable_view.setItem(index, column, item)

    def _headersChanged(self, orientation, first, last):
        """Keep formatting attached to renamed variables and refresh the Variable View."""
        if orientation == Qt.Horizontal:
            for section in range(first, min(last + 1, self.variable_view.rowCount())):
                old = self.variable_view.item(section, 0)
                new = self.model._df.columns[section]
                if old is not None and old.text() != new and old.text() in self.model.decimals:
                    self.model.decimals[new] = self.model.decimals.pop(old.text())
            self.refreshVariables()

    def editVariable(self, row, column):
        """Edit a name, display precision, or strictly validated storage type."""
        if self.conversion_running:
            return
        if column == 0:
            self.renameColumn(row)
            return
        name = self.model._df.columns[row]
        if column == 2:
            if variable_type(self.model._df[name]) != 'Numeric':
                return
            options = ['Auto'] + [str(value) for value in range(16)]
            decimals = self.model.displayDecimals(row)
            current = str(decimals if decimals is not None else 'Auto')
            chosen, accepted = QInputDialog.getItem(self, 'Display Decimals', 'Decimals (display only):', options, options.index(current), False)
            if accepted:
                self.model.decimals[name] = None if chosen == 'Auto' else int(chosen)
                self.model.dataChanged.emit(self.model.index(0, row), self.model.index(self.model.rowCount() - 1, row), [Qt.DisplayRole])
                self.refreshVariables()
            return
        chosen, accepted = QInputDialog.getItem(self, 'Convert Variable Type', 'Target type:', ['Numeric', 'Text', 'Boolean'], 0, False)
        if not accepted:
            return
        try:
            series = self.type_callback(name, chosen, None, self) if self.type_callback else self.model._df[name]
            worker = TypeConversionThread(series, chosen)
            self._conversion_worker = worker
            self._conversion_name = name
            self.conversion_running = True
            self.tabs.setEnabled(False)
            self.setWindowTitle('Data Viewer — Validating type conversion…')
            worker.finished.connect(self._finishConversion)
            worker.finished.connect(worker.deleteLater)
            worker.start()
        except Exception as error:
            MessageBox.warning(self, 'Type Conversion Error', str(error))

    def _finishConversion(self):
        """Confirm the full-column preview before committing any source-data change."""
        worker = self.sender()
        try:
            if worker.error:
                raise ValueError(worker.error)
            examples = '\n'.join(f'{worker.original.iloc[i]!r} → {worker.result.iloc[i]!r}' for i in range(min(5, len(worker.original))))
            message = (f'Convert {self._conversion_name!r} to {worker.target}?\n'
                       f'All {len(worker.original)} rows were validated. This changes actual data, not only display.\n'
                       + ('Numeric conversion removes leading zeros and may change textual formatting.\n' if worker.target == 'Numeric' else '')
                       + '\n' + examples)
            if QMessageBox.question(self, 'Confirm Type Conversion', message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            if self.type_callback:
                self.type_callback(self._conversion_name, worker.target, worker, self)
            else:
                self.model._df[self._conversion_name] = worker.result.array
            self.model.dataChanged.emit(self.model.index(0, 0), self.model.index(self.model.rowCount() - 1, self.model.columnCount() - 1))
            self.refreshVariables()
        except Exception as error:
            MessageBox.warning(self, 'Type Conversion Error', str(error))
        finally:
            self.conversion_running = False
            self._conversion_worker = None
            worker.result = None
            self.tabs.setEnabled(True)
            self.setWindowTitle('Data Viewer')

    def closeEvent(self, event):
        """Keep the conversion owner alive until validation and confirmation have finished."""
        if self.conversion_running:
            MessageBox.information(self, 'Type Conversion in Progress', 'Please wait for type conversion to finish.')
            event.ignore()
            return
        super().closeEvent(event)

    def renameColumn(self, section):
        """Request a validated source-column rename without making data cells editable."""
        if not 0 <= section < self.model.columnCount():
            return
        old_name = self.model._df.columns[section]
        name, accepted = QInputDialog.getText(
            self, 'Rename Variable', 'Variable name:', text=str(old_name))
        if not accepted or name.strip() == old_name:
            return
        try:
            if self.rename_callback is not None:
                self.rename_callback(old_name, name)
            else:
                name = validate_variable_name(name, self.model._df)
                self.model._df.rename(columns={old_name: name}, inplace=True)
                self.model.headerDataChanged.emit(Qt.Horizontal, section, section)
        except Exception as error:
            MessageBox.warning(self, 'Rename Variable Error', str(error))


class ResultFrameTableWidget(QTableWidget):
    fitRecordActivated = pyqtSignal(object)

    def __init__(self, dfs, columns, index, targetLst, fit_records=None):
        super().__init__()
        self.dfs = dfs
        self.columns = columns
        self.index = index
        self.targetLst = targetLst
        self.dataValues = dict()
        self.fit_records = fit_records or []
        self.fitRecordCells = {}
        self.fitValueLabels = {}

        self.setWindowTitle('Result View')
        self.setWindowIcon(Func.getImageObject("common/icon.png", type=1))
        self.setFocusPolicy(Qt.NoFocus)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setAlternatingRowColors(True)
        self.resize(600, 400)
        self.initUI()
        self.cellDoubleClicked.connect(self._activate_fit_record)

    @staticmethod
    def _format_value(value, decimal_num=4):
        """Format numeric results while preserving textual fit diagnostics."""
        if pd.api.types.is_number(value) and not isinstance(value, bool):
            if pd.isna(value):
                return 'NA'
            numeric_value = float(value)
            if numeric_value % 1:
                return f"{numeric_value:.{decimal_num}f}"
            return f"{numeric_value:.0f}"
        return str(value)

    @staticmethod
    def _tuple_value(value):
        """Normalize a scalar or multi-index value to a tuple."""
        return value if isinstance(value, tuple) else (value,)

    @staticmethod
    def _same_group_value(left, right):
        """Compare grouping values while treating paired missing values as equal."""
        try:
            if pd.isna(left) and pd.isna(right):
                return True
        except (TypeError, ValueError):
            pass
        return left == right

    def _record_for_result_cell(self, target_label, dataframe, source_row, source_col):
        """Resolve a displayed result cell to its fitted model and group."""
        candidates = [
            record for record in self.fit_records
            if target_label.startswith(f"{record.get('result_prefix', '')} ")
        ]
        if not candidates:
            return None
        if not candidates[0].get('group_vars'):
            return candidates[0]

        group_values = ()
        if self.index:
            group_values += self._tuple_value(dataframe.index[source_row])
        if self.columns:
            group_values += self._tuple_value(dataframe.columns[source_col])
        for record in candidates:
            record_values = record.get('group_values', ())
            if len(record_values) == len(group_values) and all(
                    self._same_group_value(left, right)
                    for left, right in zip(record_values, group_values)):
                return record
        return None

    def _register_fit_cell(self, table_row, table_col, target_label, dataframe, source_row, source_col):
        """Mark a fitted result cell as an entry point to its diagnostics plot."""
        record = self._record_for_result_cell(target_label, dataframe, source_row, source_col)
        if record is None:
            return
        self.fitRecordCells[(table_row, table_col)] = record
        item = self.item(table_row, table_col)
        if item is not None:
            item.setToolTip('Double-click to view fitted PDF and CDF diagnostics.')
        if target_label.endswith(' Converged'):
            self._add_fit_button(table_row, table_col, record)

    def _add_fit_button(self, table_row, table_col, record):
        """Add an explicit diagnostics button beside the convergence value."""
        item = self.item(table_row, table_col)
        if item is None:
            return
        cell_widget = QWidget(self)
        layout = QHBoxLayout(cell_widget)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(6)
        value_label = QLabel(item.text(), cell_widget)
        item.setText('')
        self.fitValueLabels[(table_row, table_col)] = value_label
        fit_button = QPushButton('View Fit', cell_widget)
        fit_button.setToolTip('Open fitted PDF and CDF diagnostics for this group.')
        fit_button.setFixedHeight(24)
        fit_button.clicked.connect(
            lambda _checked=False, fit_record=record: self.fitRecordActivated.emit(fit_record))
        layout.addWidget(value_label)
        layout.addWidget(fit_button)
        self.setCellWidget(table_row, table_col, cell_widget)

    def _activate_fit_record(self, row, column):
        """Open fit diagnostics when a mapped result cell is double-clicked."""
        record = self.fitRecordCells.get((row, column))
        if record is not None:
            self.fitRecordActivated.emit(record)

    def cellText(self, row, column):
        """Return the visible value of a cell, including values hosted by cell widgets."""
        value_label = self.fitValueLabels.get((row, column))
        if value_label is not None:
            return value_label.text()
        item = self.item(row, column)
        return item.text() if item is not None else ''

    def initUI(self):
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.hide()
        self.verticalHeader().hide()

        total_rows = 0
        total_columns = 0
        # 如果 index 和 columns 都为空
        if self.columns == [] and self.index == []:
            total_rows = len(self.dfs) * 3
            self.setRowCount(total_rows)
            self.setColumnCount(1)

            current_row = 0
            targetIndex = 0
            for cDF in self.dfs:
                if cDF is None:
                    continue

                target_label = str(self.targetLst[targetIndex])
                item = QTableWidgetItem(target_label)
                # self.setItem(current_row, 0, item)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                # 重新设置该单元格的 QTableWidgetItem，以便立即显示加粗字体
                self.setItem(current_row, 0, item)
                current_row += 1

                if isinstance(cDF, pd.DataFrame):
                    cValue = cDF.iloc[0, 0]
                else:
                    cValue = cDF

                cValueStr = self._format_value(cValue)

                self.setItem(current_row, 0, QTableWidgetItem(cValueStr))
                self.dataValues.update({(current_row, 0): cValue})
                if isinstance(cDF, pd.DataFrame):
                    self._register_fit_cell(current_row, 0, target_label, cDF, 0, 0)

                current_row += 2
                targetIndex += 1
        else:
            for cDF in self.dfs:
                num_rows, num_columns = cDF.shape
                total_rows += num_rows + len(self.columns) + 4
                total_columns = max(num_columns + len(self.index), total_columns, num_columns + 1)
            # 设置table行列值
            total_rows -= 2
            self.setRowCount(total_rows)
            self.setColumnCount(total_columns)

            current_row = 0  # 当前行的索引
            targetIndex = 0
            for cDF in self.dfs:
                # 填入table信息
                target_label = str(self.targetLst[targetIndex])
                item = QTableWidgetItem(target_label)
                # self.setItem(current_row, 0, item)
                font = item.font()
                font.setBold(True)
                item.setFont(font)

                # 重新设置该单元格的 QTableWidgetItem，以便立即显示加粗字体
                self.setItem(current_row, 0, item)
                targetIndex += 1
                current_row += 1
                # 填入 分类汇总 columns 名
                if self.columns:
                    column_index = cDF.columns
                    column_lst = column_index.names
                    column_pos = 0
                    for i in range(len(column_lst)):
                        if self.index:
                            column_pos = len(self.index) - 1
                        self.setItem(current_row + i, column_pos, QTableWidgetItem(str(column_lst[i])))
                    # 填入 columns 具体的信息
                    count = column_pos + 1
                    for value in column_index:
                        if isinstance(value, (float, int, str, bool)):
                            value = [value]
                        for index in range(len(value)):
                            self.setItem(current_row + index, count, QTableWidgetItem(str(value[index])))
                        count += 1
                # 填入 分类汇总index 名
                current_row += len(self.columns)
                if self.index:
                    index_index = cDF.index
                    index_lst = index_index.names
                    for i in range(len(index_lst)):
                        self.setItem(current_row, i, QTableWidgetItem(str(index_lst[i])))
                    # 填入 index 具体的信息
                    current_row += 1
                    tmpIndex = current_row
                    for value in index_index:
                        if isinstance(value, (float, int, str, bool)):
                            value = [value]

                        if hasattr(value, '__iter__'):
                            for index in range(len(value)):
                                self.setItem(tmpIndex, index, QTableWidgetItem(str(value[index])))
                        else:
                            self.setItem(tmpIndex, 0, QTableWidgetItem(str(value)))
                        tmpIndex += 1
                # 填入具体的数据
                # values = cDF.values
                # 遍历二维数组
                index_pos = 1
                if self.index:
                    index_pos = len(self.index)

                for row in range(cDF.shape[0]):
                    for col in range(cDF.shape[1]):
                        cValue = cDF.iat[row, col]
                        cValueStr = self._format_value(cValue)

                        self.setItem(current_row + row, index_pos + col, QTableWidgetItem(cValueStr))
                        self.dataValues.update({(current_row + row, index_pos + col): cValue})
                        self._register_fit_cell(
                            current_row + row, index_pos + col, target_label, cDF, row, col)

                # 跳到下一个表格
                current_row += cDF.shape[0] + 1

    def updateTable(self, decimal_num: str):

        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                locTuple = (row, col)

                if locTuple in self.dataValues:
                    item = self.item(row, col)
                    cValue = self.dataValues[locTuple]
                    formatted_value = self._format_value(cValue, decimal_num)
                    if locTuple in self.fitValueLabels:
                        self.fitValueLabels[locTuple].setText(formatted_value)
                        item.setText('')
                    else:
                        item.setText(formatted_value)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 示例DataFrame
    data = {'Column1': range(1, 10001),  # 增加更多行
            'Column2': ['A'] * 10000,  # 增加更多列
            'Column3': [i * 0.1 for i in range(10000)]}  # 增加更多列
    df = pd.DataFrame(data)

    viewer = DataFrameTableWidget(df)
    viewer.show()

    sys.exit(app.exec_())
