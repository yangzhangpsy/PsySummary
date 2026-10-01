import sys
import pandas as pd

from app.psyDataFunc import PsyDataFunc as Func
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, \
    QHeaderView, QApplication, QMainWindow, QTableView, QAbstractItemView
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex, pyqtSignal


class PandasModel(QAbstractTableModel):
    def __init__(self, inputDF, parent=None):
        super(PandasModel, self).__init__(parent)
        self.start_col = 0
        self.start_row = 0

        self._df = inputDF
        self._viewport_data = pd.DataFrame()

    def rowCount(self, parent=QModelIndex()):
        return self._df.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            if not self._viewport_data.empty:
                value = self._viewport_data.iat[index.row() - self.start_row, index.column() - self.start_col]
                return str(value)
        # if role == Qt.BackgroundRole:
        #     return QColor(Qt.white)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            elif orientation == Qt.Vertical:
                return str(self._df.index[section])
        return None

    def updateViewportData(self, start_row, end_row, start_col, end_col):
        self.start_col = start_col
        self.start_row = start_row

        self.beginResetModel()
        self._viewport_data = self._df.iloc[start_row:end_row + 1, start_col:end_col + 1].copy()
        self.endResetModel()


class DataFrameTableWidget(QMainWindow):
    def __init__(self, dataframe):
        super().__init__()
        self.start_col = 0
        self.start_row = 0
        self.model = PandasModel(dataframe)
        self.view = QTableView()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Data Viewer')
        self.setWindowIcon(Func.getImageObject("icon.png", type=1))
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.resize(800, 600)

        self.view.setModel(self.model)
        self.view.resizeColumnsToContents()
        self.view.setAcceptDrops(False)
        self.view.setSelectionMode(QAbstractItemView.NoSelection)
        self.view.setFocusPolicy(Qt.NoFocus)
        self.view.setAlternatingRowColors(True)
        self.view.setVerticalScrollMode(QTableView.ScrollPerPixel)
        self.view.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.view.viewport().installEventFilter(self)

        # 设置表格视图为主窗口的中央部件
        layout = QVBoxLayout()
        layout.addWidget(self.view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def eventFilter(self, source, event):
        if source is self.view.viewport() and event.type() == event.Paint:
            rect = self.view.viewport().rect()

            self.start_row = self.view.rowAt(rect.top())
            end_row = self.view.rowAt(rect.bottom())

            self.start_col = self.view.columnAt(rect.left())
            end_col = self.view.columnAt(rect.right())
            if end_row == -1:
                end_row = self.model.rowCount() - 1
            if end_col == -1:
                end_col = self.model.columnCount() - 1
            self.model.updateViewportData(self.start_row, end_row, self.start_col, end_col)
        return super().eventFilter(source, event)

        # self.setCentralWidget(self.view)

    def on_scroll(self):
        row_offset = self.view.verticalScrollBar().value()
        col_offset = self.view.horizontalScrollBar().value()

        view_port_rect = self.view.viewport().geometry()
        num_rows_to_show = int(view_port_rect.height() / self.view.rowHeight(0))
        num_cols_to_show = int(3 * view_port_rect.width() / self.view.columnWidth(0))

        self.model.layoutAboutToBeChanged.emit()

        self.model._viewport_data = self.model._df.iloc[row_offset:row_offset + num_rows_to_show, col_offset:col_offset + num_cols_to_show]
        self.model.layoutChanged.emit()


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
        self.setWindowIcon(Func.getImageObject("icon.png", type=1))
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
