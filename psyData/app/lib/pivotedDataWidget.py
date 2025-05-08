import time

from PyQt5.QtCore import QTimer, QEventLoop
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QPushButton, QApplication, QFileDialog, QHBoxLayout, QSpinBox
import pandas as pd
import numpy as np

from app.lib.fitRTsDistThread import FitRTsDistThread
from app.psyDataFunc import PsyDataFunc
from app.tool import StatisticTool, FlashMessageBox
from app.lib.dataFrameTableWidget import ResultFrameTableWidget


def getStandardError(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))


def groupby_to_pivot_tables(grouped_result, index_var=None, columns_var=None):
    """
    Convert grouped results into multiple pivot-table-like DataFrames.

    Parameters:
    - grouped_result: Result of groupby operation
    - index_var: Variable name for pivot table rows (optional)
    - columns_var: Variable name for pivot table columns (optional)

    Returns: Dictionary of DataFrames
    """
    # Convert grouped result to DataFrame
    # Handle both Series and DataFrame inputs
    if isinstance(grouped_result, pd.Series):
        df = grouped_result.apply(pd.Series)
    else:
        df = grouped_result

        # Reset index to prepare for pivot operation
    df_reset = df.reset_index()

    # Extract original grouping variable names
    group_vars = list(df_reset.columns[:len(df.index.names)])

    # Dictionary to store pivot tables for each column
    pivot_tables = []

    # Iterate through columns (excluding grouping columns)
    for col in df.columns:
        # Handle different pivoting scenarios
        if index_var and columns_var:
            # Both index and columns variables specified
            # Create standard two-dimensional pivot table
            pivot_table = df_reset.pivot(
                index=index_var,
                columns=columns_var,
                values=col
            )
        elif columns_var and not index_var:
            # Only columns variable specified
            # Use pivot_table to get column-wise aggregation
            pivot_table = df_reset.pivot_table(
                index=None,
                columns=columns_var,
                values=col,
                aggfunc='first'
            )
        elif index_var and not columns_var:
            # Only index variable specified
            # Use pivot_table to get index-wise aggregation
            pivot_table = df_reset.pivot_table(
                index=index_var,
                columns=None,
                values=col,
                aggfunc='first'
            )
        else:
            # Return original column data
            pivot_table = df_reset[[col]].copy()

            # Store pivot table in results dictionary
        pivot_tables.append(pivot_table)

    return pivot_tables


def checkVariablesDuplication(row_vars, col_vars, target_vars):
    rowAndColVars = row_vars + col_vars
    for target_var in target_vars:
        target_var_name, operation = target_var.split('@')
        allVariables = rowAndColVars.copy()
        allVariables.append(target_var_name)

        if len(allVariables) != len(set(allVariables)):
            seen = set()
            duplicates = set([x for x in allVariables if x in seen or seen.add(x)])

            raise Exception(
                f'Using a variable across multiple lists within Rows, Columns, or Data list is not allowed.\n'
                f'Please remove the multiple used variable(s) {duplicates} and retry!')


def generateScript(row_vars, col_vars, target_vars, ruleList):
    analysis_script = [PsyDataFunc.list2Script(row_vars, 'rowVariables'),
                       PsyDataFunc.list2Script(col_vars, 'colVariables'),
                       PsyDataFunc.list2Script(ruleList, 'ruleList'),
                       PsyDataFunc.list2Script(target_vars, 'targetVariables'),
                       "aggData.summaryData(rowVariables, colVariables, ruleList, targetVariables, cdfPoolingOmegas)"]

    PsyDataFunc.genScript(analysis_script)


def handleFitThreadSignal(infoType: int, InfoStr: str, ShowTime: bool = True):
    PsyDataFunc.printOut(InfoStr, infoType, ShowTime)


class PivotedDataWidget(QWidget):
    def __init__(self, dataframe, row_vars, col_vars, target_vars, ruleList):
        super(PivotedDataWidget, self).__init__()
        self.fit_dist_thread = None
        self.table = None
        self.msg_box = None
        self.asynchronous = False  # status of asynchronous of the fit thread
        self.resultList = []
        self.filterStr = ''
        self.ruleList = ruleList
        self.result_frame_var_names = []
        self.fitMethods = ['Gamma (k, θ)',
                           'Weibull (k, θ)',
                           'LogNormal (k, θ)',
                           'Wald (m, a)',
                           'Ex-Wald (m, a, τ)',
                           'Shifted Wald (m, a, shift)',
                           'Ex-Gaussian (μ, σ, τ)',
                           'Inv-Gaussian (μ, λ)',
                           'Shifted Inv-Gaussian (μ, λ, shift)']

        self.initUI(dataframe, row_vars, col_vars, target_vars)

    def fitDistInBackground(self, dataFrame, operation, row_vars, col_vars, independentVarName,
                            distribution='Ex-Gaussian'):
        # self.asynchronous = True
        self.fit_dist_thread = FitRTsDistThread(dataFrame, operation, row_vars, col_vars, independentVarName,
                                                distribution)

        self.fit_dist_thread.fitStatus.connect(handleFitThreadSignal)
        self.fit_dist_thread.finished.connect(self.handleFitFinished)

        self.fit_dist_thread.start()

    def initUI(self, dataframe, row_vars, col_vars, target_vars):
        self.setWindowIcon(PsyDataFunc.getImageObject("icon.png", type=1))
        self.setWindowTitle("Aggregation Results")
        self.resize(400, 700)

        self.all_layout = QVBoxLayout()
        self.btns_layout = QHBoxLayout()

        pushButton = QPushButton('Clipboard')
        exportButton = QPushButton('Export')

        # Create a QSpinBox for controlling decimal places
        self.decimal_spin_box = QSpinBox()
        self.decimal_spin_box.setRange(0, 12)
        self.decimal_spin_box.setValue(4)
        self.decimal_spin_box.setSuffix(" decimal places")
        self.decimal_spin_box.valueChanged.connect(self.update_table)

        pushButton.setFixedWidth(100)
        exportButton.setFixedWidth(100)

        pushButton.clicked.connect(self.copyToClipboard)
        exportButton.clicked.connect(self.exportData)

        self.btns_layout.addWidget(pushButton)
        self.btns_layout.addWidget(exportButton)
        self.btns_layout.addWidget(self.decimal_spin_box)

        filters_Info = '\n'.join(self.ruleList)

        self.filterStr = filters_Info
        self.all_layout.addWidget(QLabel(filters_Info))

        try:
            checkVariablesDuplication(row_vars, col_vars, target_vars)

            for target_var in target_vars:
                target_var_name, operation = target_var.split('@')
                if not pd.api.types.is_numeric_dtype(dataframe[target_var_name]):
                    dataframe[target_var_name] = pd.to_numeric(dataframe[target_var_name], errors='coerce')

            tmpDataFrame = StatisticTool.filterData(row_vars, col_vars, dataframe, self.ruleList)

            # we check this within the filterData function
            # StatisticTool.checkEmptyNullValue(tmpDataFrame, row_vars, col_vars)

            self.result_frame_var_names = []
            for target_var in target_vars:
                target_var_name, operation = target_var.split('@')

                result = None
                if operation == 'Mean':
                    if len(row_vars) == 0 and len(col_vars) == 0:
                        result = tmpDataFrame[target_var_name].mean()
                    else:
                        result = pd.pivot_table(tmpDataFrame, index=row_vars, columns=col_vars, values=target_var_name)
                elif operation == 'Median':
                    if len(row_vars) == 0 and len(col_vars) == 0:
                        result = tmpDataFrame[target_var_name].median()
                    else:
                        result = pd.pivot_table(tmpDataFrame, index=row_vars, columns=col_vars, values=target_var_name,
                                                aggfunc='median')
                elif operation == 'Mode':
                    if len(row_vars) == 0 and len(col_vars) == 0:
                        result = tmpDataFrame[target_var_name].mode().iloc[0]
                    else:
                        result = pd.pivot_table(tmpDataFrame, index=row_vars, columns=col_vars, values=target_var_name,
                                                aggfunc=lambda x: x.mode().iloc[0])
                elif operation == 'Count':
                    if len(row_vars) == 0 and len(col_vars) == 0:
                        result = tmpDataFrame[target_var_name].count()
                    else:
                        result = pd.pivot_table(tmpDataFrame, index=row_vars, columns=col_vars, values=target_var_name,
                                                aggfunc='count')
                elif operation == 'Standard Deviation':
                    if len(row_vars) == 0 and len(col_vars) == 0:
                        result = tmpDataFrame[target_var_name].std()
                    else:
                        result = pd.pivot_table(tmpDataFrame, index=row_vars, columns=col_vars, values=target_var_name,
                                                aggfunc='std')
                elif operation == 'Max':
                    if len(row_vars) == 0 and len(col_vars) == 0:
                        result = tmpDataFrame[target_var_name].max()
                    else:
                        result = pd.pivot_table(tmpDataFrame, index=row_vars, columns=col_vars, values=target_var_name,
                                                aggfunc='max')
                elif operation == 'Min':
                    if len(row_vars) == 0 and len(col_vars) == 0:
                        result = tmpDataFrame[target_var_name].min()
                    else:
                        result = pd.pivot_table(tmpDataFrame, index=row_vars, columns=col_vars, values=target_var_name,
                                                aggfunc='min')
                elif operation == 'Variance':
                    if len(row_vars) == 0 and len(col_vars) == 0:
                        result = tmpDataFrame[target_var_name].var()
                    else:
                        result = pd.pivot_table(tmpDataFrame, index=row_vars, columns=col_vars, values=target_var_name,
                                                aggfunc='var')
                elif operation == 'Standard Error':
                    if len(row_vars) == 0 and len(col_vars) == 0:
                        result = tmpDataFrame[target_var_name].sem()
                    else:
                        result = pd.pivot_table(tmpDataFrame, index=row_vars, columns=col_vars, values=target_var_name,
                                                aggfunc=getStandardError)

                elif operation in self.fitMethods:
                    self.asynchronous = True
                    self.fitDistInBackground(tmpDataFrame, operation, row_vars, col_vars, target_var_name, operation)

                    # waiting until the fit thread is finished
                    self.process_with_async_wait()

                """
                handle concatenate results and result names only for non-fitting methods
                """
                if operation not in self.fitMethods:
                    self.updateResultDataframe(result, target_var)

                pd.set_option('display.float_format', lambda x: '%.10f' % x)
                # generate analysis script
                generateScript(row_vars, col_vars, target_vars, self.ruleList)

        except Exception as e:
            raise Exception(e)

        self.createResultTable(col_vars, row_vars)

    def start_fitting(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_fitting_status)
        self.timer.start(100)  # 100ms interval

    def process_with_async_wait(self):
        # 创建一个事件循环
        loop = QEventLoop()

        # 创建定时器用于处理异步操作
        timer = QTimer()
        last_print_time = time.time()

        def check_async_status():
            nonlocal last_print_time

            # 检查是否还在异步状态
            if not self.asynchronous:
                # 停止定时器
                timer.stop()
                # 退出事件循环
                loop.quit()
                return

                # 处理事件
            QApplication.processEvents()

            # 周期性打印（如果需要）
            current_time = time.time()
            if current_time - last_print_time > 1:
                last_print_time = current_time
                # print('...')
                # 可选：PsyDataFunc.printOut("Fitting ...", 0)

        # 设置定时器间隔和回调
        timer.setInterval(100)  # 100ms 检查一次
        timer.timeout.connect(check_async_status)

        # 启动定时器和事件循环
        timer.start()
        loop.exec_()

    def createResultTable(self, col_vars, row_vars):
        self.table = ResultFrameTableWidget(self.resultList, col_vars, row_vars, self.result_frame_var_names)

        self.all_layout.addWidget(self.table)
        self.all_layout.addLayout(self.btns_layout)
        self.setLayout(self.all_layout)

    def handleFitFinished(self, result, target_var, row_vars, col_vars):
        result = groupby_to_pivot_tables(result, row_vars, col_vars)
        self.updateResultDataframe(result, target_var)

        self.asynchronous = False
        self.fit_dist_thread.quit()
        self.fit_dist_thread.wait()

    def updateResultDataframe(self, result, target_var):
        if isinstance(target_var, list):
            self.result_frame_var_names.extend(target_var)
        else:
            self.result_frame_var_names.append(target_var)

        if isinstance(result, list):
            self.resultList.extend(result)
        else:
            self.resultList.append(result)

    def update_table(self):
        decimal_places = self.decimal_spin_box.value()

        self.table.updateTable(decimal_places)

    # 获取数据

    def getTextData(self):
        data = self.filterStr + '\n' + '\n'
        for row in range(self.table.rowCount()):
            for column in range(self.table.columnCount()):
                item = self.table.item(row, column)
                if item is not None:
                    data += item.text() + '\t'
                else:
                    data += '\t'
            data += '\n'
        return data

    # 导出数据
    def exportData(self):
        data = self.getTextData()
        try:
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'Text Files (*.txt)')
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(data)
        except Exception as e:
            print(e)

    # 复制表格内容到剪贴板
    def copyToClipboard(self):
        data = self.getTextData()

        self.showFlashMessage('Data copied to clipboard')
        clipboard = QApplication.clipboard()
        clipboard.setText(data)

    def showFlashMessage(self, e):
        self.msg_box = FlashMessageBox('Flash Message', str(e))
        self.msg_box.show()
