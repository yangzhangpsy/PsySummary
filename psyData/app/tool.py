import re
from operator import lt, le, gt, ge

import numpy as np
import pandas as pd
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

from app.lib import MessageBox
from app.lib.cdfPoolingWidget import fit_outlier_model, CdfPoolingWidget
from app.psyDataFunc import PsyDataFunc
from app.rtDist import CDF_pooling_main


# from rtDist import CDF_pooling_main
# from app.lib import fit_outlier_model, CdfPoolingWidget


def isCompareCond(expression: str):
    comparison_operators = {'<', '>', '<=', '>='}
    return any(operator in expression for operator in comparison_operators)


def executeDataFilter(dataFrame, variableName: str, compareType: str, value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        values = value.values
    else:
        values = value

    compare_functions = {
        '<': lt,
        '<=': le,
        '>': gt,
        '>=': ge
    }

    valid_compare_types = ['<', '<=', '>', '>=']
    if compareType not in valid_compare_types:
        raise ValueError(f"CompareType {compareType} is invalid. Should be one of: {valid_compare_types}")

    compare_operation = compare_functions.get(compareType)
    filtered_index = compare_operation(dataFrame[[variableName]].values, values)
    return filtered_index


def contains_empty_list(x):
    return any(hasattr(item, '__len__') and len(item) == 0 for item in x)
    # for item in x:
    #     try:
    #         if len(item) == 0:
    #             return True
    #     except Exception as e:
    #         if 'has no len()' in str(e):
    #             continue
    # return False


def check_dataframe_empty_cols(df, checkList, checkEmptyArray=True):
    if isinstance(checkList, str):
        checkList = [checkList]
    if checkEmptyArray:
        contains_empty_cols = df[checkList].apply(contains_empty_list)
    else:
        contains_empty_cols = df[checkList].isnull().any()

    return contains_empty_cols[contains_empty_cols].index.to_list()


def sumTable2DataFrame(row_var_list: list, column_var_list: list, sumTable, dataFrame, getShiftingZ: bool = False):
    variableNamesAll = row_var_list.copy()
    variableNamesAll.extend(column_var_list)

    output_df = pd.DataFrame(columns=['values'], index=dataFrame.index)

    if row_var_list:
        for rowIndex in sumTable.index:
            if isinstance(sumTable.index, pd.MultiIndex):
                row_value_list = list(rowIndex)
            else:
                row_value_list = [rowIndex]

            if column_var_list:
                for colIndex in sumTable.columns:
                    category_Value_list = row_value_list.copy()

                    if isinstance(sumTable.columns, pd.MultiIndex):
                        category_Value_list.extend(list(colIndex))
                    else:
                        category_Value_list.extend([colIndex])

                    logical_compare = dataFrame[(variableNamesAll)] == category_Value_list
                    output_df.loc[logical_compare.all(axis=1), 'values'] = sumTable.loc[rowIndex, colIndex]
            else:
                category_Value_list = row_value_list.copy()
                logical_compare = dataFrame[(variableNamesAll)] == category_Value_list
                output_df.loc[logical_compare.all(axis=1), 'values'] = sumTable.loc[rowIndex, sumTable.columns[0]]
    else:
        for colIndex in sumTable.columns:
            if isinstance(sumTable.columns, pd.MultiIndex):
                category_Value_list = list(colIndex)
            else:
                category_Value_list = [colIndex]

            logical_compare = dataFrame[(variableNamesAll)] == category_Value_list
            output_df.loc[logical_compare.all(axis=1), 'values'] = sumTable.loc[sumTable.index[0], colIndex]

    if getShiftingZ:
        output_df['values'] = output_df['values'].apply(StatisticTool.singleShiftZs)

    return output_df


def validate_inputs(row_var_list, column_var_list, sumTable, dataFrame):
    if not row_var_list and not column_var_list:
        raise ValueError("At least one of row_var_list or column_var_list must be non-empty.")
    if not isinstance(sumTable, pd.DataFrame) or not isinstance(dataFrame, pd.DataFrame):
        raise TypeError("sumTable and dataFrame must be an type of pd.DataFrame.")
    if not sumTable.index.is_unique or not dataFrame.index.is_unique:
        raise ValueError("Index in sumTable and dataFrame must be unique.")


def getValueInExpression(expression: str):
    numbers_str_list = re.findall(r"-?\d+\.\d+|-?\d+", expression)
    if not numbers_str_list:
        raise ValueError("No number found in expression.")

    nz = int(numbers_str_list[0]) if '.' not in numbers_str_list[0] else float(numbers_str_list[0])

    return nz


def doFilterOutData(row_var_list: list, column_var_list: list, expression: str, dataFrame, columnName: str):
    compareTypeStr = expression[:2].strip()
    nz = None

    if 'Shifting Z' in expression or 'SD' in expression or 'MAD' in expression:
        # the cutoff value type is a shifting z or specific times of sd
        if len(row_var_list) == 0 and len(column_var_list) == 0:
            if 'MAD' in expression:
                # here mean actually is median
                mean = dataFrame[columnName].median()
                # the magic num 1.4826 come from Rousseeuw & Croux, 1993, see detail in Leys JESP, 2013,764-766
                # here sd actually is MAD
                sd = 1.4826 * (np.median(np.abs(dataFrame[columnName] - mean)))
            else:
                mean = dataFrame[columnName].mean()
                sd = dataFrame[columnName].std()

            if 'Shifting Z' in expression:
                nz = StatisticTool.singleShiftZs(dataFrame.shape[0])
            elif 'SD' in expression or 'MAD' in expression:
                nz = getValueInExpression(expression)
        else:
            mean_table = pd.pivot_table(dataFrame, index=row_var_list, columns=column_var_list, values=columnName)
            std_table = pd.pivot_table(dataFrame, index=row_var_list, columns=column_var_list, values=columnName,
                                       aggfunc='std')
            if 'MAD' in expression:
                median_table = pd.pivot_table(dataFrame, index=row_var_list, columns=column_var_list, values=columnName,
                                              aggfunc='median')
                # fake mean, which is actually the median
                mean = sumTable2DataFrame(row_var_list, column_var_list, median_table, dataFrame)

                temp_var_name = columnName + '_temp_median_diff'
                while temp_var_name in dataFrame:
                    temp_var_name = f"{columnName}_temp_median_diff_{int(np.random.rand(1) * 1000)}"

                # calculate the MAD b*median(abs(x - median(x)))
                dataFrame[temp_var_name] = np.abs(dataFrame[columnName] - mean.iloc[:, 0])

                median_table2 = pd.pivot_table(dataFrame, index=row_var_list, columns=column_var_list,
                                               values=temp_var_name,
                                               aggfunc='median')

                sd = sumTable2DataFrame(row_var_list, column_var_list, median_table2, dataFrame)
                sd *= 1.4826
                # remove the temp_var (abs(x - median(x)))
                dataFrame.drop(columns=[temp_var_name])

            else:
                mean = sumTable2DataFrame(row_var_list, column_var_list, mean_table, dataFrame)
                sd = sumTable2DataFrame(row_var_list, column_var_list, std_table, dataFrame)

            if 'Shifting Z' in expression:
                count_table = pd.pivot_table(dataFrame, index=row_var_list, columns=column_var_list, values=columnName,
                                             aggfunc='count')
                nz = sumTable2DataFrame(row_var_list, column_var_list, count_table, dataFrame, True)
            elif 'SD' in expression or 'MAD' in expression:
                nz = getValueInExpression(expression)

        if compareTypeStr == '>' or compareTypeStr == '>=':
            cutoff_Value = mean - nz * sd
        else:
            cutoff_Value = mean + nz * sd

    else:
        # the cutoff value type is a raw number
        cutoff_Value = getValueInExpression(expression)

    filtered_index = executeDataFilter(dataFrame, columnName, compareTypeStr, cutoff_Value)

    return filtered_index


class StatisticTool:
    @staticmethod
    def checkEmptyNullValue(dataFrame, row_var_list, column_var_list, checkEmptyOnly=False):
        if isinstance(row_var_list, str):
            row_var_list = [row_var_list]
        if isinstance(column_var_list, str):
            column_var_list = [column_var_list]

        variableNamesAll = row_var_list.copy()
        variableNamesAll.extend(column_var_list)

        contains_empty_cols = check_dataframe_empty_cols(dataFrame, variableNamesAll)
        if contains_empty_cols:
            raise Exception(
                f"The following variables contains empty []:{contains_empty_cols},\n filter the empty values out by defining checklist in the filter window")

        if checkEmptyOnly:
            return False

        contains_null_cols = check_dataframe_empty_cols(dataFrame, variableNamesAll, False)
        if contains_null_cols:
            raise Exception(
                f"The following variables contains null values:{contains_null_cols},\n filter the null values out by defining checklist in the filter window")

        return False

    @staticmethod
    def singleShiftZs(count):
        if count >= 100:
            z_score = 2.5
        elif 50 <= count < 100:
            z_score = ((count - 50) * ((2.50 - 2.48) / 50)) + 2.48
        elif 35 <= count < 50:
            z_score = ((count - 35) * ((2.48 - 2.45) / 15)) + 2.45
        elif 30 <= count < 35:
            z_score = ((count - 30) * ((2.45 - 2.431) / 5)) + 2.431
        elif 25 <= count < 30:
            z_score = ((count - 25) * ((2.431 - 2.41) / 5)) + 2.41
        elif 20 <= count < 25:
            z_score = ((count - 20) * ((2.41 - 2.391) / 5)) + 2.391
        elif 15 <= count < 20:
            z_score = ((count - 15) * ((2.391 - 2.326) / 5)) + 2.326
        elif count == 14:
            z_score = 2.31
        elif count == 13:
            z_score = 2.274
        elif count == 12:
            z_score = 2.246
        elif count == 11:
            z_score = 2.22
        elif count == 10:
            z_score = 2.173
        elif count == 9:
            z_score = 2.12
        elif count == 8:
            z_score = 2.05
        elif count == 7:
            z_score = 1.961
        elif count == 6:
            z_score = 1.841
        elif count == 5:
            z_score = 1.68
        elif count == 4:
            z_score = 1.458
        else:
            z_score = 1
        return z_score

    # 转换规则

    @staticmethod
    def filterData(row_var_list, column_var_list, dataFrame, ruleList):
        StatisticTool.checkEmptyNullValue(dataFrame, row_var_list, column_var_list)

        tmp_data_frame = dataFrame.copy()

        be_printed_omega_str = ''

        for rule in ruleList:
            variable_name, conditional_expression = rule.split(':')
            variable_name = variable_name.strip()

            if 'Pooling CDF' != conditional_expression:
                be_printed_omega_str += '-1, '

            # 区分range规则和checklist规则
            if isCompareCond(conditional_expression):
                if not pd.api.types.is_numeric_dtype(tmp_data_frame[variable_name]):
                    tmp_data_frame[variable_name] = pd.to_numeric(tmp_data_frame[variable_name], errors='coerce')

                # range规则
                if 'and' in conditional_expression:
                    expression_1, expression_2 = conditional_expression.split('and')

                    filter_index1 = doFilterOutData(row_var_list, column_var_list, expression_1, tmp_data_frame, variable_name)
                    filter_index2 = doFilterOutData(row_var_list, column_var_list, expression_2, tmp_data_frame, variable_name)

                    tmp_data_frame = tmp_data_frame[np.logical_and(filter_index1, filter_index2)]
                elif 'or' in conditional_expression:
                    expression_1, expression_2 = conditional_expression.split('or')

                    filter_index1 = doFilterOutData(row_var_list, column_var_list, expression_1, tmp_data_frame, variable_name)
                    filter_index2 = doFilterOutData(row_var_list, column_var_list, expression_2, tmp_data_frame, variable_name)

                    tmp_data_frame = tmp_data_frame[np.logical_or(filter_index1, filter_index2)]
                else:
                    filter_index1 = doFilterOutData(row_var_list, column_var_list, conditional_expression,
                                                     tmp_data_frame, variable_name)
                    tmp_data_frame = tmp_data_frame[filter_index1]

            elif 'Pooling CDF' == conditional_expression:
                omega_value = -1
                try:
                    tmp_data_frame = CDF_pooling_main(tmp_data_frame, row_var_list, column_var_list, variable_name)
                except Exception as e:
                    PsyDataFunc.printOut(f"Failed to fit the data. The 'Pooling CDF' filter will be skipped. detailed Error: {e}", 4)
                else:
                    po_hat, omega_value = fit_outlier_model(tmp_data_frame[f"{variable_name}_cdf"])

                    if omega_value:
                        cdfPoolingDialog = CdfPoolingWidget(tmp_data_frame[f"{variable_name}_cdf"], po_hat, omega_value)

                        cdfPoolingDialog.exec_()

                        omega_value = cdfPoolingDialog.omega_hat

                        if omega_value == -1:
                            PsyDataFunc.printOut(
                                f"Aborted CDF pooling. The 'Pooling CDF' filter will be skipped, "
                                f"and no changes will be made to the data.", 4)
                        else:
                            filtered_df = tmp_data_frame[tmp_data_frame[f"{variable_name}_cdf"] > omega_value]
                            tmp_data_frame = filtered_df
                    else:
                        PsyDataFunc.printOut(
                            f"Failed to fit the CDF data. The 'Pooling CDF' filter will be skipped, "
                            f"and no changes will be made to the data.", 4)

                if omega_value == -1:
                    be_printed_omega_str += f'-1, '
                else:
                    omega_value_str = f"{omega_value:.{6}f}".rstrip('0').rstrip('.')
                    be_printed_omega_str += f'{omega_value_str}, '
            else:
                # checkList rules
                data = conditional_expression.split('=')
                data = [numStr.strip() for numStr in data]
                data = data[1:]
                # a possible bug here, double check later
                data = [numStr[1:-1] if "'" in numStr else float(numStr) for numStr in data]
                filtered_df = tmp_data_frame[tmp_data_frame[variable_name].isin(data)]
                tmp_data_frame = filtered_df

        PsyDataFunc.genScript(f'cdfPoolingOmegas = [{be_printed_omega_str[:-2]}]')

        return tmp_data_frame


class FlashMessageBox(MessageBox):
    def __init__(self, title, textStr, timeout=2000):
        super().__init__()
        self.title = title
        self.textStr = textStr
        self.timeout = timeout
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setStandardButtons(QMessageBox.Ok)
        self.setDefaultButton(QMessageBox.Ok)
        # self.setWindowIcon(Func.getImageObject("common/icon.png", type=1))
        # self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setText(self.textStr)

        QTimer.singleShot(self.timeout, lambda: self.close())
