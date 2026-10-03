import re
from app.dataPreparation import grouped_filter_estimates, is_range_expression, parse_checklist_values, split_filter_rule
from operator import lt, le, gt, ge

import numpy as np
import pandas as pd
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout
from app.psyDataFunc import PsyDataFunc as Func
from app.lib import MessageBox
from app.psyDataFunc import PsyDataFunc
from app.rtDist import CDF_pooling_main
from app.lib.cdfPoolingWidget import fit_outlier_model, CdfPoolingWidget


CONDITION_WISE_FILTER_REFERENCE = (
    'André, Q. (2022). Outlier exclusion procedures must be blind to the researcher\'s hypothesis. '
    'Journal of Experimental Psychology: General, 151(1), 213–223. '
    'https://doi.org/10.1037/xge0001069'
)


def warnConditionWiseFiltering(row_var_list, column_var_list, data_frame, rule_list):
    """Log a warning when active filters are combined with multiple observed data cells."""
    grouping_variables = list(dict.fromkeys(list(row_var_list) + list(column_var_list)))
    if not rule_list or not grouping_variables:
        return 0
    combination_count = int(data_frame[grouping_variables].drop_duplicates().shape[0])
    if combination_count > 1:
        grouping_label = ' × '.join(grouping_variables)
        PsyDataFunc.printOut(
            'Potential condition-wise filtering: The current Rows × Columns settings define '
            f'{combination_count} observed data cells ({grouping_label}) while filters are active. '
            'Any exclusion criterion estimated within these cells is applied separately by condition, '
            'which can exaggerate between-condition differences. Consider computing exclusion criteria '
            'blind to the experimental condition (for example, within participant but collapsed across '
            f'conditions). Reference: {CONDITION_WISE_FILTER_REFERENCE}',
            4,
        )
    return combination_count


def isCompareCond(expression: str):
    return is_range_expression(expression)


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
    multiplier = None

    if 'Shifting Z' in expression or 'SD' in expression or 'MAD' in expression:
        # Estimate the center and scale appropriate to the selected rule.
        if len(row_var_list) == 0 and len(column_var_list) == 0:
            values = dataFrame[columnName]
            if 'MAD' in expression:
                center = values.median(skipna=True)
                # Normal-consistent MAD (Rousseeuw & Croux, 1993; Leys et al., 2013).
                scale = 1.4826 * (values - center).abs().median(skipna=True)
            else:
                center = values.mean(skipna=True)
                scale = values.std(skipna=True, ddof=1)

            if 'Shifting Z' in expression:
                multiplier = StatisticTool.singleShiftZs(values.count())
            elif 'SD' in expression or 'MAD' in expression:
                multiplier = getValueInExpression(expression)
        else:
            shifting = 'Shifting Z' in expression
            center, scale, counts = grouped_filter_estimates(
                dataFrame, row_var_list + column_var_list, columnName,
                mad='MAD' in expression, count_needed=shifting)
            if shifting:
                # Evaluate the coefficient once per distinct count, not once per row.
                coefficients = {count: StatisticTool.singleShiftZs(count) for count in pd.unique(counts) if pd.notna(count)}
                multiplier = pd.Series(counts).map(coefficients).to_numpy().reshape(-1, 1)
            else:
                multiplier = getValueInExpression(expression)

        if compareTypeStr == '>' or compareTypeStr == '>=':
            cutoff_Value = center - multiplier * scale
        else:
            cutoff_Value = center + multiplier * scale

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
    def filterData(row_var_list, column_var_list, dataFrame, ruleList,
                   record_script=True, script_collector=None):
        StatisticTool.checkEmptyNullValue(dataFrame, row_var_list, column_var_list)

        tmp_data_frame = dataFrame.copy()

        be_printed_omega_str = ''

        for rule in ruleList:
            variable_name, conditional_expression = split_filter_rule(rule)

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
                            # Retain the non-tail region; values above omega are candidates for slow outliers.
                            filtered_df = tmp_data_frame[tmp_data_frame[f"{variable_name}_cdf"] <= omega_value]
                            tmp_data_frame = filtered_df
                    else:
                        PsyDataFunc.printOut(
                            f"Failed to fit the CDF data. The 'Pooling CDF' filter will be skipped, "
                            f"and no changes will be made to the data.", 4)

                if omega_value == -1:
                    be_printed_omega_str += f'-1, '
                else:
                    # Replay the exact cutoff instead of rounding it to six decimal places.
                    omega_value_str = repr(float(omega_value))
                    be_printed_omega_str += f'{omega_value_str}, '
            else:
                # checkList rules
                data = parse_checklist_values(conditional_expression)
                filtered_df = tmp_data_frame[tmp_data_frame[variable_name].isin(data)]
                tmp_data_frame = filtered_df

        if record_script:
            script_line = f'cdfPoolingOmegas = [{be_printed_omega_str[:-2]}]'
            if script_collector is not None:
                script_collector.append(script_line)
            else:
                PsyDataFunc.genScript(script_line)

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
        # self.setWindowIcon(Func.getImageObject("icon.png", type=1))
        # self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setText(self.textStr)

        QTimer.singleShot(self.timeout, lambda: self.close())
