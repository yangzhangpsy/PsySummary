import csv
import os
import re
from operator import lt, le, ge, gt

import mat73

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize

from .rtDist import gamma_estimate_x, wald_estimate_x, ex_wald_estimate_x, ex_gaussian_estimate_x, \
    inverse_gaussian_estimate_x, shifted_inverse_gaussian_estimate_x, weibull_estimate_x, log_normal_estimate_x, \
    shift_wald_estimate_x, CDF_pooling_main


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

    compare_operation = compare_functions.get(compareType)
    filtered_index = compare_operation(dataFrame[[variableName]].values, values)
    return filtered_index


def contains_empty_list(x):
    return any(hasattr(item, '__len__') and len(item) == 0 for item in x)


def getStandardError(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))


def getValueInExpression(expression: str):
    numbers_str_list = re.findall(r"-?\d+\.\d+?|-?\d+", expression)

    nz = int(numbers_str_list[0]) if '.' not in numbers_str_list[0] else float(numbers_str_list[0])
    return nz


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

                    logical_compare = dataFrame[variableNamesAll] == category_Value_list
                    output_df.loc[logical_compare.all(axis=1), 'values'] = sumTable.loc[rowIndex, colIndex]
            else:
                category_Value_list = row_value_list.copy()
                logical_compare = dataFrame[variableNamesAll] == category_Value_list
                output_df.loc[logical_compare.all(axis=1), 'values'] = sumTable.loc[rowIndex, sumTable.columns[0]]
    else:
        for colIndex in sumTable.columns:
            if isinstance(sumTable.columns, pd.MultiIndex):
                category_Value_list = list(colIndex)
            else:
                category_Value_list = [colIndex]

            logical_compare = dataFrame[variableNamesAll] == category_Value_list
            output_df.loc[logical_compare.all(axis=1), 'values'] = sumTable.loc[sumTable.index[0], colIndex]

    if getShiftingZ:
        output_df['values'] = output_df['values'].apply(singleShiftZs)

    return output_df


def doFilterOutData(row_var_list: list, column_var_list: list, expression: str, dataFrame, columnName: str):
    compareTypeStr = expression[:2].strip()
    nz = None

    if 'Shifting Z' in expression or 'SD' in expression or 'MAD' in expression:
        # the cutoff value type is a shifting z or specific times of sd
        if len(row_var_list) == 0 and len(column_var_list) == 0:
            if 'MAD' in expression:
                # here mean actually is median
                mean = dataFrame[columnName].median()
                # here sd actually is MAD
                sd = 1.4826 * (np.median(np.abs(dataFrame[columnName] - mean)))
            else:
                mean = dataFrame[columnName].mean()
                sd = dataFrame[columnName].std()

            if 'Shifting Z' in expression:
                nz = singleShiftZs(dataFrame.shape[0])
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


def filterDataFunc(row_var_list, column_var_list, dataFrame, ruleList, omegaValues=None):
    if omegaValues is None:
        omegaValues = [-1 for _ in ruleList]

    tmp_data_frame = dataFrame.copy()

    for index, rule in enumerate(ruleList):
        variable_name, conditional_expression = rule.split(':')
        variable_name = variable_name.strip()
        # 区分range规则和checklist规则
        if isCompareCond(conditional_expression):
            if not pd.api.types.is_numeric_dtype(tmp_data_frame[variable_name]):
                tmp_data_frame[variable_name] = pd.to_numeric(tmp_data_frame[variable_name], errors='coerce')

            # range规则
            if 'and' in conditional_expression:
                expression_1, expression_2 = conditional_expression.split('and')

                filter_index1 = doFilterOutData(row_var_list, column_var_list, expression_1, tmp_data_frame,
                                                variable_name)
                filter_index2 = doFilterOutData(row_var_list, column_var_list, expression_2, tmp_data_frame,
                                                variable_name)

                tmp_data_frame = tmp_data_frame[np.logical_and(filter_index1, filter_index2)]
            elif 'or' in conditional_expression:
                expression_1, expression_2 = conditional_expression.split('or')

                filter_index1 = doFilterOutData(row_var_list, column_var_list, expression_1, tmp_data_frame,
                                                variable_name)
                filter_index2 = doFilterOutData(row_var_list, column_var_list, expression_2, tmp_data_frame,
                                                variable_name)

                tmp_data_frame = tmp_data_frame[np.logical_or(filter_index1, filter_index2)]
            else:
                filter_index1 = doFilterOutData(row_var_list, column_var_list, conditional_expression,
                                                tmp_data_frame, variable_name)
                tmp_data_frame = tmp_data_frame[filter_index1]

        elif 'Pooling CDF' == conditional_expression:

            if 0 < omegaValues[index] < 1:
                tmp_data_frame = CDF_pooling_main(tmp_data_frame, row_var_list, column_var_list, variable_name)

                filtered_df = tmp_data_frame[tmp_data_frame[f"{variable_name}_cdf"] > omegaValues[index]]
                tmp_data_frame = filtered_df
            else:
                print(f"Skip 'Pool CDF' as the ω is out of the range [0, 1], and no changes will be made to the data.")
        else:
            # checkList rules
            data = conditional_expression.split('=')
            data = [numStr.strip() for numStr in data]
            data = data[1:]
            # a possible bug here, double check later
            data = [numStr[1:-1] if "'" in numStr else float(numStr) for numStr in data]
            filtered_df = tmp_data_frame[tmp_data_frame[variable_name].isin(data)]
            tmp_data_frame = filtered_df

    return tmp_data_frame


def outlier_mode_lnlike(params, cdf_values):
    po, omega = params

    if not (0 <= po <= 1 and 0 <= omega < 1):
        return np.inf

    valid_density = (1 - po)

    outlier_density = np.zeros_like(cdf_values)
    mask = cdf_values >= omega
    outlier_range = 1 - omega
    if outlier_range == 0:
        return np.inf

    outlier_density[mask] = 2 * (cdf_values[mask] - omega) / (outlier_range ** 2)

    mixture_density = valid_density + po * outlier_density
    mixture_density = np.maximum(mixture_density, 1e-12)

    return -np.sum(np.log(mixture_density))


def fit_outlier_model(cdf_values):
    init_params = [0.01, 0.98]
    bounds = [(0.0001, 0.2), (0.7, 0.999)]

    result = minimize(outlier_mode_lnlike, np.array(init_params), args=(cdf_values,), bounds=bounds)
    if result.success:
        return result.x
    else:
        return [None, None]


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


def flattenValue(value):
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):  # 如果是列表（二维数组）
        return value[0]
    else:  # 如果是整数值或其他类型的值
        if len(value) == 0:
            return None
        if len(value[0]) == 0:
            return value[0]
        if isinstance(value[0], str):
            return value[0]
        return value[0][0]  # 直接返回该值


def flattenValueMat73(value):
    if isinstance(value, (np.bool_, bool)):
        return float(value)

    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        elif value.size == 1:
            return value.item()
        elif value.ndim == 1 and not isinstance(value[0], str):
            return -np.sum(value)
        return value[0]

    elif isinstance(value, (str, list)):
        # Handle list case efficiently
        return value[0] if isinstance(value, list) and len(value) > 0 else value
    return value


def readDatFile(file_path, containsHeader=True, encodingFormat='utf-8', delimiter='\s'):
    try:
        df = None

        with open(file_path, 'r', encoding=encodingFormat) as file:
            lines = file.readlines()

            if containsHeader:
                variable_names = re.split(delimiter, lines[0].strip())  # 第一行为变量名
                data = [re.split(delimiter, line.strip()) for line in lines]  # 以分隔符分隔的变量值
            else:
                data = [re.split(delimiter, line.strip()) for line in lines]
                variable_names = [f"Column{i + 1}" for i in range(len(data[0]))]

            df = pd.DataFrame(data)

            if len(variable_names) >= df.shape[1]:
                df.columns = variable_names[:df.shape[1]]
            else:
                df.columns = variable_names + [f'untitled{iVar}' for iVar in range(df.shape[1] - len(variable_names))]
            # return df
    except Exception as e:
        raise IOError(f"Error in reading file! File probably changed/moved.:{file_path}:{e}", 3)
    finally:
        return df


class AggregateData(object):
    def __init__(self):
        super().__init__()

        self.data = pd.DataFrame()
        self.resultList = []
        self.fitMethods = ['Gamma (k, θ)',
                           'Weibull (k, θ)',
                           'LogNormal (k, θ)',
                           'Wald (m, a)',
                           'Ex-Wald (m, a, τ)',
                           'Shifted Wald (m, a, shift)',
                           'Ex-Gaussian (μ, σ, τ)',
                           'Inv-Gaussian (μ, λ)',
                           'Shifted Inv-Gaussian (μ, λ, shift)']

        self.DISTRIBUTION_MAP = {
            'Gamma (k, θ)': (['shape (k)', 'scale (θ)'], gamma_estimate_x),
            'Wald (m, a)': (['mean rate (m)', 'response threshold (a)'], wald_estimate_x),
            'Ex-Wald (m, a, τ)': (['mean rate (m)', 'response threshold (a)', 'τ'], ex_wald_estimate_x),
            'Shifted Wald (m, a, shift)': (['mean rate (m)', 'response threshold (a)', 'shift'], shift_wald_estimate_x),
            'Ex-Gaussian (μ, σ, τ)': (['mu (μ)', 'sigma (σ)', 'tau (τ)'], ex_gaussian_estimate_x),
            'Inv-Gaussian (μ, λ)': (['mu (μ)', 'lambda (λ)'], inverse_gaussian_estimate_x),
            'Shifted Inv-Gaussian (μ, λ, shift)': (
            ['mu (μ)', 'lambda (λ)', 'shift'], shifted_inverse_gaussian_estimate_x),
            'Weibull (k, θ)': (['shape (k)', 'scale (θ)'], weibull_estimate_x),
            'LogNormal (k, θ)': (['shape (k)', 'scale (θ)'], log_normal_estimate_x)}

    def savePsyData(self, file_path):
        self.data.to_csv(file_path, sep='|', quoting=csv.QUOTE_NONNUMERIC, index=False, header=True)

    def readPsyDataFiles(self, files: list):
        dfs = []
        for file in files:
            df = pd.read_csv(file, sep='|', quoting=csv.QUOTE_NONNUMERIC, index_col=False)

            dfs.append(df)

        self.data = pd.concat(dfs, ignore_index=True)

    def readDatFiles(self, fileList, containsHeader=True, encodingFormat='utf-8', delimiter='\s'):
        all_dfs = []

        for file in fileList:
            df = readDatFile(file, containsHeader, encodingFormat, delimiter)

            if df is not None:
                fileName = os.path.basename(file)
                if fileName not in df.columns:
                    df = df.assign(fileName=fileName)
                all_dfs.append(df)

        self.data = pd.concat(all_dfs, ignore_index=True)

    def readMatlabFiles(self, files, appendDataModel=False):
        # Pre-allocate a list to store DataFrames for better efficiency
        data_frames = []

        for file in files:
            mat_data = loadmat(file)
            variable = mat_data.get('allResults_APL')

            # Get column names from the first row: need to be confirmed
            # handle emtpy variable name in the first row
            columnList = []
            untitled_num = 1

            for line in variable[0]:
                for row in line:
                    if row and isinstance(row.flat[0], str):
                        columnList.append(row.flat[0])
                    else:
                        cTitle = f'untitled{untitled_num}'
                        untitled_num += 1
                        while cTitle in columnList:
                            cTitle = f'untitled{untitled_num}'
                            untitled_num += 1
                        columnList.append(f'untitled{untitled_num}')
            # columnList = [str(row.flat[0]) for line in variable[0] for row in line]

            # Convert to DataFrame, skipping the first row which contains column names
            df = pd.DataFrame(variable[1:], columns=columnList)
            # Apply the flatten function to all DataFrame elements
            df = df.applymap(flattenValue)

            # Add filename if not present
            if 'filename' not in columnList:
                df['filename'] = os.path.basename(file)
            # Append DataFrame to list
            data_frames.append(df)

        if appendDataModel:
            data_frames.append(self.data)

            # Concatenate all DataFrames at once for efficiency
        self.data = pd.concat(data_frames, ignore_index=True)

    def readMatlabFiles73(self, files, appendDataModel=False):
        # Pre-allocate a list to store DataFrames for better efficiency
        data_frames = []

        for file in files:
            mat_dat = mat73.loadmat(file)
            variable = mat_dat.get('allResults_APL', None)

            df = pd.DataFrame(variable[1:], columns=variable[0])
            df = df.applymap(flattenValueMat73)

            # Add filename if not present
            if 'filename' not in variable[1:]:
                df['filename'] = os.path.basename(file)

            # Append DataFrame to list
            data_frames.append(df)
            # Concatenate all DataFrames at once for efficiency
        if appendDataModel:
            data_frames.append(self.data)
        self.data = pd.concat(data_frames, ignore_index=True)

    def calculateVariable(self, target_variable_name, calculate_expression):
        self.data[target_variable_name] = eval(calculate_expression)

    def filterData(self, row_vars, col_vars, ruleList, omegaValues=None):
        if omegaValues is None:
            omegaValues = [-1 for _ in ruleList]

        tmpDataFrame = filterDataFunc(row_vars, col_vars, self.data, ruleList, omegaValues)
        return tmpDataFrame

    def summaryData(self, row_vars, col_vars, ruleList, target_vars, omegaValues):
        if omegaValues is None:
            omegaValues = [-1 for _ in ruleList]

        for target_var in target_vars:
            target_var_name, operation = target_var.split('@')
            if not pd.api.types.is_numeric_dtype(self.data[target_var_name]):
                self.data[target_var_name] = pd.to_numeric(self.data[target_var_name], errors='coerce')

        tmpDataFrame = filterDataFunc(row_vars, col_vars, self.data, ruleList, omegaValues)

        result_frame_var_names = []

        for target_var in target_vars:
            target_var_name, operation = target_var.split('@')

            result = None

            # tmpDataFrame[target_var_name] = pd.to_numeric(tmpDataFrame[target_var_name], errors='coerce')

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
                group_vars = row_vars + col_vars
                parameter_names, estimate_func = self.DISTRIBUTION_MAP[operation]

                if 'Wald' in operation:
                    print(f"Start to fit the data dist via {operation}...")
                    print(
                        f"See detailed info in Heathcote, (2004), Behavior Research Methods, Instruments & Computers, 36(4): 678-694 ...")
                else:
                    print(f"Start to fit the data dist via {operation}...")
                # Perform estimation (with or without grouping)
                if group_vars:
                    # Grouped estimation
                    grouped_result = tmpDataFrame.groupby(group_vars)[target_var_name].apply(estimate_func)
                else:
                    # Single estimation
                    grouped_result = pd.Series({'result': estimate_func(tmpDataFrame[target_var_name])})

                result = groupby_to_pivot_tables(grouped_result, row_vars, col_vars)

                target_var = [f"{target_var_name}@{operation} {item}" for item in parameter_names]

            if isinstance(target_var, list):
                result_frame_var_names.extend(target_var)
            else:
                result_frame_var_names.append(target_var)

            pd.set_option('display.float_format', lambda x: '%.10f' % x)

            if result is not None:
                if isinstance(result, list):
                    self.resultList.extend(result)
                else:
                    self.resultList.append(result)

        # print out rules and results info in command window
        print('filter rules:')
        print('========================')
        for rule in ruleList:
            print(rule)

        print('========================')

        print('results:')
        print('========================')

        for result in self.resultList:
            print(result)
