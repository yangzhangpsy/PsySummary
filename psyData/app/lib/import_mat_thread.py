import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

from PyQt5.QtCore import QThread, pyqtSignal


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
        if len(value[0]) > 1:
            return ','.join(map(str, value[0]))
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
    # elif isinstance(value, str):
    #     return value
    # elif isinstance(value, list) and len(value) > 0:
    #     return value[0]
    return value


class ImportMatThread(QThread):
    # 0 none
    #  1 success
    #  2 fail
    #  3 compile error
    #  4 warning
    readStatus = pyqtSignal(int, str)
    finished = pyqtSignal(pd.DataFrame, int, list, bool)  # Used for passing the loaded data

    #  0 1 2 for success, fail (has no allResults_APL variable within the mat file),
    def __init__(self, fileList, fileType=1, appendDataModel=False, parent=None):
        super(ImportMatThread, self).__init__(parent)
        if isinstance(fileList, str):
            self.files = [fileList]
        else:
            self.files = fileList
        self.type = fileType
        self.append_data_model = appendDataModel

    # def __del__(self):
    #     self.wait()

    def run(self):
        self.readData()
        self.quit()

    def readData(self):

        try:
            if self.type == 1:
                df = self.readMatlabFiles()
            else:
                df = self.readMatlabFiles73()

            if df.size > 0:
                self.finished.emit(df, self.type, self.files, self.append_data_model)

        except Exception as e:
            self.readStatus.emit(2, f"IOError: {e}")

    def readMatlabFiles(self):
        # Pre-allocate a list to store DataFrames for better efficiency
        data_frames = []
        combined_df = pd.DataFrame()

        for file in self.files:
            if file.endswith('.mat'):
                try:
                    mat_data = loadmat(file)
                    variable = mat_data.get('allResults_APL', None)
                    if variable is None:
                        self.readStatus.emit(2, f"Failed to find 'allResults_APL' in {os.path.basename(file)} !")

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
                    self.readStatus.emit(0, f"Reading file: {file}")
                except Exception as e:
                    self.readStatus.emit(2, f"IOError: {e}")

            # Concatenate all DataFrames at once for efficiency
        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)

        return combined_df

    def readMatlabFiles73(self):
        import mat73
        # Pre-allocate a list to store DataFrames for better efficiency
        data_frames = []
        combined_df = pd.DataFrame()

        for file in self.files:
            if file.endswith('.mat'):
                try:
                    mat_dat = mat73.loadmat(file)
                    variable = mat_dat.get('allResults_APL', None)
                    if variable is None:
                        self.readStatus.emit(2,
                                             f"Failed to find variable allResults_APL in {os.path.basename(file)}.mat !")
                        continue

                    df = pd.DataFrame(variable[1:], columns=variable[0])
                    # df = df.applymap(flattenValueMat73)
                    df = df.map(flattenValueMat73)
                    # df = df.apply(lambda col: col.map(flattenValueMat73))

                    # Add filename if not present
                    if 'filename' not in variable[1:]:
                        df['filename'] = os.path.basename(file)

                    # Append DataFrame to list
                    data_frames.append(df)

                    self.readStatus.emit(0, f"Reading file: {file}")
                except Exception as e:
                    self.readStatus.emit(2, f"IOError: {e}")
            # Concatenate all DataFrames at once for efficiency
        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)

        return combined_df
