import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from app.rtDist import gamma_estimate_x, ex_wald_estimate_x, wald_estimate_x, log_normal_estimate_x, \
    weibull_estimate_x, ex_gaussian_estimate_x, inverse_gaussian_estimate_x, shifted_inverse_gaussian_estimate_x, \
    shift_wald_estimate_x


class FitRTsDistThread(QThread):
    # Signals for reporting fitting status and results
    # infoType, infoString, showTimeInfo or not
    fitStatus = pyqtSignal(int, str, bool)
    finished = pyqtSignal(object, list, list, list)

    # Mapping of distribution names to their configurations
    # Structure: {Display Name: (Internal Name, Parameter Names, Estimation Function)}
    DISTRIBUTION_MAP = {
        'Gamma (k, θ)': (['shape (k)', 'scale (θ)'], gamma_estimate_x),
        'Wald (m, a)': (['mean rate (m)', 'response threshold (a)'], wald_estimate_x),
        'Ex-Wald (m, a, τ)': (['mean rate (m)', 'response threshold (a)', 'τ'], ex_wald_estimate_x),
        'Shifted Wald (m, a, shift)': (['mean rate (m)', 'response threshold (a)', 'shift'], shift_wald_estimate_x),
        'Ex-Gaussian (μ, σ, τ)': (['mu (μ)', 'sigma (σ)', 'tau (τ)'], ex_gaussian_estimate_x),
        'Inv-Gaussian (μ, λ)': (['mu (μ)', 'lambda (λ)'], inverse_gaussian_estimate_x),
        'Shifted Inv-Gaussian (μ, λ, shift)': (['mu (μ)', 'lambda (λ)', 'shift'], shifted_inverse_gaussian_estimate_x),
        'Weibull (k, θ)': (['shape (k)', 'scale (θ)'], weibull_estimate_x),
        'LogNormal (k, θ)': (['shape (k)', 'scale (θ)'], log_normal_estimate_x)
    }

    def __init__(self, dataFrame, operation, row_vars, col_vars, independentVarName, distribution, parent=None):
        """
        Initialize the fitting thread with necessary parameters.

        Args:
            dataFrame (pd.DataFrame): Input data for distribution fitting
            operation (str): Type of operation being performed
            row_vars (list): Row grouping variables
            col_vars (list): Column grouping variables
            independentVarName (str): Name of the independent variable
            distribution (str): Distribution to fit
            parent (QObject, optional): Parent QObject
        """
        super().__init__(parent)
        self.dataFrame = dataFrame
        self.operation = operation
        self.row_vars = row_vars
        self.col_vars = col_vars
        self.independentVarName = independentVarName
        self.distribution = distribution

    def run(self):
        """
        Main thread execution method.
        Handles distribution fitting and error management.
        """
        try:
            # Process the distribution fitting
            self._process_distribution()

        except Exception as e:
            # Emit error status if fitting fails
            self.fitStatus.emit(2, f'Fitting error: {str(e)}', True)
            self.finished.emit(None, [], self.row_vars, self.col_vars)

    def _process_distribution(self):
        """
        Core method for processing distribution fitting.

        Returns:
            pd.Series: Fitted distribution results

        Workflow:
        1. Validate distribution
        2. Prepare grouping variables
        3. Perform estimation
        4. Generate parameter names
        5. Emit results
        """
        # Emit start status
        # self.fitStatus.emit(0, f'Start to fit the distribution via {self.distribution}...', False)

        # Validate and retrieve distribution configuration
        dist_key = self.distribution
        if dist_key not in self.DISTRIBUTION_MAP:
            self.fitStatus.emit(2, f'Invalid distribution parameter: {dist_key}.', True)
            self.finished.emit(None, [], self.row_vars, self.col_vars)
            return

        # Unpack distribution details
        parameter_names, estimate_func = self.DISTRIBUTION_MAP[dist_key]

        # Special handling for Ex-Wald (known to be slow)
        if 'Wald' in dist_key:
            self.fitStatus.emit(0, f"Start to fit the data dist via {dist_key}...", False)
            self.fitStatus.emit(0, f'See detailed info in Heathcote, (2004), <i>Behavior Research Methods, Instruments & Computers</i>, 36(4): 678-694 ...', False)
        else:
            self.fitStatus.emit(0, f"Start to fit the data dist via {dist_key}...", False)

        prepared_frame = self._prepare_fit_dataframe()

        # Prepare grouping variables
        group_vars = self.row_vars + self.col_vars

        # Perform estimation (with or without grouping)
        if group_vars:
            # Grouped estimation
            grouped_result = prepared_frame.groupby(group_vars)[self.independentVarName].apply(estimate_func)
        else:
            # Single estimation
            grouped_result = pd.Series({'result': estimate_func(prepared_frame[self.independentVarName])})

        if grouped_result.empty:
            raise ValueError('No fit results were generated for the current filters.')

            # Generate fully qualified parameter names
        full_parameter_names = [f"{self.independentVarName}@{self.operation} {item}" for item in parameter_names]

        # Emit final results
        self.finished.emit(grouped_result, full_parameter_names, self.row_vars, self.col_vars)

    def _prepare_fit_dataframe(self):
        required_columns = []
        for column_name in self.row_vars + self.col_vars + [self.independentVarName]:
            if column_name not in required_columns:
                required_columns.append(column_name)

        target_values = pd.to_numeric(self.dataFrame[self.independentVarName], errors='coerce')
        valid_mask = target_values.notna() & np.isfinite(target_values)

        prepared_frame = self.dataFrame.loc[valid_mask, required_columns].copy()
        prepared_frame[self.independentVarName] = target_values.loc[valid_mask].to_numpy()

        if prepared_frame.empty:
            raise ValueError(
                f"No valid numeric data remain in '{self.independentVarName}' after filtering.")

        return prepared_frame
