"""Background fitting worker for PsySummary cognitive RT models."""

import numpy as np
import pandas as pd
from copy import deepcopy
from PyQt5.QtCore import QThread, pyqtSignal

from app.cognitiveModels import fit_cognitive_model
from app.fitCancellation import FitCancelled, raise_if_fit_cancelled
from app.cognitiveModelSpec import (
    ACCURACY_CODING, BOUNDARY_CODING_LABELS, RATCLIFF_MODEL, RESPONSE_CODING,
    cognitive_model_reference_text, validate_model_data, model_result_parameters,
    ValidatedModelData,
)


COGNITIVE_DIAGNOSTIC_NAMES = [
    'N valid', 'Converged', 'Response counts', 'Log-likelihood', 'AIC', 'BIC',
]


class FitCognitiveModelThread(QThread):
    """Fit one structured model independently within each Rows/Columns group."""

    fitStatus = pyqtSignal(int, str, bool)
    finished = pyqtSignal(object, list, list, list, object)
    cancelled = pyqtSignal()
    conditionProgress = pyqtSignal(int, int)

    def __init__(self, dataframe, specification, row_vars, col_vars, parent=None,
                 validation_receipt=None):
        """Initialize the grouped cognitive-model worker."""
        super().__init__(parent)
        self.dataframe = dataframe
        self.specification = deepcopy(specification)
        self.row_vars = list(row_vars)
        self.col_vars = list(col_vars)
        self._validation_receipt = validation_receipt

    def run(self):
        """Fit all groups and emit result arrays plus diagnostics records."""
        try:
            self._process_model()
        except FitCancelled:
            self.cancelled.emit()
        except Exception as error:
            self.fitStatus.emit(2, f'Cognitive model fitting error: {error}', True)
            self.finished.emit(None, [], self.row_vars, self.col_vars, [])

    def _process_model(self):
        """Prepare groups, fit each one, and emit table-ready results."""
        cancel_check = self.isInterruptionRequested
        raise_if_fit_cancelled(cancel_check)
        specification = self.specification
        model = specification['model']
        self.fitStatus.emit(0, cognitive_model_reference_text(model), False)
        if model == RATCLIFF_MODEL:
            coding = specification.get('boundary_coding', RESPONSE_CODING)
            mapping = specification.get('response_mapping', {})
            details = (
                'Accuracy boundaries are fitted; z and d are mirrored when the correct physical '
                'response is lower.' if coding == ACCURACY_CODING else
                'Mapped response boundaries are fitted; v is reversed when the correct physical '
                'response is lower.')
            self.fitStatus.emit(
                0, f'Coding scheme: {BOUNDARY_CODING_LABELS.get(coding, coding)}; '
                   f'response mapping={mapping}. {details}', False)
        group_vars = self.row_vars + self.col_vars
        if not (isinstance(self._validation_receipt, ValidatedModelData)
                and self._validation_receipt.matches(specification, self.dataframe, group_vars)):
            validate_model_data(specification, self.dataframe, group_vars)
        required = list(dict.fromkeys(
            group_vars + [specification['rt_variable'], specification['response_variable']]
            + ([specification['accuracy_variable']] if specification.get('accuracy_variable') else [])))
        missing = [name for name in required if name not in self.dataframe.columns]
        if missing:
            raise ValueError(f"Required model variable(s) are missing: {', '.join(missing)}.")
        prepared = self.dataframe[required].copy()
        fit_records = []
        grouped = prepared.groupby(group_vars, dropna=False, sort=False, observed=True) if group_vars else None
        total_groups = grouped.ngroups if grouped is not None else 1
        current_group = 0

        def fit_group(group_frame, group_values=()):
            nonlocal current_group
            raise_if_fit_cancelled(cancel_check)
            current_group += 1
            self.conditionProgress.emit(current_group, total_groups)
            fit = fit_cognitive_model(
                group_frame, specification, validate=False,
                cancel_check=cancel_check)
            raise_if_fit_cancelled(cancel_check)
            fit['group_vars'] = list(group_vars)
            fit['group_values'] = group_values
            fit['result_prefix'] = f"{specification['rt_variable']}@{model}"
            fit['distribution'] = model
            fit['data'] = fit['rt']
            fit_records.append(fit)
            group_label = 'Overall' if not group_vars else ', '.join(
                f'{name}={value}' for name, value in zip(group_vars, group_values))
            self.fitStatus.emit(
                0,
                f"Fit diagnostics [{group_label}]: Converged={'Yes' if fit['converged'] else 'No'}; "
                f"N={fit['n_valid']}; responses={fit['response_counts']}; "
                f"LL={fit['log_likelihood']:.6g}; AIC={fit['aic']:.6g}; BIC={fit['bic']:.6g}; "
                f"seed={fit['random_seed']}; optimizer={fit['optimizer_method']} "
                f"({fit['message']})",
                False,
            )
            diagnostics = [
                fit['n_valid'],
                'Yes' if fit['converged'] else 'No',
                str(fit['response_counts']),
            ]
            if specification.get('accuracy_variable'):
                diagnostics.append(fit['accuracy_rate'])
            diagnostics.extend([fit['log_likelihood'], fit['aic'], fit['bic']])
            return np.asarray(list(fit['parameters']) + diagnostics, dtype=object)

        if group_vars:
            results = {}
            for group_key, group_frame in grouped:
                group_values = group_key if isinstance(group_key, tuple) else (group_key,)
                results[group_key] = fit_group(group_frame, group_values)
            grouped_result = pd.Series(results)
            grouped_result.index.names = group_vars
        else:
            grouped_result = pd.Series({'result': fit_group(prepared)})

        parameter_labels = [
            f"{parameter['name']} ({parameter['mode'].capitalize()})"
            for parameter in model_result_parameters(specification)
        ]
        prefix = f"{specification['rt_variable']}@{model}"
        diagnostic_names = list(COGNITIVE_DIAGNOSTIC_NAMES)
        if specification.get('accuracy_variable'):
            diagnostic_names.insert(3, 'Accuracy rate')
        output_names = [f'{prefix} {name}' for name in parameter_labels + diagnostic_names]
        raise_if_fit_cancelled(cancel_check)
        self.finished.emit(grouped_result, output_names, self.row_vars, self.col_vars, fit_records)
