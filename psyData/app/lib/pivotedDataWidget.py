from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QPushButton, QApplication, QFileDialog, QHBoxLayout, QSpinBox, QMenu
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from app.dataPreparation import prepare_summary_frame, safe_mode

from app.psyDataFunc import PsyDataFunc as Func
from app.lib.fitRTsDistThread import FitRTsDistThread
from app.lib.fitCognitiveModelThread import FitCognitiveModelThread
from app.cognitiveModelSpec import (
    COGNITIVE_MODEL_NAMES, split_target, resolve_analysis_seeds, ValidatedModelData,
)
from app.psyDataFunc import PsyDataFunc
from app.tool import StatisticTool, FlashMessageBox, warnConditionWiseFiltering
from app.lib.dataFrameTableWidget import ResultFrameTableWidget
from app.lib.rtFitDiagnostics import RTFitDiagnosticsDialog
from app.diagnosticCurveCache import DiagnosticCurveCache


RT_FIT_METHODS = (
    'Gamma (k, θ)',
    'Shifted Gamma (k, θ, shift)',
    'Weibull (k, θ)',
    'Shifted Weibull (k, θ, shift)',
    'LogNormal (k, θ)',
    'Shifted LogNormal (k, θ, shift)',
    'Wald (m, a)',
    'Ex-Wald (m, a, τ)',
    'Shifted Wald (m, a, shift)',
    'Ex-Gaussian (μ, σ, τ)',
    'Inv-Gaussian (μ, λ)',
    'Shifted Inv-Gaussian (μ, λ, shift)',
)
MODEL_FIT_METHODS = frozenset(RT_FIT_METHODS + tuple(COGNITIVE_MODEL_NAMES))


def getStandardError(x):
    """Calculate sample standard error using only non-missing observations."""
    return pd.Series(x).sem(skipna=True, ddof=1)


def groupby_to_pivot_tables(grouped_result, index_var=None, columns_var=None):
    """
    Convert grouped results into multiple pivot-table-like DataFrames.

    Parameters:
    - grouped_result: Result of groupby operation
    - index_var: Variable name for pivot table rows (optional)
    - columns_var: Variable name for pivot table columns (optional)

    Returns: Dictionary of DataFrames
    """
    if grouped_result is None:
        return []

    # Convert grouped result to DataFrame
    # Handle both Series and DataFrame inputs
    if isinstance(grouped_result, pd.Series):
        if grouped_result.empty:
            return []
        df = grouped_result.apply(pd.Series)
    else:
        df = grouped_result.copy()

    if isinstance(df, pd.Series) or df.empty or not hasattr(df, 'columns'):
        return []

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
                aggfunc='first', observed=True
            )
        elif index_var and not columns_var:
            # Only index variable specified
            # Use pivot_table to get index-wise aggregation
            pivot_table = df_reset.pivot_table(
                index=index_var,
                columns=None,
                values=col,
                aggfunc='first', observed=True
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
        target_var_name, operation, specification = split_target(target_var)
        allVariables = rowAndColVars.copy()
        allVariables.append(target_var_name)
        if specification:
            allVariables.append(specification['response_variable'])
            if specification.get('accuracy_variable'):
                allVariables.append(specification['accuracy_variable'])

        if len(allVariables) != len(set(allVariables)):
            seen = set()
            duplicates = set([x for x in allVariables if x in seen or seen.add(x)])

            raise Exception(
                f'Using a variable across multiple lists within Rows, Columns, or Data list is not allowed.\n'
                f'Please remove the multiple used variable(s) {duplicates} and retry!')


class ModelPreparationError(ValueError):
    """Report all invalid cognitive targets before starting the fitting queue."""


class PreparedAnalysis:
    """Own one run's filtered frame, resolved settings, and validation receipts."""

    def __init__(self, dataframe, row_vars, col_vars, target_vars, rule_list, *,
                 log=None, cdf_decider=None, check_cancelled=None, progress=None):
        check = check_cancelled or (lambda: None)
        self.source = dataframe
        self.row_vars = list(row_vars)
        self.col_vars = list(col_vars)
        self.target_vars = resolve_analysis_seeds(target_vars)
        self.rule_list = list(rule_list)
        self.filter_script_lines = []
        self.validation_receipts = {}
        if self.rule_list:
            self.dataframe = StatisticTool.filterData(
                self.row_vars, self.col_vars, dataframe, self.rule_list,
                record_script=True, script_collector=self.filter_script_lines,
                log=log, cdf_decider=cdf_decider, check_cancelled=check, progress=progress)
        else:
            StatisticTool.checkEmptyNullValue(dataframe, self.row_vars, self.col_vars)
            self.dataframe = dataframe
            self.filter_script_lines.append('cdfPoolingOmegas = []')
        if self.dataframe.empty:
            message = 'No data remain after applying the current filters. Analysis was skipped.'
            (log or PsyDataFunc.printOut)(message, 4)
            raise ValueError(message)
        problems = []
        for index, target in enumerate(self.target_vars):
            check()
            variable, model, specification = split_target(target)
            if model not in COGNITIVE_MODEL_NAMES:
                continue
            try:
                if progress:
                    progress(f'Validating model {index + 1} of {len(self.target_vars)}…')
                if not specification:
                    raise ValueError('Open Model Settings to configure this model.')
                self.validation_receipts[index] = ValidatedModelData(
                    specification, self.dataframe, self.row_vars + self.col_vars)
            except (TypeError, ValueError, KeyError) as error:
                problems.append(f'{variable}@{model}:\n{error}')
        if problems:
            raise ModelPreparationError('\n\n'.join(problems))
        checkVariablesDuplication(self.row_vars, self.col_vars, self.target_vars)
        check()

    def matches(self, source, row_vars, col_vars, target_vars, rule_list):
        """Reject preparation reuse for another source or a changed run configuration."""
        return (source is self.source and list(row_vars) == self.row_vars
                and list(col_vars) == self.col_vars and list(target_vars) == self.target_vars
                and list(rule_list) == self.rule_list)


def generateScript(row_vars, col_vars, target_vars, ruleList, filter_script_lines=()):
    if any(isinstance(target, dict) for target in target_vars):
        target_script = f'targetVariables = {target_vars!r}'
    else:
        target_script = PsyDataFunc.list2Script(target_vars, 'targetVariables')
    analysis_script = list(filter_script_lines) + [PsyDataFunc.list2Script(row_vars, 'rowVariables'),
                       PsyDataFunc.list2Script(col_vars, 'colVariables'),
                       PsyDataFunc.list2Script(ruleList, 'ruleList'),
                       target_script,
                       "aggData.summaryData(rowVariables, colVariables, ruleList, targetVariables, cdfPoolingOmegas)"]

    PsyDataFunc.genScript(analysis_script)


def handleFitThreadSignal(infoType: int, InfoStr: str, ShowTime: bool = True):
    PsyDataFunc.printOut(InfoStr, infoType, ShowTime)


class PivotedDataWidget(QWidget):
    analysisFinished = pyqtSignal()
    analysisFailed = pyqtSignal(str)
    analysisCancelled = pyqtSignal()
    analysisProgress = pyqtSignal(int, int, str)
    analysisStage = pyqtSignal(str)
    loadResultsRequested = pyqtSignal()
    saveResultsRequested = pyqtSignal()

    def __init__(self, dataframe, row_vars, col_vars, target_vars, ruleList, parent=None,
                 prepared_analysis=None, snapshot=None):
        super(PivotedDataWidget, self).__init__(parent)
        self.fit_dist_thread = None
        self.table = None
        self.msg_box = None
        self.resultList = []
        self.filterStr = ''
        self.ruleList = list(ruleList)
        self.result_frame_var_names = []
        self.fit_error_message = None
        self.fit_records = []
        self._curve_cache = DiagnosticCurveCache()
        self.fit_diagnostics_dialog = None
        self.fitMethods = list(RT_FIT_METHODS)
        self.cognitiveFitMethods = list(COGNITIVE_MODEL_NAMES)
        self._row_vars = list(row_vars)
        self._col_vars = list(col_vars)
        self._loaded_snapshot = snapshot
        self._result_metadata = (dict(snapshot.get('metadata', {})) if snapshot is not None else {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'software_version': QApplication.applicationVersion() or 'unreported (source build)',
            'pandas_version': pd.__version__, 'numpy_version': np.__version__})
        if snapshot is not None:
            # A historical result must never enter preparation, script recording or fitting.
            self._prepared_analysis = None
            self._target_vars = snapshot['targets']
            self._target_index = len(self._target_vars)
            self._fit_started_count = self._fit_target_count = 0
            self._active_fit_label = None
            self._analysis_complete = True
            self._cancel_requested = False
            self._tmp_dataframe = None
            self.resultList = snapshot['results']
            self.result_frame_var_names = snapshot['labels']
            self.fit_records = snapshot['fit_records']
            self.initUI(None, row_vars, col_vars, self._target_vars)
            return
        if prepared_analysis is None:
            prepared_analysis = PreparedAnalysis(dataframe, row_vars, col_vars, target_vars, ruleList)
        elif not (isinstance(prepared_analysis, PreparedAnalysis)
                  and prepared_analysis.matches(dataframe, row_vars, col_vars, target_vars, ruleList)):
            raise ValueError('The prepared analysis does not match this data or configuration.')
        self._prepared_analysis = prepared_analysis
        self._target_vars = prepared_analysis.target_vars
        self._target_index = 0
        self._fit_started_count = 0
        self._fit_target_count = sum(
            split_target(target)[1] in MODEL_FIT_METHODS for target in target_vars)
        self._active_fit_label = None
        self._analysis_complete = False
        self._cancel_requested = False
        self._tmp_dataframe = prepared_analysis.dataframe

        self.initUI(dataframe, row_vars, col_vars, self._target_vars)

    def fitDistInBackground(self, dataFrame, operation, row_vars, col_vars, independentVarName,
                            distribution='Ex-Gaussian'):
        self.fit_dist_thread = FitRTsDistThread(
            dataFrame, operation, row_vars, col_vars, independentVarName,
            distribution, parent=self, curve_cache=self._curve_cache)

        self.fit_dist_thread.fitStatus.connect(self.handleFitStatus)
        self.fit_dist_thread.conditionProgress.connect(self._conditionFitProgress)
        self.fit_dist_thread.curvePreparationProgress.connect(self._curvePreparationProgress)
        self.fit_dist_thread.resultReady.connect(self.handleFitFinished)
        self.fit_dist_thread.cancelled.connect(self.handleFitCancelled)

        self.fit_dist_thread.start()

    def fitCognitiveModelInBackground(self, dataFrame, specification, row_vars, col_vars):
        """Start one grouped cognitive-model fitting worker."""
        self.fit_dist_thread = FitCognitiveModelThread(
            dataFrame, specification, row_vars, col_vars, parent=self,
            validation_receipt=self._prepared_analysis.validation_receipts.get(self._target_index),
            curve_cache=self._curve_cache)
        self.fit_dist_thread.fitStatus.connect(self.handleFitStatus)
        self.fit_dist_thread.conditionProgress.connect(self._conditionFitProgress)
        self.fit_dist_thread.curvePreparationProgress.connect(self._curvePreparationProgress)
        self.fit_dist_thread.resultReady.connect(self.handleFitFinished)
        self.fit_dist_thread.cancelled.connect(self.handleFitCancelled)
        self.fit_dist_thread.start()

    def initUI(self, dataframe, row_vars, col_vars, target_vars):
        self.setWindowIcon(Func.getImageObject("icon.png", type=1))
        self.setWindowTitle("Aggregation Results")
        self.resize(400, 700)

        self.all_layout = QVBoxLayout(self)
        self.results_content = QWidget(self)
        self.content_layout = QVBoxLayout(self.results_content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.results_footer = QWidget(self)
        self.btns_layout = QHBoxLayout(self.results_footer)
        self.btns_layout.setContentsMargins(0, 0, 0, 0)

        self.clipboard_button = QPushButton('Clipboard', self.results_footer)
        self.export_button = QPushButton('Export', self.results_footer)
        self.load_results_button = QPushButton('Load Results…', self.results_footer)
        self.load_results_button.setToolTip('Open a saved .psyresult without changing the current data or setup.')
        self.load_results_button.clicked.connect(lambda _checked=False: self.loadResultsRequested.emit())

        # Create a QSpinBox for controlling decimal places
        self.decimal_spin_box = QSpinBox(self.results_footer)
        self.decimal_spin_box.setRange(0, 12)
        self.decimal_spin_box.setValue(4)
        self.decimal_spin_box.setSuffix(" decimal places")
        self.decimal_spin_box.valueChanged.connect(self.update_table)

        self.clipboard_button.setFixedWidth(100)
        self.export_button.setFixedWidth(100)

        self.clipboard_button.clicked.connect(self.copyToClipboard)
        export_menu = QMenu(self.export_button)
        export_menu.addAction('Save Results (.psyresult)…', lambda _checked=False: self.saveResultsRequested.emit())
        export_menu.addAction('Export Table (.txt)…', self.exportData)
        self.export_button.setMenu(export_menu)

        self.btns_layout.addWidget(self.clipboard_button)
        self.btns_layout.addWidget(self.export_button)
        self.btns_layout.addWidget(self.load_results_button)
        self.btns_layout.addWidget(self.decimal_spin_box)
        self.clipboard_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.decimal_spin_box.setEnabled(False)

        filters_Info = '\n'.join(self.ruleList)

        self.filterStr = filters_Info
        if self._loaded_snapshot is not None:
            path = self._loaded_snapshot.get('_loaded_path', '')
            label = QLabel('Loaded results: ' + Path(path).name)
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            label.setToolTip(path + '\nSaved: ' + self._loaded_snapshot.get('saved_at', ''))
            self.all_layout.addWidget(label)
        filter_label = QLabel(filters_Info)
        filter_label.setTextFormat(Qt.PlainText)
        self.all_layout.addWidget(filter_label)
        self.all_layout.addWidget(self.results_content, 1)
        self.all_layout.addWidget(self.results_footer)

        if self._loaded_snapshot is not None:
            self.createResultTable(col_vars, row_vars)
            self.decimal_spin_box.setValue(self._loaded_snapshot['decimals'])
            self.update_table()
            return

        try:
            if hasattr(self._prepared_analysis, 'condition_logs'):
                for message, kind in self._prepared_analysis.condition_logs:
                    PsyDataFunc.printOut(message, kind)
            else:
                warnConditionWiseFiltering(row_vars, col_vars, dataframe, self.ruleList)
            generateScript(row_vars, col_vars, target_vars, self.ruleList,
                           self._prepared_analysis.filter_script_lines)
            if self._fit_target_count:
                QTimer.singleShot(0, self._processNextTarget)
            else:
                self._processNextTarget(raise_errors=True)
        except Exception as e:
            raise Exception(e)

    def _processNextTarget(self, raise_errors=False):
        """Process synchronous targets until the next model fit, then yield to its worker."""
        if self._analysis_complete:
            return
        if self._cancel_requested:
            self._finishCancellation()
            return
        try:
            while self._target_index < len(self._target_vars):
                target_var = self._target_vars[self._target_index]
                target_var_name, operation, specification = split_target(target_var)
                if operation in MODEL_FIT_METHODS:
                    self._startModelFit(target_var_name, operation, specification)
                    return

                result = self._calculateSummaryResult(target_var_name, operation)
                self.updateResultDataframe(result, f'{target_var_name}@{operation}')
                self._target_index += 1

            self._finishAnalysis()
        except Exception as error:
            if raise_errors:
                raise
            self._failAnalysis(str(error))

    def _calculateSummaryResult(self, target_var_name, operation):
        """Calculate one non-model summary on the prepared filtered data."""
        dataframe = prepare_summary_frame(self._tmp_dataframe, target_var_name, operation)
        if operation == 'Mean':
            aggregate = 'mean'
        elif operation == 'Median':
            aggregate = 'median'
        elif operation == 'Mode':
            aggregate = safe_mode
        elif operation == 'Count':
            aggregate = 'count'
        elif operation == 'Standard Deviation':
            aggregate = 'std'
        elif operation == 'Max':
            aggregate = 'max'
        elif operation == 'Min':
            aggregate = 'min'
        elif operation == 'Variance':
            aggregate = 'var'
        elif operation == 'Standard Error':
            aggregate = getStandardError
        else:
            raise ValueError(f'Unsupported summary operation: {operation}.')

        series = dataframe[target_var_name]
        if not self._row_vars and not self._col_vars:
            if callable(aggregate):
                return aggregate(series)
            if aggregate == 'count':
                return series.count()
            options = {'skipna': True}
            if aggregate in {'std', 'var'}:
                options['ddof'] = 1
            return getattr(series, aggregate)(**options)
        return pd.pivot_table(
            dataframe, index=self._row_vars, columns=self._col_vars,
            values=target_var_name, aggfunc=aggregate, observed=True)

    def _conditionFitProgress(self, current, total):
        """Forward sparse group progress only from the currently active worker."""
        if self.sender() is self.fit_dist_thread and not self._cancel_requested:
            self.analysisProgress.emit(
                self._fit_started_count, self._fit_target_count,
                f'{self._active_fit_label} · Condition {current} of {total}')

    def _curvePreparationProgress(self, current, total):
        """Distinguish post-fit curve preparation from parameter optimization."""
        if self.sender() is self.fit_dist_thread and not self._cancel_requested:
            self._conditionFitProgress(current, total)
            self.analysisStage.emit('Preparing diagnostic curves…')
            handleFitThreadSignal(0, f'Preparing diagnostic curves: {self._active_fit_label} '
                                 f'· Condition {current} of {total}…', False)

    def _startModelFit(self, target_var_name, operation, specification):
        """Start one model worker and return immediately to the Qt event loop."""
        if self.fit_dist_thread is not None:
            raise RuntimeError('A model fitting worker is already active.')
        if operation in self.fitMethods:
            self.validateFitInput(self._tmp_dataframe, target_var_name, operation)
        elif not specification:
            raise ValueError(
                f"{target_var_name}@{operation} has no model settings. "
                'Double-click the Data item and configure the model first.')

        self.fit_error_message = None
        self._fit_started_count += 1
        self._active_fit_label = f'{target_var_name} @ {operation}'
        self.analysisProgress.emit(
            self._fit_started_count, self._fit_target_count,
            self._active_fit_label)
        if self._fit_started_count == 1:
            handleFitThreadSignal(
                0, f'Model fitting started: {self._fit_target_count} model(s) queued.', False)
        progress = (
            f'Fitting {self._fit_started_count}/{self._fit_target_count}: '
            f'{self._active_fit_label}…')
        handleFitThreadSignal(0, progress, False)

        if operation in self.fitMethods:
            self.fitDistInBackground(
                self._tmp_dataframe, operation, self._row_vars, self._col_vars,
                target_var_name, operation)
        else:
            self.fitCognitiveModelInBackground(
                self._tmp_dataframe, specification, self._row_vars, self._col_vars)

    def _finishAnalysis(self):
        """Build the final table and announce completion after every target is processed."""
        if self._analysis_complete:
            return
        if not self.resultList:
            raise ValueError('No valid results were generated for the current selection.')
        pd.set_option('display.float_format', lambda x: '%.10f' % x)
        self.createResultTable(self._col_vars, self._row_vars)
        self._analysis_complete = True
        if self._fit_target_count:
            handleFitThreadSignal(1, 'Model fitting finished.', False)
        self.analysisFinished.emit()

    def _failAnalysis(self, message):
        """Show and emit one terminal asynchronous-analysis failure."""
        if self._analysis_complete:
            return
        self._analysis_complete = True
        handleFitThreadSignal(2, f'Model fitting failed: {message}', False)
        self._curve_cache.close()
        self.analysisFailed.emit(message)

    def createResultTable(self, col_vars, row_vars):
        self.table = ResultFrameTableWidget(
            self.resultList, col_vars, row_vars, self.result_frame_var_names, self.fit_records)
        self.table.fitRecordActivated.connect(self.showFitDiagnostics)

        self.content_layout.addWidget(self.table)
        self.clipboard_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.decimal_spin_box.setEnabled(True)

    def hasResults(self):
        """Return whether a completed result table is available for publication."""
        return self._analysis_complete and self.table is not None and bool(self.resultList)

    def handleFitFinished(self, result, target_var, row_vars, col_vars, fit_records):
        """Collect one worker result, then continue the asynchronous target queue."""
        worker = self.fit_dist_thread
        if result is not None:
            result = groupby_to_pivot_tables(result, row_vars, col_vars)
            if result:
                self.updateResultDataframe(result, target_var)
                self.fit_records.extend(fit_records)
            elif not self.fit_error_message:
                self.fit_error_message = 'No valid fit results were generated for the current filters.'

        if worker is not None:
            worker.deleteLater()
        self.fit_dist_thread = None

        if self._cancel_requested:
            self._finishCancellation()
            return

        if result is None or self.fit_error_message:
            message = self.fit_error_message or 'No model fit result was returned.'
            self._failAnalysis(message)
            return

        handleFitThreadSignal(
            1,
            f'Finished {self._fit_started_count}/{self._fit_target_count}: '
            f'{self._active_fit_label}.',
            False,
        )
        self._target_index += 1
        QTimer.singleShot(0, self._processNextTarget)

    def cancelAnalysis(self):
        """Request cooperative cancellation of the active model-fitting queue."""
        if self._analysis_complete or self._cancel_requested:
            return
        self._cancel_requested = True
        worker = self.fit_dist_thread
        if worker is not None:
            worker.requestInterruption()
        else:
            QTimer.singleShot(0, self._finishCancellation)

    def handleFitCancelled(self):
        """Clean up a cooperatively cancelled worker and cancel the full queue."""
        worker = self.fit_dist_thread
        if worker is not None:
            worker.deleteLater()
        self.fit_dist_thread = None
        self._finishCancellation()

    def _finishCancellation(self):
        """Emit one terminal cancellation signal without reporting a fitting error."""
        if self._analysis_complete:
            return
        self._analysis_complete = True
        self._curve_cache.close()
        self.analysisCancelled.emit()

    def showFitDiagnostics(self, selected_record=None):
        """Open fitted PDF and CDF diagnostics for a result-table cell."""
        if not self.fit_records:
            return
        initial_index = 0
        if selected_record is not None:
            initial_index = next(
                (index for index, record in enumerate(self.fit_records) if record is selected_record), 0)
        self.fit_diagnostics_dialog = RTFitDiagnosticsDialog(
            self.fit_records, self, initial_index=initial_index)
        self.fit_diagnostics_dialog.show()

    def handleFitStatus(self, infoType: int, infoStr: str, showTime: bool = True):
        if infoType >= 2:
            self.fit_error_message = infoStr
        handleFitThreadSignal(infoType, infoStr, showTime)

    def validateFitInput(self, dataFrame, target_var_name, operation):
        if dataFrame.empty:
            raise ValueError(
                f"No data remain after applying the current filters, so {operation} cannot be fitted.")

        numeric_values = pd.to_numeric(dataFrame[target_var_name], errors='coerce')
        numeric_values = numeric_values[np.isfinite(numeric_values)]
        if numeric_values.empty:
            raise ValueError(
                f"No valid numeric data remain in '{target_var_name}' after filtering, so {operation} cannot be fitted.")
        if operation.startswith('Shifted ') and len(numeric_values) < 4:
            raise ValueError(
                f"At least four valid observations are required to fit {operation}.")
        if operation.startswith('Shifted ') and (numeric_values <= 0).any():
            raise ValueError(
                f"{operation} requires strictly positive RT observations.")

    @classmethod
    def fromSnapshot(cls, snapshot, parent=None):
        """Construct a read-only historical result without running an analysis."""
        from app.resultArchive import validate_snapshot
        validate_snapshot(snapshot)
        return cls(None, snapshot['rows'], snapshot['columns'], snapshot['targets'], snapshot['rules'],
                   parent=parent, snapshot=snapshot)

    def resultSnapshot(self):
        """Capture display settings and retain immutable result references for background saving."""
        if not self._analysis_complete or not self.resultList:
            raise ValueError('Wait for a complete result before saving.')
        return {'results': self.resultList, 'labels': list(self.result_frame_var_names),
                'rows': list(self._row_vars), 'columns': list(self._col_vars),
                'targets': list(self._target_vars), 'rules': list(self.ruleList),
                'fit_records': self.fit_records, 'decimals': self.decimal_spin_box.value(),
                'metadata': dict(self._result_metadata)}

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
                data += self.table.cellText(row, column) + '\t'
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
