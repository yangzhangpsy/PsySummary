"""Read-only preparation with buffered scripts and GUI-owned CDF decisions."""

from copy import deepcopy
import threading
import time

import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from app.cognitiveModelSpec import short_rt_summary
from app.dataPreparation import split_filter_rule
from app.fitCancellation import FitCancelled
from app.tool import StatisticTool, warnConditionWiseFiltering
from app.rtDist import _ACTIVE_CANCEL_CHECK


class CDFDecision:
    """Exchange one cutoff through an event; no QWidget is created by the worker."""

    def __init__(self, values, po, omega):
        self.values, self.po, self.omega = values, po, omega
        self.ready = threading.Event()
        self.value = -1

    def resolve(self, value):
        self.value = value
        self.ready.set()


class AnalysisPreparationThread(QThread):
    """Prepare analysis, preview, export, viewer or model-dialog data from a settings snapshot."""

    progress = pyqtSignal(str)
    logMessage = pyqtSignal(str, int)
    cdfRequested = pyqtSignal(object)

    def __init__(self, source, settings, purpose, parent=None):
        super().__init__(parent)
        self.source = source
        self.settings = deepcopy(settings)
        self.purpose = purpose
        self.result = None
        self.script_lines = []
        self.short_warnings = []
        self.error = None
        self.cancelled = False
        self.timings = {}
        self._cancel = threading.Event()

    def requestInterruption(self):
        self._cancel.set()
        super().requestInterruption()

    def isInterruptionRequested(self):
        return self._cancel.is_set()

    def _check(self):
        if self._cancel.is_set():
            raise InterruptedError('Data preparation cancelled.')

    def _decide_cdf(self, values, po, omega):
        decision = CDFDecision(values, po, omega)
        self._check()
        self.progress.emit('Waiting for CDF cutoff confirmation…')
        self.cdfRequested.emit(decision)
        while not decision.ready.wait(.05):
            self._check()
        self._check()
        return decision.value

    def _filter(self, source, record):
        settings = self.settings
        return StatisticTool.filterData(
            settings['rows'], settings['columns'], source, settings['rules'],
            record_script=record, script_collector=self.script_lines,
            log=self.logMessage.emit, cdf_decider=self._decide_cdf,
            check_cancelled=self._check, progress=self.progress.emit)

    def run(self):
        start = time.perf_counter()
        token = _ACTIVE_CANCEL_CHECK.set(self.isInterruptionRequested)
        try:
            self._check()
            if self.purpose == 'run':
                from app.lib.pivotedDataWidget import PreparedAnalysis
                self.result = PreparedAnalysis(
                    self.source, self.settings['rows'], self.settings['columns'],
                    self.settings['targets'], self.settings['rules'],
                    log=self.logMessage.emit, cdf_decider=self._decide_cdf,
                    check_cancelled=self._check, progress=self.progress.emit)
                self.timings['filter_and_validation'] = time.perf_counter() - start
                self.progress.emit('Checking retained reaction times…')
                for target in self.result.target_vars:
                    self._check()
                    if isinstance(target, dict):
                        description = short_rt_summary(target['model_specification'], self.result.dataframe,
                                                       self.result.row_vars + self.result.col_vars)
                        if description: self.short_warnings.append(description)
                self.result.condition_logs = []
                warnConditionWiseFiltering(
                    self.result.row_vars, self.result.col_vars, self.source, self.result.rule_list,
                    log=lambda message, kind: self.result.condition_logs.append((message, kind)))
            elif self.purpose == 'preview':
                self.result = self._preview()
            elif self.purpose == 'export':
                self.result = self._filter(self.source, True) if self.settings['rules'] else self.source
                if not self.settings['rules']:
                    self.script_lines.append('cdfPoolingOmegas = []')
            elif self.purpose in ('view', 'model-settings'):
                self.result = self._filter(self.source, False) if self.settings['rules'] else self.source
            else:
                raise ValueError('Unknown preparation purpose.')
            self._check()
        except (InterruptedError, FitCancelled):
            self.cancelled = True
            self.result = None
        except Exception as error:
            self.error = error
            self.result = None
        finally:
            self.timings['total'] = time.perf_counter() - start
            _ACTIVE_CANCEL_CHECK.reset(token)

    def _preview(self):
        """Filter only rule/group columns and keep duplicate source indices positional."""
        self.progress.emit('Preparing preview data…')
        targets = []
        for target in self.settings['targets']:
            self._check()
            variable = (target['model_specification']['rt_variable'] if isinstance(target, dict)
                        else target.split('@', 1)[0])
            if (variable not in targets and variable in self.source.columns
                    and pd.to_numeric(self.source[variable], errors='coerce').notna().any()):
                targets.append(variable)
        if not targets:
            raise ValueError('No numeric Data variable is defined. Drag a numeric variable into the Data area.')
        groups = self.settings['rows'] + self.settings['columns']
        columns = list(dict.fromkeys(groups + [split_filter_rule(rule)[0] for rule in self.settings['rules']]))
        marker = '__psysummary_preview_row_id__'
        while marker in self.source.columns: marker += '_'
        narrow = self.source.loc[:, columns].copy(deep=False)
        narrow[marker] = np.arange(len(self.source), dtype=np.intp)
        retained = self._filter(narrow, False)
        mask = np.zeros(len(self.source), dtype=bool)
        mask[retained[marker].to_numpy(dtype=np.intp)] = True
        del retained, narrow
        self._check()
        # Numeric conversion/copy is also preparation; the dialog only owns this compact snapshot.
        data = self.source.loc[:, list(dict.fromkeys(targets + groups))].copy()
        data.index = pd.RangeIndex(len(data))
        for variable in targets:
            self._check()
            data[variable] = pd.to_numeric(data[variable], errors='coerce')
        return {'data': data, 'mask': mask, 'targets': targets}
