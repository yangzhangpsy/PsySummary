"""Configuration dialog for PsySummary cognitive response-time models."""

from copy import deepcopy

import numpy as np
import pandas as pd
from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QMessageBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from app.cognitiveModelSpec import (
    ACCURACY_CODING, BOUNDARY_CODING_LABELS, LBA_MODEL, RATCLIFF_MODEL,
    RESPONSE_CODING, RDM_MODEL, default_parameters,
    make_model_specification, model_response_mapping, parameter_tooltip, response_value_token,
    validate_model_specification,
)


class CognitiveModelDialog(QDialog):
    """Collect data mappings, response mappings, and fixed/free parameters."""

    NONE_LABEL = '(None)'

    def __init__(self, dataframe, model, rt_variable, specification=None, parent=None):
        """Initialize a model settings dialog.

        :param dataframe: Data frame supplying variables and observed response values.
        :param model: Selected cognitive-model display name.
        :param rt_variable: Variable initially assigned as reaction time.
        :param specification: Optional previously saved model specification.
        :param parent: Optional parent widget.
        :return: None.
        """
        super().__init__(parent)
        self.dataframe = dataframe if dataframe is not None else pd.DataFrame()
        self.model = model
        self.rt_variable = rt_variable
        self.original_specification = (
            deepcopy(specification)
            if specification and specification.get('model') == model else None
        )
        self._response_values = []
        self._parameter_widgets = []
        self._mapping_widgets = []
        self.setWindowTitle(f'{model} Settings')
        self.resize(820, 760)
        self._build_ui()
        self._load_initial_state()
        self._update_mapping_presentation()
        self._update_validation()
        self.accuracy_combo.currentTextChanged.connect(self._update_validation)

    def _build_ui(self):
        """Build the data, mapping, parameter, and optimizer controls."""
        root_layout = QVBoxLayout(self)

        data_group = QGroupBox('Data Mapping')
        data_layout = QGridLayout(data_group)
        self.rt_label = QLabel(self.rt_variable)
        self.boundary_coding_combo = QComboBox()
        for coding, label in BOUNDARY_CODING_LABELS.items():
            self.boundary_coding_combo.addItem(label, coding)
        self.boundary_coding_combo.setCurrentIndex(
            self.boundary_coding_combo.findData(ACCURACY_CODING))
        self.boundary_coding_combo.setVisible(self.model == RATCLIFF_MODEL)
        self.response_combo = QComboBox()
        self.accuracy_combo = QComboBox()
        self.unit_combo = QComboBox()
        variables = [str(column) for column in self.dataframe.columns]
        self.response_combo.addItems(
            [self.NONE_LABEL] + [name for name in variables if name != self.rt_variable])
        self.accuracy_combo.addItems([self.NONE_LABEL] + [name for name in variables if name != self.rt_variable])
        self.unit_combo.addItems(['seconds', 'milliseconds'])
        self.rt_label.setToolTip('Reaction-time variable used as the observed model latency.')
        self.unit_combo.setToolTip('Unit used by the RT variable. Model time parameters are stored in seconds.')
        self.response_combo.setToolTip(
            'Observed response or choice variable used to identify the winning boundary or accumulator.')
        self.accuracy_combo.setToolTip(
            ('Required correctness variable. Retained rows must use 1 for correct and 0 for error; '
             'recode or exclude other and missing values in Filter Data.')
            if self.model == RATCLIFF_MODEL else
            'Optional 0/1 correctness variable used for descriptive accuracy output.')
        self.boundary_coding_combo.setToolTip(
            'Accuracy Coding fits correct/error boundaries and mirrors z and d when the correct '
            'physical response is lower. Response Coding fits the mapped response boundaries and '
            'reverses v when the correct physical response is lower.')
        boundary_coding_label = QLabel('Boundary Coding:')
        boundary_coding_label.setVisible(self.model == RATCLIFF_MODEL)
        rt_variable_label = QLabel('RT Variable:')
        rt_unit_label = QLabel('RT Unit:')
        response_variable_label = QLabel('Response Variable:')
        accuracy_variable_label = QLabel('Accuracy Variable:')
        for label in (
                boundary_coding_label, rt_variable_label, rt_unit_label, response_variable_label,
                accuracy_variable_label):
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        data_layout.addWidget(boundary_coding_label, 0, 0)
        data_layout.addWidget(self.boundary_coding_combo, 0, 1, 1, 3)
        data_layout.addWidget(rt_variable_label, 1, 0)
        data_layout.addWidget(self.rt_label, 1, 1)
        data_layout.addWidget(rt_unit_label, 1, 2)
        data_layout.addWidget(self.unit_combo, 1, 3)
        data_layout.addWidget(response_variable_label, 2, 0)
        data_layout.addWidget(self.response_combo, 2, 1)
        data_layout.addWidget(accuracy_variable_label, 2, 2)
        data_layout.addWidget(self.accuracy_combo, 2, 3)
        data_layout.setColumnStretch(1, 1)
        data_layout.setColumnStretch(3, 1)

        self.mapping_group = QGroupBox('Response Mapping')
        mapping_layout = QVBoxLayout(self.mapping_group)
        self.accuracy_mapping_label = QLabel('Model Boundary Mapping (from Accuracy Variable)')
        self.accuracy_mapping_table = QTableWidget(2, 2)
        self.accuracy_mapping_table.setHorizontalHeaderLabels(
            ['Accuracy Value', 'Model Boundary'])
        self.accuracy_mapping_table.horizontalHeader().setStretchLastSection(True)
        self.accuracy_mapping_table.verticalHeader().setVisible(False)
        self.accuracy_mapping_table.verticalHeader().setDefaultSectionSize(26)
        self.accuracy_mapping_table.setColumnWidth(0, 220)
        for row, (observed, boundary) in enumerate(
                (('0 (Error)', 'lower'), ('1 (Correct)', 'upper'))):
            for column, text in enumerate((observed, boundary)):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.accuracy_mapping_table.setItem(row, column, item)
        self._set_table_visible_rows(self.accuracy_mapping_table, 2)
        self.physical_mapping_label = QLabel(
            'Physical Response Mapping (used only to orient z and d)')
        self.mapping_table = QTableWidget(0, 2)
        self.mapping_table.setHorizontalHeaderLabels(['Observed Value', 'Model Response'])
        self.mapping_table.horizontalHeader().setStretchLastSection(True)
        self.mapping_table.verticalHeader().setVisible(False)
        self.mapping_table.verticalHeader().setDefaultSectionSize(26)
        self.mapping_table.setColumnWidth(0, 220)
        mapping_layout.addWidget(self.accuracy_mapping_label)
        mapping_layout.addWidget(self.accuracy_mapping_table)
        mapping_layout.addWidget(self.physical_mapping_label)
        mapping_layout.addWidget(self.mapping_table)

        parameter_group = QGroupBox('Parameters')
        parameter_layout = QVBoxLayout(parameter_group)
        self.parameter_table = QTableWidget(0, 5)
        self.parameter_table.setHorizontalHeaderLabels(
            ['Parameter', 'Mode', 'Fixed Value / Start', 'Lower', 'Upper'])
        self.parameter_table.horizontalHeader().setStretchLastSection(True)
        self.parameter_table.verticalHeader().setVisible(False)
        self.parameter_table.verticalHeader().setDefaultSectionSize(26)
        self.parameter_table.setColumnWidth(0, 110)
        self.parameter_table.setColumnWidth(1, 100)
        self.parameter_table.setColumnWidth(2, 190)
        self.parameter_table.setColumnWidth(3, 110)
        parameter_layout.addWidget(self.parameter_table)

        optimizer_group = QGroupBox('Fit Options')
        optimizer_layout = QGridLayout(optimizer_group)
        self.starts_spin = QSpinBox()
        self.starts_spin.setRange(1, 100)
        self.starts_spin.setValue(3)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setSpecialValueText('Time-based')
        self.seed_spin.setValue(0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(100, 100000)
        self.iterations_spin.setValue(1000)
        starts_label = QLabel('Multiple Starts:')
        seed_label = QLabel('Random Seed:')
        iterations_label = QLabel('Maximum Iterations:')
        starts_tip = ('Number of optimizer runs. The best likelihood across the different '
                      'starting points is retained.')
        seed_tip = ('Seed used to generate additional optimizer starting points. Time-based creates '
                    'a new seed when fitting starts; enter a positive integer for reproducible fits.')
        iterations_tip = 'Maximum number of optimizer iterations allowed for each starting point.'
        starts_label.setToolTip(starts_tip)
        self.starts_spin.setToolTip(starts_tip)
        seed_label.setToolTip(seed_tip)
        self.seed_spin.setToolTip(seed_tip)
        iterations_label.setToolTip(iterations_tip)
        self.iterations_spin.setToolTip(iterations_tip)
        optimizer_layout.addWidget(starts_label, 0, 0)
        optimizer_layout.addWidget(self.starts_spin, 0, 1)
        optimizer_layout.addWidget(seed_label, 0, 2)
        optimizer_layout.addWidget(self.seed_spin, 0, 3)
        optimizer_layout.addWidget(iterations_label, 0, 4)
        optimizer_layout.addWidget(self.iterations_spin, 0, 5)
        optimizer_layout.setColumnStretch(1, 1)
        optimizer_layout.setColumnStretch(3, 1)
        optimizer_layout.setColumnStretch(5, 1)

        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet('color: #8b1a1a;')
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setText('Save Settings')
        self.button_box.button(QDialogButtonBox.Ok).setToolTip(
            'Save and close, including incomplete settings. Settings are checked again before running.')

        root_layout.addWidget(data_group)
        root_layout.addWidget(self.mapping_group)
        root_layout.addWidget(parameter_group, 1)
        root_layout.addWidget(optimizer_group)
        root_layout.addWidget(self.validation_label)
        root_layout.addWidget(self.button_box)

        self.response_combo.currentTextChanged.connect(self._response_variable_changed)
        self.boundary_coding_combo.currentIndexChanged.connect(
            self._boundary_coding_changed)
        self.unit_combo.currentTextChanged.connect(self._reset_parameters_for_current_mapping)
        self.button_box.accepted.connect(self._accept_if_valid)
        self.button_box.rejected.connect(self.reject)

    def _boundary_coding_changed(self, *_args):
        """Refresh mapping labels and validation after changing boundary coding."""
        self._update_mapping_presentation()
        self._update_validation()

    def _update_mapping_presentation(self):
        """Distinguish model accuracy boundaries from physical-response orientation."""
        accuracy_coding = (
            self.model == RATCLIFF_MODEL
            and self.boundary_coding_combo.currentData() == ACCURACY_CODING
        )
        self.accuracy_mapping_label.setVisible(accuracy_coding)
        self.accuracy_mapping_table.setVisible(accuracy_coding)
        self.physical_mapping_label.setVisible(accuracy_coding)
        if accuracy_coding:
            self.mapping_group.setTitle('Boundary Mapping')
            self.mapping_table.setHorizontalHeaderLabels(
                ['Observed Response', 'Physical Boundary'])
            self.mapping_table.setToolTip(
                'This physical response mapping does not define the fitted Correct/Error '
                'boundaries. It determines when z is mirrored to a-z and d to -d.')
        else:
            self.mapping_group.setTitle(
                'Boundary Mapping' if self.model == RATCLIFF_MODEL else 'Response Mapping')
            headers = (['Observed Response', 'Model Boundary']
                       if self.model == RATCLIFF_MODEL else
                       ['Observed Value', 'Model Response'])
            self.mapping_table.setHorizontalHeaderLabels(headers)
            self.mapping_table.setToolTip('')

    def _load_initial_state(self):
        """Load an existing specification or choose conservative defaults."""
        specification = self.original_specification
        if specification:
            signal_blockers = [
                QSignalBlocker(self.response_combo),
                QSignalBlocker(self.accuracy_combo),
                QSignalBlocker(self.boundary_coding_combo),
                QSignalBlocker(self.unit_combo),
            ]
            try:
                for combo, field in ((self.response_combo, 'response_variable'),
                                     (self.accuracy_combo, 'accuracy_variable')):
                    saved_variable = specification.get(field)
                    if saved_variable and combo.findText(saved_variable) < 0:
                        combo.addItem(saved_variable)
                self.response_combo.setCurrentText(
                    specification.get('response_variable') or self.NONE_LABEL)
                self.accuracy_combo.setCurrentText(
                    specification.get('accuracy_variable') or self.NONE_LABEL)
                self.unit_combo.setCurrentText(specification.get('rt_unit', 'seconds'))
                boundary_coding = specification.get('boundary_coding', RESPONSE_CODING)
                coding_index = self.boundary_coding_combo.findData(boundary_coding)
                self.boundary_coding_combo.setCurrentIndex(max(0, coding_index))
            finally:
                del signal_blockers
            optimizer = specification.get('optimizer', {})
            self.starts_spin.setValue(int(optimizer.get('starts', 3)))
            seed = optimizer.get('seed')
            self.seed_spin.setValue(0 if seed is None else int(seed))
            self.iterations_spin.setValue(int(optimizer.get('max_iterations', 1000)))
            saved_response_values = specification.get('response_values', [])
            response_variable = specification.get('response_variable')
            if response_variable in self.dataframe.columns:
                current_values = list(pd.unique(self.dataframe[response_variable].dropna()))
            else:
                current_values = []
            current_by_key = {response_value_token(value): value for value in current_values}
            saved_keys = [response_value_token(value) for value in saved_response_values]
            response_values = [
                current_by_key[key] for key in saved_keys if key in current_by_key
            ]
            response_values.extend(
                value for value in current_values if response_value_token(value) not in saved_keys)
            response_set_unchanged = (
                len(saved_response_values) == len(response_values)
                and set(saved_keys) == {response_value_token(value) for value in response_values}
            )
            saved_mapping = model_response_mapping(specification)
            mapping = {str(value): saved_mapping[response_value_token(value)]
                       for value in response_values
                       if saved_mapping.get(response_value_token(value)) is not None}
            old_indices = {}
            if not response_set_unchanged and self.model != RATCLIFF_MODEL:
                for index, value in enumerate(response_values, 1):
                    old_indices[index] = mapping.get(str(value))
                    mapping[str(value)] = index
            self._set_response_mapping(response_values, mapping)
            if response_set_unchanged or self.model == RATCLIFF_MODEL:
                self._set_parameter_rows(specification.get('parameters', []))
            else:
                saved_parameters = {row['name']: row for row in specification.get('parameters', [])}
                parameters = default_parameters(self.model, response_values, self._minimum_rt_seconds())
                for index, parameter in enumerate(parameters):
                    name = parameter['name']
                    old_name = name
                    if '[' in name:
                        base, suffix = name.split('[', 1)
                        old_name = f'{base}[{old_indices.get(int(suffix[:-1]))}]'
                    if old_name in saved_parameters:
                        parameters[index] = dict(saved_parameters[old_name], name=name)
                self._set_parameter_rows(parameters)
            return

        numeric_rt = pd.to_numeric(self.dataframe.get(self.rt_variable, pd.Series(dtype=float)), errors='coerce')
        median_rt = float(numeric_rt.dropna().median()) if numeric_rt.notna().any() else 1.0
        self.unit_combo.setCurrentText('milliseconds' if median_rt > 20 else 'seconds')

    def _response_variable_changed(self, variable_name):
        """Rebuild mapping and defaults when the response source changes."""
        if variable_name == self.NONE_LABEL or variable_name not in self.dataframe.columns:
            self._set_response_mapping([], {})
            self._set_parameter_rows([])
            self._update_validation()
            return
        values = list(pd.unique(self.dataframe[variable_name].dropna()))
        self._set_response_mapping(values, {})
        self._reset_parameters_for_current_mapping()
        self._update_validation()

    def _set_response_mapping(self, response_values, saved_mapping):
        """Populate observed response values and their model-side meanings."""
        self._response_values = [value.item() if hasattr(value, 'item') else value for value in response_values]
        self._clear_table_contents(self.mapping_table)
        self._mapping_widgets = []
        self.mapping_table.setRowCount(len(self._response_values))
        for row, value in enumerate(self._response_values):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(row, 0, item)
            mapping_combo = QComboBox()
            if self.model == RATCLIFF_MODEL:
                mapping_combo.addItems(['lower', 'upper'])
                default_mapping = 'lower' if row == 0 else 'upper'
            else:
                for accumulator_index in range(1, len(self._response_values) + 1):
                    mapping_combo.addItem(f'Accumulator {accumulator_index}', accumulator_index)
                default_mapping = row + 1
            saved_value = saved_mapping.get(str(value), default_mapping)
            if self.model == RATCLIFF_MODEL:
                mapping_combo.setCurrentText(str(saved_value))
            else:
                saved_index = mapping_combo.findData(saved_value)
                mapping_combo.setCurrentIndex(saved_index if saved_index >= 0 else row)
            self.mapping_table.setCellWidget(row, 1, mapping_combo)
            self._mapping_widgets.append(mapping_combo)
            mapping_combo.currentIndexChanged.connect(self._update_validation)
        self._set_table_visible_rows(self.mapping_table, 2)

    def _minimum_rt_seconds(self):
        """Return the smallest positive RT after applying the selected unit."""
        values = pd.to_numeric(self.dataframe.get(self.rt_variable, pd.Series(dtype=float)), errors='coerce')
        values = values[np.isfinite(values) & (values > 0)]
        minimum = float(values.min()) if not values.empty else 0.2
        return minimum / 1000.0 if self.unit_combo.currentText() == 'milliseconds' else minimum

    def _reset_parameters_for_current_mapping(self):
        """Replace parameter rows with defaults for the current response mapping."""
        if not self._response_values:
            return
        self._set_parameter_rows(default_parameters(
            self.model, self._response_values, self._minimum_rt_seconds()))

    def _set_parameter_rows(self, parameters):
        """Populate fixed/free parameter controls from serializable rows."""
        self._clear_table_contents(self.parameter_table)
        self._parameter_widgets = []
        self.parameter_table.setRowCount(len(parameters))
        for row, parameter in enumerate(parameters):
            name_item = QTableWidgetItem(parameter['name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            description = parameter_tooltip(self.model, parameter['name'])
            name_item.setToolTip(description)
            self.parameter_table.setItem(row, 0, name_item)
            mode_combo = QComboBox()
            mode_combo.addItems(['free', 'fixed'])
            mode_combo.setCurrentText(parameter.get('mode', 'free'))
            value_edit = QLineEdit(str(parameter.get('value', '')))
            lower_edit = QLineEdit(str(parameter.get('lower', '')))
            upper_edit = QLineEdit(str(parameter.get('upper', '')))
            mode_combo.setToolTip(
                f'{description}\n\nFixed keeps this parameter constant. Free estimates it independently in each group.')
            value_edit.setToolTip(
                f'{description}\n\nEnter the constant value for Fixed mode or optimizer starting value for Free mode.')
            lower_edit.setToolTip(
                f'{description}\n\nLower optimization bound used when this parameter is Free.')
            upper_edit.setToolTip(
                f'{description}\n\nUpper optimization bound used when this parameter is Free.')
            self.parameter_table.setCellWidget(row, 1, mode_combo)
            self.parameter_table.setCellWidget(row, 2, value_edit)
            self.parameter_table.setCellWidget(row, 3, lower_edit)
            self.parameter_table.setCellWidget(row, 4, upper_edit)
            self._parameter_widgets.append((parameter['name'], mode_combo, value_edit, lower_edit, upper_edit))
            mode_combo.currentIndexChanged.connect(self._update_validation)
            for editor in (value_edit, lower_edit, upper_edit):
                editor.textChanged.connect(self._update_validation)
        visible_rows = min(max(len(parameters), 6), 9)
        self._set_table_visible_rows(self.parameter_table, visible_rows)

    @staticmethod
    def _clear_table_contents(table):
        """Remove old cell widgets immediately before rebuilding a table."""
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                widget = table.cellWidget(row, column)
                if widget is not None:
                    widget.hide()
                    table.removeCellWidget(row, column)
                    widget.deleteLater()
        table.clearContents()
        table.setRowCount(0)

    @staticmethod
    def _set_table_visible_rows(table, row_count):
        """Fix a table height to show the requested number of rows without clipping."""
        row_height = table.verticalHeader().defaultSectionSize()
        header_height = table.horizontalHeader().sizeHint().height()
        frame_height = table.frameWidth() * 2
        table.setFixedHeight(header_height + row_height * row_count + frame_height + 2)

    def _current_mapping(self):
        """Return the user-selected mapping from observed values to model responses."""
        mapping = {}
        for value, combo in zip(self._response_values, self._mapping_widgets):
            mapping[str(value)] = combo.currentText() if self.model == RATCLIFF_MODEL else combo.currentData()
        return mapping

    def specification(self):
        """Return the current controls as a serializable model specification."""
        response_variable = self.response_combo.currentText()
        if response_variable == self.NONE_LABEL:
            response_variable = None
        specification = make_model_specification(
            self.model,
            self.rt_variable,
            response_variable,
            self._response_values,
            minimum_rt=self._minimum_rt_seconds(),
            accuracy_variable=(None if self.accuracy_combo.currentText() == self.NONE_LABEL
                               else self.accuracy_combo.currentText()),
            rt_unit=self.unit_combo.currentText(),
            boundary_coding=self.boundary_coding_combo.currentData(),
        )
        specification['response_mapping'] = self._current_mapping()
        specification['parameters'] = [
            {
                'name': name,
                'mode': mode.currentText(),
                'value': self._draft_number(value.text()),
                'lower': self._draft_number(lower.text()),
                'upper': self._draft_number(upper.text()),
            }
            for name, mode, value, lower, upper in self._parameter_widgets
        ]
        specification['optimizer'] = {
            'method': 'SLSQP',
            'starts': self.starts_spin.value(),
            'seed': None if self.seed_spin.value() == 0 else self.seed_spin.value(),
            'max_iterations': self.iterations_spin.value(),
        }
        return specification

    @staticmethod
    def _draft_number(text):
        """Keep incomplete numeric input serializable without discarding the user's draft."""
        try:
            value = float(text)
            return value if np.isfinite(value) else text
        except ValueError:
            return text

    def _update_validation(self, *_args):
        """Show non-blocking validation feedback while allowing settings to be saved."""
        try:
            specification = self.specification()
            validate_model_specification(specification, self.dataframe.columns)
            accuracy_variable = specification.get('accuracy_variable')
            if accuracy_variable:
                rt = pd.to_numeric(self.dataframe[specification['rt_variable']], errors='coerce')
                responses = self.dataframe[specification['response_variable']]
                retained = np.isfinite(rt) & responses.notna()
                accuracy = pd.to_numeric(
                    self.dataframe.loc[retained, accuracy_variable], errors='coerce')
                if accuracy.empty or not np.isfinite(accuracy).all() or not accuracy.isin([0, 1]).all():
                    raise ValueError(
                        'Accuracy Variable must contain only 0 (error) and 1 (correct), with no '
                        'missing values. Recode or exclude invalid rows in Filter Data.')
        except (TypeError, ValueError) as error:
            self.validation_label.setStyleSheet('color: #8b1a1a;')
            self.validation_label.setText(f'{error}\nYou can save these settings now and correct them before running.')
            return
        self.validation_label.clear()
        self.validation_label.setStyleSheet('')
        if self.model == RATCLIFF_MODEL:
            coding = specification.get('boundary_coding', RESPONSE_CODING)
            if coding == ACCURACY_CODING:
                message = (
                    'Accuracy Coding fits Correct/Error boundaries. The physical response mapping '
                    'is retained by using z and d for upper-correct trials and a-z and -d for '
                    'lower-correct trials.')
            else:
                message = (
                    'Response Coding fits the mapped response boundaries. Drift uses v for '
                    'upper-correct trials and -v for lower-correct trials.')
            self.validation_label.setText(
                message + ' One shared parameter set is estimated from all retained trials.')

    def _accept_if_valid(self):
        """Save the current draft; execution performs strict validation separately."""
        self.accept()


def edit_cognitive_model(dataframe, model, rt_variable, specification=None, parent=None):
    """Open a cognitive-model dialog and return the accepted specification."""
    dialog = CognitiveModelDialog(dataframe, model, rt_variable, specification, parent)
    return dialog.specification() if dialog.exec_() == QDialog.Accepted else None
