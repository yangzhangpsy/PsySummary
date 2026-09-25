import html

import numpy as np
from scipy.integrate import cumulative_trapezoid
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import hsv_to_rgb, to_hex
from matplotlib.figure import Figure
from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QHBoxLayout, QLabel, QRadioButton, QVBoxLayout,
)

from app.rtDist import rt_distribution_cdf, rt_distribution_pdf
from app.cognitiveModelSpec import COGNITIVE_MODEL_NAMES, RATCLIFF_MODEL
from app.cognitiveModels import cognitive_model_pdf


class CheckableComboBox(QComboBox):
    """Provide a compact multi-select combo box with checkable rows."""

    selectionChanged = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the checkable combo box.

        :param parent: Optional parent widget.
        :return: None.
        """
        super().__init__(parent)
        self.setModel(QStandardItemModel(self))
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText('Select conditions...')
        self.view().viewport().installEventFilter(self)

    def add_check_item(self, text, checked=False):
        """Append a checkable condition.

        :param text: User-visible condition label.
        :param checked: Whether the condition starts selected.
        :return: None.
        """
        item = QStandardItem(text)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Checked if checked else Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)
        self._update_summary()

    def checked_indices(self):
        """Return model-row indices for all selected conditions.

        :return: List of selected row indices.
        """
        return [
            index for index in range(self.model().rowCount())
            if self.model().item(index).checkState() == Qt.Checked
        ]

    def eventFilter(self, watched, event):
        """Toggle a row without closing the popup so several rows can be selected.

        :param watched: Object receiving the event.
        :param event: Qt event to inspect.
        :return: Whether the event was consumed.
        """
        if watched is self.view().viewport() and event.type() == QEvent.MouseButtonRelease:
            index = self.view().indexAt(event.pos())
            if index.isValid():
                item = self.model().itemFromIndex(index)
                item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
                self._update_summary()
                self.selectionChanged.emit()
            return True
        return super().eventFilter(watched, event)

    def _update_summary(self):
        """Show the selected condition or a compact count in the line edit.

        :return: None.
        """
        selected = [self.model().item(index).text() for index in self.checked_indices()]
        if len(selected) == 1:
            summary = selected[0]
        elif selected:
            summary = f'{len(selected)} conditions selected'
        else:
            summary = ''
        self.lineEdit().setText(summary)


class RTFitDiagnosticsDialog(QDialog):
    """Show empirical RT distributions alongside their fitted PDF and CDF."""

    CURVE_COLORS = [
        '#0072BD',
        '#D95319',
        '#EDB120',
        '#7E2F8E',
        '#77AC30',
        '#4DBEEE',
        '#A2142F',
        '#1F77B4',
        '#FF7F0E',
        '#2CA02C',
        '#D62728',
        '#9467BD',
        '#8C564B',
        '#E377C2',
        '#7F7F7F',
        '#BCBD22',
        '#17BECF',
        '#393B79',
        '#637939',
        '#8C6D31',
        '#843C39',
        '#7B4173',
        '#3182BD',
        '#31A354',
        '#756BB1',
        '#636363',
        '#E6550D',
    ]

    def __init__(self, fit_records, parent=None, initial_index=0):
        """Initialize a selectable diagnostics plot.

        :param fit_records: Fitted group records produced by the RT fitting thread.
        :param parent: Optional parent widget.
        :param initial_index: Fit record selected when the dialog opens.
        :return: None.
        """
        super().__init__(parent)
        self.fit_records = fit_records
        self.setWindowTitle('RT Fit Diagnostics')
        self.resize(850, 680)

        layout = QVBoxLayout(self)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel('Display:'))
        self.response_mode_button = QRadioButton('Response')
        self.accuracy_mode_button = QRadioButton('Correct / Error')
        self.display_mode_group = QButtonGroup(self)
        self.display_mode_group.setExclusive(True)
        for button in (self.response_mode_button, self.accuracy_mode_button):
            self.display_mode_group.addButton(button)
        self.response_mode_button.setChecked(True)
        self.accuracy_mode_button.setToolTip(
            'Requires an Accuracy/Correct Variable in the cognitive-model settings.')
        mode_row.addWidget(self.response_mode_button)
        mode_row.addWidget(self.accuracy_mode_button)
        mode_row.addStretch(1)
        self.group_selector = CheckableComboBox()
        for index, record in enumerate(fit_records):
            self.group_selector.add_check_item(
                self._record_label(record), checked=index == initial_index)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)

        layout.addLayout(mode_row)
        layout.addWidget(self.group_selector)
        layout.addWidget(self.status_label)
        layout.addWidget(self.canvas)
        self.group_selector.selectionChanged.connect(self._selection_changed)
        self.display_mode_group.buttonClicked.connect(self._draw_selected_records)
        self._update_accuracy_mode_availability()
        self._draw_selected_records()

    def _selection_changed(self):
        """Update display-mode availability and redraw after group selection changes."""
        self._update_accuracy_mode_availability()
        self._draw_selected_records()

    def _update_accuracy_mode_availability(self):
        """Enable Correct/Error only when every selected record provides accuracy data."""
        selected_records = [
            self.fit_records[index] for index in self.group_selector.checked_indices()
        ]
        supported = bool(selected_records) and all(
            record.get('model') in COGNITIVE_MODEL_NAMES
            and bool(record.get('specification', {}).get('accuracy_variable'))
            and record.get('accuracy') is not None
            for record in selected_records
        )
        self.accuracy_mode_button.setEnabled(supported)
        if not supported and self.accuracy_mode_button.isChecked():
            self.response_mode_button.setChecked(True)

    @staticmethod
    def _group_label(record):
        """Build a readable label for one fitted group assignment.

        :param record: Fit record dictionary.
        :return: Human-readable group label.
        """
        group_vars = record.get('group_vars', [])
        group_values = record.get('group_values', ())
        if not group_vars:
            group_label = 'Overall'
        else:
            group_label = ', '.join(f'{name}={value}' for name, value in zip(group_vars, group_values))
        return group_label

    @classmethod
    def _record_label(cls, record):
        """Build a label containing the distribution and group assignment.

        :param record: Fit record dictionary.
        :return: Human-readable fit label.
        """
        group_label = cls._group_label(record)
        return f"{record['distribution']} — {group_label}"

    def _draw_selected_records(self):
        """Overlay density and cumulative diagnostics for all checked groups.

        :return: None.
        """
        selected_indices = self.group_selector.checked_indices()
        self.figure.clear()
        density_axis = self.figure.add_subplot(211)
        cdf_axis = self.figure.add_subplot(212)
        if not selected_indices:
            self.status_label.setText('Select one or more conditions to display.')
            density_axis.text(0.5, 0.5, 'No conditions selected', ha='center', va='center')
            cdf_axis.text(0.5, 0.5, 'No conditions selected', ha='center', va='center')
            self.canvas.draw_idle()
            return

        status_lines = []
        selected_prefixes = {
            self.fit_records[index].get('result_prefix') for index in selected_indices
        }
        use_compact_curve_labels = len(selected_prefixes) == 1
        curve_counts = [
            self._record_curve_count(self.fit_records[index]) for index in selected_indices
        ]
        curve_colors = self._curve_colors(sum(curve_counts))
        color_offset = 0
        for index, curve_count in zip(selected_indices, curve_counts):
            record = self.fit_records[index]
            record_colors = curve_colors[color_offset:color_offset + curve_count]
            color_offset += curve_count
            color = record_colors[0]
            label = self._record_label(record)
            curve_label = self._group_label(record) if use_compact_curve_labels else label
            if record.get('model') in COGNITIVE_MODEL_NAMES:
                status_lines.append(self._draw_cognitive_record(
                    record, density_axis, cdf_axis, record_colors, curve_label, label))
                continue
            data = np.asarray(record['data'], dtype=float)
            parameters = np.asarray(record['parameters'], dtype=float)
            status = 'Converged' if record['converged'] else 'Not converged'
            status_lines.append(
                f'<span style="color:{color}">■</span> <b>{html.escape(label)}</b>: {status}; '
                f"N={record['n_valid']}; boundary={record['parameter_boundary']} "
                f"({html.escape(record['boundary_details'])}); shift warning={record['shift_warning']} "
                f"({html.escape(record['shift_warning_details'])}); LL={self._number(record['log_likelihood'])}; "
                f"AIC={self._number(record['aic'])}; BIC={self._number(record['bic'])}."
            )
            if data.size == 0:
                continue

            bins = min(50, max(8, int(np.sqrt(data.size))))
            density_axis.hist(
                data, bins=bins, density=True, histtype='stepfilled', alpha=0.12,
                color=color, edgecolor=color, label=f'{curve_label} — observed')
            sorted_data = np.sort(data)
            empirical_cdf = np.arange(1, data.size + 1) / data.size
            cdf_axis.step(
                sorted_data, empirical_cdf, where='post', color=color, linestyle='--',
                linewidth=1.8, label=f'{curve_label} — empirical CDF')

            if np.all(np.isfinite(parameters)):
                spread = max(float(np.ptp(data)), abs(float(np.mean(data))) * 0.05, 1e-6)
                x_values = np.linspace(float(np.min(data)) - 0.05 * spread,
                                       float(np.max(data)) + 0.10 * spread, 500)
                try:
                    pdf_values = rt_distribution_pdf(record['distribution'], x_values, parameters)
                    cdf_values = rt_distribution_cdf(record['distribution'], x_values, parameters)
                    valid_pdf = np.isfinite(pdf_values) & (pdf_values >= 0)
                    valid_cdf = np.isfinite(cdf_values)
                    density_axis.plot(
                        x_values[valid_pdf], pdf_values[valid_pdf], color=color,
                        linewidth=2, label=f'{curve_label} — fitted PDF')
                    cdf_axis.plot(
                        x_values[valid_cdf], cdf_values[valid_cdf], color=color,
                        linewidth=2, label=f'{curve_label} — fitted CDF')
                except Exception as error:
                    status_lines.append(
                        f'<span style="color:{color}">Curve unavailable for {html.escape(label)}: '
                        f'{html.escape(str(error))}</span>')

        self.status_label.setText('<br>'.join(status_lines))

        accuracy_mode = self.accuracy_mode_button.isChecked()
        density_axis.set_ylabel('Density')
        density_axis.set_title(
            'Observed correct/error RT distributions and fitted densities'
            if accuracy_mode else 'Observed RT distributions and fitted densities')
        density_axis.legend(loc='best')
        cdf_axis.set_xlabel('Reaction time')
        cdf_axis.set_ylabel('Cumulative probability')
        cdf_axis.set_title(
            'Correct/error empirical CDFs (dashed) and fitted CDFs (solid)'
            if accuracy_mode else 'Empirical CDFs (dashed) and fitted CDFs (solid)')
        cdf_axis.set_ylim(-0.02, 1.02)
        cdf_axis.legend(loc='best')
        self.canvas.draw_idle()

    def _record_curve_count(self, record):
        """Return the number of independently colored series drawn for one fit record."""
        if record.get('model') not in COGNITIVE_MODEL_NAMES:
            return 1
        if self.accuracy_mode_button.isChecked():
            return 2
        return max(1, len(record.get('specification', {}).get('response_values', [])))

    @classmethod
    def _curve_colors(cls, count):
        """Return enough distinct colors for every currently selected curve series."""
        colors = list(cls.CURVE_COLORS[:count])
        golden_ratio = 0.618033988749895
        while len(colors) < count:
            index = len(colors) - len(cls.CURVE_COLORS)
            hue = (0.11 + index * golden_ratio) % 1.0
            saturation = 0.68 if index % 2 == 0 else 0.82
            value = 0.78 if (index // 2) % 2 == 0 else 0.92
            colors.append(to_hex(hsv_to_rgb((hue, saturation, value))).upper())
        return colors

    def _draw_cognitive_record(self, record, density_axis, cdf_axis, curve_colors,
                               curve_label, label):
        """Draw response-conditioned diagnostics for one cognitive RT model fit."""
        data = np.asarray(record['rt'], dtype=float)
        responses = np.asarray(record['response'], dtype=int)
        status = 'Converged' if record['converged'] else 'Not converged'
        status_line = (
            f'<b>{html.escape(label)}</b>: {status}; N={record["n_valid"]}; '
            f'responses={html.escape(str(record["response_counts"]))}; '
            f'LL={self._number(record["log_likelihood"])}; '
            f'AIC={self._number(record["aic"])}; BIC={self._number(record["bic"])}.'
        )
        if record.get('accuracy_rate') is not None:
            status_line += f' Accuracy={self._number(record["accuracy_rate"])}.'
        if data.size == 0:
            return status_line
        if self.accuracy_mode_button.isChecked():
            return self._draw_cognitive_accuracy_record(
                record, density_axis, cdf_axis, curve_colors, curve_label, status_line)
        specification = record['specification']
        response_values = specification['response_values']
        response_mapping = specification['response_mapping']
        mapped_labels = []
        for response_index in range(1, len(response_values) + 1):
            target = ('lower' if response_index == 1 else 'upper') \
                if record['model'] == RATCLIFF_MODEL else response_index
            mapped_labels.append(next(
                (value for value in response_values if response_mapping.get(str(value)) == target), target))
        spread = max(float(np.ptp(data)), abs(float(np.mean(data))) * 0.05, 1e-6)
        x_values = np.linspace(0.0, float(np.max(data)) + 0.10 * spread, 400)
        for response_index, observed_value in enumerate(mapped_labels, start=1):
            mask = responses == response_index
            response_data = np.sort(data[mask])
            if response_data.size == 0:
                continue
            color = curve_colors[response_index - 1]
            response_label = f'{curve_label} / {observed_value}'
            edges = np.histogram_bin_edges(response_data, bins=min(40, max(6, int(np.sqrt(response_data.size)))))
            counts, edges = np.histogram(response_data, bins=edges)
            widths = np.diff(edges)
            centers = edges[:-1] + widths / 2.0
            density_axis.step(
                centers, counts / (data.size * widths), where='mid', color=color,
                linestyle='--', linewidth=1.4, label=f'{response_label} — observed')
            empirical = np.arange(1, response_data.size + 1) / data.size
            cdf_axis.step(
                response_data, empirical, where='post', color=color, linestyle='--',
                linewidth=1.4, label=f'{response_label} — empirical')
            try:
                fitted_density = cognitive_model_pdf(
                    record['model'], x_values, response_index,
                    record['parameter_names'], record['parameters'], len(response_values))
                density_axis.plot(
                    x_values, fitted_density, color=color, linewidth=2,
                    label=f'{response_label} — fitted')
                fitted_cdf = cumulative_trapezoid(fitted_density, x_values, initial=0.0)
                cdf_axis.plot(
                    x_values, fitted_cdf, color=color, linewidth=2,
                    label=f'{response_label} — fitted')
            except Exception as error:
                status_line += f' Curve unavailable: {html.escape(str(error))}.'
        return status_line

    def _draw_cognitive_accuracy_record(self, record, density_axis, cdf_axis, curve_colors,
                                        curve_label, status_line):
        """Draw observed and model-implied correct/error RT distributions."""
        accuracy = record.get('accuracy')
        if accuracy is None:
            return status_line + ' Correct/Error diagnostics require an Accuracy/Correct Variable.'
        data = np.asarray(record['rt'], dtype=float)
        responses = np.asarray(record['response'], dtype=int)
        accuracy = np.asarray(accuracy, dtype=float)
        valid = np.isfinite(accuracy) & np.isin(accuracy, (0.0, 1.0))
        if not np.any(valid):
            return status_line + ' No valid Accuracy/Correct values are available for diagnostics.'
        valid_count = int(np.sum(valid))
        category_colors = curve_colors[:2]
        for category, category_label, color in zip((1.0, 0.0), ('Correct', 'Error'), category_colors):
            category_data = np.sort(data[valid & (accuracy == category)])
            if category_data.size == 0:
                continue
            response_label = f'{curve_label} / {category_label}'
            edges = np.histogram_bin_edges(
                category_data, bins=min(40, max(6, int(np.sqrt(category_data.size)))))
            counts, edges = np.histogram(category_data, bins=edges)
            widths = np.diff(edges)
            centers = edges[:-1] + widths / 2.0
            density_axis.step(
                centers, counts / (valid_count * widths), where='mid', color=color,
                linestyle='--', linewidth=1.4, label=f'{response_label} — observed')
            empirical = np.arange(1, category_data.size + 1) / valid_count
            cdf_axis.step(
                category_data, empirical, where='post', color=color, linestyle='--',
                linewidth=1.4, label=f'{response_label} — empirical')

        response_count = len(record['specification']['response_values'])
        correct_weights, warning = self._correct_response_weights(
            responses[valid], accuracy[valid], response_count)
        if correct_weights is None:
            return status_line + f' {warning}'
        spread = max(float(np.ptp(data[valid])), abs(float(np.mean(data[valid]))) * 0.05, 1e-6)
        x_values = np.linspace(0.0, float(np.max(data[valid])) + 0.10 * spread, 400)
        try:
            response_densities = np.asarray([
                cognitive_model_pdf(
                    record['model'], x_values, response_index,
                    record['parameter_names'], record['parameters'], response_count)
                for response_index in range(1, response_count + 1)
            ])
            correct_density = np.sum(correct_weights[:, None] * response_densities, axis=0)
            error_density = np.sum((1.0 - correct_weights)[:, None] * response_densities, axis=0)
            for density, category_label, color in zip(
                    (correct_density, error_density), ('Correct', 'Error'), category_colors):
                response_label = f'{curve_label} / {category_label}'
                density_axis.plot(
                    x_values, density, color=color, linewidth=2,
                    label=f'{response_label} — fitted')
                fitted_cdf = cumulative_trapezoid(density, x_values, initial=0.0)
                cdf_axis.plot(
                    x_values, fitted_cdf, color=color, linewidth=2,
                    label=f'{response_label} — fitted')
        except Exception as error:
            status_line += f' Curve unavailable: {html.escape(str(error))}.'
        return status_line

    @staticmethod
    def _correct_response_weights(responses, accuracy, response_count):
        """Infer design weights for each correct response from response and accuracy rows."""
        if response_count == 2:
            correct_responses = np.where(accuracy == 1.0, responses, 3 - responses)
        else:
            correct_responses = np.unique(responses[accuracy == 1.0])
            if correct_responses.size != 1:
                return None, (
                    'Fitted Correct/Error curves are unavailable because a multi-response model '
                    'requires one stable correct response per fitted group.')
            correct_responses = np.full(responses.shape, correct_responses[0], dtype=int)
        weights = np.bincount(correct_responses, minlength=response_count + 1)[1:].astype(float)
        weights /= np.sum(weights)
        return weights, ''

    @staticmethod
    def _number(value):
        """Format a diagnostic number for the status label.

        :param value: Numeric value to format.
        :return: Compact display string.
        """
        return f'{value:.4f}' if np.isfinite(value) else 'NA'
