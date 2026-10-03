import html

import numpy as np
from scipy.integrate import cumulative_trapezoid
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import hsv_to_rgb, to_hex
from matplotlib.figure import Figure
from PyQt5.QtCore import QRect, QSize, Qt, QThread
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QSplitter, QTextBrowser, QFrame, QVBoxLayout, QWidget,
)

from app.rtDist import rt_distribution_cdf, rt_distribution_pdf
from app.cognitiveModelSpec import (
    COGNITIVE_MODEL_NAMES, RATCLIFF_MODEL, RESPONSE_CODING,
)
from app.cognitiveModels import fitted_accuracy_pdf, fitted_response_pdf



_CURVE_LABEL = '\ufff0curve\ufff1'
_RECORD_LABEL = '\ufff0record\ufff1'


class _NumericAxes:
    """Collect numeric plotting commands without importing or calling Matplotlib objects."""

    def __init__(self, commands, axis, cancel_check):
        self.commands = commands
        self.axis = axis
        self.cancel_check = cancel_check

    def _append(self, method, args, options):
        if self.cancel_check():
            raise InterruptedError('Curve preparation cancelled.')
        self.commands.append((self.axis, method, args, options))

    def plot(self, *args, **options):
        self._append('plot', args, options)

    def step(self, *args, **options):
        self._append('step', args, options)

    def hist(self, values, bins, density, histtype, **options):
        # Match Matplotlib's stepfilled histogram, but calculate bins off the GUI thread.
        counts, edges = np.histogram(values, bins=bins, density=density)
        self._append('stairs', (counts, edges), dict(options, fill=True))


class _CurveBuilder:
    """Prepare color/selection-independent curves using the existing diagnostic formulas."""

    def __init__(self, accuracy_mode):
        self.accuracy_mode = accuracy_mode

    def prepare(self, record, cancel_check):
        commands = []
        density_axis = _NumericAxes(commands, 0, cancel_check)
        cdf_axis = _NumericAxes(commands, 1, cancel_check)
        if record.get('model') in COGNITIVE_MODEL_NAMES:
            count = 2 if self.accuracy_mode else max(1, len(record['specification']['response_values']))
            status = self._draw_cognitive_record(
                record, density_axis, cdf_axis, list(range(count)), _CURVE_LABEL, _RECORD_LABEL)
            return {'commands': commands, 'status': status}
        data = np.asarray(record['data'], dtype=float)
        parameters = np.asarray(record['parameters'], dtype=float)
        convergence = 'Converged' if record['converged'] else 'Not converged'
        status = (
            f'<b>{_RECORD_LABEL}</b>: {convergence}; '
            f"N={record['n_valid']}; boundary={record['parameter_boundary']} "
            f"({html.escape(record['boundary_details'])}); shift warning={record['shift_warning']} "
            f"({html.escape(record['shift_warning_details'])}); LL={self._number(record['log_likelihood'])}; "
            f"AIC={self._number(record['aic'])}; BIC={self._number(record['bic'])}.")
        if data.size:
            bins = min(50, max(8, int(np.sqrt(data.size))))
            density_axis.hist(data, bins=bins, density=True, histtype='stepfilled',
                              alpha=0.12, color=0, edgecolor=0, label=f'{_CURVE_LABEL} — observed')
            sorted_data = np.sort(data)
            empirical = np.arange(1, data.size + 1) / data.size
            cdf_axis.step(sorted_data, empirical, where='post', color=0, linestyle='--',
                          linewidth=1.8, label=f'{_CURVE_LABEL} — empirical CDF')
            if np.all(np.isfinite(parameters)):
                spread = max(float(np.ptp(data)), abs(float(np.mean(data))) * 0.05, 1e-6)
                x_values = np.linspace(float(np.min(data)) - 0.05 * spread,
                                       float(np.max(data)) + 0.10 * spread, 500)
                try:
                    pdf = rt_distribution_pdf(record['distribution'], x_values, parameters)
                    if cancel_check():
                        raise InterruptedError('Curve preparation cancelled.')
                    cdf = rt_distribution_cdf(record['distribution'], x_values, parameters)
                    valid_pdf = np.isfinite(pdf) & (pdf >= 0)
                    valid_cdf = np.isfinite(cdf)
                    density_axis.plot(x_values[valid_pdf], pdf[valid_pdf], color=0,
                                      linewidth=2, label=f'{_CURVE_LABEL} — Fitted model')
                    cdf_axis.plot(x_values[valid_cdf], cdf[valid_cdf], color=0,
                                  linewidth=2, label=f'{_CURVE_LABEL} — Fitted model')
                except Exception as error:
                    status += f' Curve unavailable: {html.escape(str(error))}.'
        return {'commands': commands, 'status': status}

    def _draw_cognitive_record(self, record, density_axis, cdf_axis, curve_colors,
                               curve_label, label):
        """Draw response-conditioned diagnostics for one cognitive RT model fit."""
        data = np.asarray(record['rt'], dtype=float)
        responses = np.asarray(record['response'], dtype=int)
        status = 'Converged' if record['converged'] else 'Not converged'
        status_line = (
            f'<b>{html.escape(label)}</b>: {status}; N={record["n_valid"]}; '
            f'coding={html.escape(record.get("boundary_coding", RESPONSE_CODING))}; '
            f'responses={html.escape(str(record["response_counts"]))}; '
            f'LL={self._number(record["log_likelihood"])}; '
            f'AIC={self._number(record["aic"])}; BIC={self._number(record["bic"])}.'
        )
        if record.get('accuracy_rate') is not None:
            status_line += f' Accuracy={self._number(record["accuracy_rate"])}.'
        if data.size == 0:
            return status_line
        if self.accuracy_mode:
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
                fitted_density = fitted_response_pdf(record, x_values, response_index)
                density_axis.plot(
                    x_values, fitted_density, color=color, linewidth=2,
                    label=f'{response_label} — Fitted model')
                fitted_cdf = cumulative_trapezoid(fitted_density, x_values, initial=0.0)
                cdf_axis.plot(
                    x_values, fitted_cdf, color=color, linewidth=2,
                    label=f'{response_label} — Fitted model')
            except Exception as error:
                status_line += f' Curve unavailable: {html.escape(str(error))}.'
        return status_line


    def _draw_cognitive_accuracy_record(self, record, density_axis, cdf_axis, curve_colors,
                                        curve_label, status_line):
        """Draw observed and model-implied correct/error RT distributions."""
        accuracy = record.get('accuracy')
        if accuracy is None:
            return status_line + ' Correct/Error diagnostics require an Accuracy Variable.'
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
        correct_weights = record.get('correct_response_weights')
        warning = ''
        if correct_weights is None:
            correct_weights, warning = self._correct_response_weights(
                responses[valid], accuracy[valid], response_count)
        else:
            correct_weights = np.asarray(correct_weights, dtype=float)
        if correct_weights is None:
            return status_line + f' {warning}'
        observed_accuracy = float(np.mean(accuracy[valid]))
        weight_labels = (
            ['lower', 'upper'] if record['model'] == RATCLIFF_MODEL
            else [f'Accumulator {index}' for index in range(1, response_count + 1)])
        weights_text = ' / '.join(
            f'{name}: {weight:.2%}' for name, weight in zip(weight_labels, correct_weights))
        status_line += (
            f'<br>Observed correct/error: {observed_accuracy:.2%} / {1.0 - observed_accuracy:.2%}. '
            f'Correct-response weights ({weights_text}) reflect which response should be correct. '
            'Both displays use the same shared parameter set; switching display does not refit the model.')
        spread = max(float(np.ptp(data[valid])), abs(float(np.mean(data[valid]))) * 0.05, 1e-6)
        x_values = np.linspace(0.0, float(np.max(data[valid])) + 0.10 * spread, 400)
        try:
            correct_density = fitted_accuracy_pdf(record, x_values, correct=True)
            error_density = fitted_accuracy_pdf(record, x_values, correct=False)
            for density, category_label, color in zip(
                    (correct_density, error_density), ('Correct', 'Error'), category_colors):
                response_label = f'{curve_label} / {category_label}'
                density_axis.plot(
                    x_values, density, color=color, linewidth=2,
                    label=f'{response_label} — Fitted model')
                fitted_cdf = cumulative_trapezoid(density, x_values, initial=0.0)
                cdf_axis.plot(
                    x_values, fitted_cdf, color=color, linewidth=2,
                    label=f'{response_label} — Fitted model')
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



class DiagnosticCurveThread(QThread):
    """Prepare the selected condition and display modes; never access a widget or Matplotlib artist."""

    def __init__(self, jobs, generation):
        # Application ownership keeps a closing/destroyed result window from destroying a running thread.
        super().__init__(QApplication.instance())
        self.jobs = jobs
        self.generation = generation
        self.results = {}
        QApplication.instance().aboutToQuit.connect(self.finishBeforeQuit)

    def run(self):
        try:
            for key, record in self.jobs:
                if self.isInterruptionRequested():
                    break
                try:
                    payload = _CurveBuilder(key[1]).prepare(record, self.isInterruptionRequested)
                except Exception as error:
                    payload = {'commands': [], 'status': f'Curve unavailable: {html.escape(str(error))}',
                               'failed': True}
                if self.isInterruptionRequested():
                    break
                self.results[key] = payload
        finally:
            self.jobs = []

    def finishBeforeQuit(self):
        """Join only during final application shutdown, never during normal GUI interaction."""
        self.requestInterruption()
        self.wait()


class FitDetailsBrowser(QTextBrowser):
    """Display rich fit details in a bounded, read-only scrolling region."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._details_text = ''
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet('background: transparent;')
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setOpenLinks(False)

    def setText(self, text):
        """Replace details and reset scrolling without changing the region's height."""
        self._details_text = text
        self.setHtml(text)
        self.verticalScrollBar().setValue(0)

    def text(self):
        """Return the original rich text for diagnostics inspection."""
        return self._details_text


class RTFitDiagnosticsDialog(QDialog):
    """Show empirical RT distributions alongside their fitted PDF and CDF."""

    _correct_response_weights = staticmethod(_CurveBuilder._correct_response_weights)
    _number = staticmethod(_CurveBuilder._number)

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
        """Initialize the single-condition diagnostics browser.

        :param fit_records: Fitted records available in the left list.
        :param parent: Optional parent widget.
        :param initial_index: Record selected when opened from a result cell.
        :return: None.
        """
        super().__init__(parent)
        self.fit_records = fit_records
        self._curve_worker = None
        self._request_generation = 0
        self._requested_selection = (-1, ())
        self._closed = False
        self._updating_modes = False
        self._mode_preferences = {}
        self.setWindowTitle('RT Fit Diagnostics')
        self.resize(1120, 820)

        self.mode_panel = QWidget()
        mode_row = QHBoxLayout(self.mode_panel)
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(QLabel('Display By:'))
        self.response_mode_button = QCheckBox('Response')
        self.accuracy_mode_button = QCheckBox('Correct / Error')
        mode_row.addWidget(self.response_mode_button)
        mode_row.addWidget(self.accuracy_mode_button)
        mode_row.addStretch()

        self.group_list = QListWidget()
        self.group_list.setAlternatingRowColors(True)
        self.group_list.setWordWrap(True)
        self.group_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.group_list.setMinimumWidth(300)
        self.group_list.setToolTip('Select one model and Rows × Columns combination.')
        for record in fit_records:
            prefix = record.get('result_prefix', '')
            variable = prefix.split('@', 1)[0] or record.get('specification', {}).get('rt_variable', 'RT')
            groups = '\n'.join(f'{name} = {value}' for name, value in zip(
                record.get('group_vars', []), record.get('group_values', ())))
            text = f"Data: {variable}\nModel: {record['distribution']}"
            if groups:
                text += '\n' + groups
            item = QListWidgetItem(text)
            item.setToolTip(text)
            self.group_list.addItem(item)
        self.status_label = FitDetailsBrowser()
        # Reserve room once for this result set, not for whichever record is currently selected.
        self._detail_line_budget = 4
        for record in fit_records:
            text = ' '.join(str(record.get(key, '')) for key in (
                'distribution', 'group_vars', 'group_values', 'boundary_details',
                'shift_warning_details', 'response_counts'))
            lines = 3 + (len(text) + 79) // 80
            if record.get('model') in COGNITIVE_MODEL_NAMES:
                lines += 4
            self._detail_line_budget = min(10, max(self._detail_line_budget, lines))
        self._resize_details_region()
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        group_panel = QWidget()
        group_layout = QVBoxLayout(group_panel)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_heading = QLabel('Data × Model × Rows × Columns combinations:')
        group_heading.setWordWrap(True)
        group_layout.addWidget(group_heading)
        group_layout.addWidget(self.group_list, 1)
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.status_label)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(group_panel)
        splitter.addWidget(plot_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 790])
        self.close_button = QPushButton('Close')
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_layout.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.mode_panel)
        layout.addWidget(splitter, 1)
        layout.addLayout(close_layout)
        self.group_list.currentRowChanged.connect(self._selection_changed)
        self.response_mode_button.toggled.connect(self._mode_changed)
        self.accuracy_mode_button.toggled.connect(self._mode_changed)
        self.close_button.clicked.connect(self.close)
        if fit_records:
            self.group_list.setCurrentRow(min(max(initial_index, 0), len(fit_records) - 1))
        else:
            self.response_mode_button.setEnabled(False)
            self.accuracy_mode_button.setEnabled(False)
            self.status_label.setText('No fitted conditions available.')
        self._position_within_available_screen()

    @staticmethod
    def _bounded_geometry(size, center, available):
        """Return a centered window rectangle constrained to one available screen."""
        width = min(max(1, size.width()), max(1, available.width()))
        height = min(max(1, size.height()), max(1, available.height()))
        maximum_x = available.right() - width + 1
        maximum_y = available.bottom() - height + 1
        x = min(max(center.x() - width // 2, available.left()), maximum_x)
        y = min(max(center.y() - height // 2, available.top()), maximum_y)
        return QRect(x, y, width, height)

    def _position_within_available_screen(self):
        """Center on the parent window's screen without crossing its usable bounds."""
        parent = self.parentWidget()
        anchor_window = parent.window() if parent is not None else None
        primary_screen = QApplication.primaryScreen()
        if anchor_window is not None:
            center = anchor_window.frameGeometry().center()
        else:
            if primary_screen is None:
                return
            center = primary_screen.availableGeometry().center()

        screen = QApplication.screenAt(center) or primary_screen
        if screen is None:
            return
        self.winId()
        client_geometry = self.geometry()
        frame_geometry = self.frameGeometry()
        left_margin = max(0, client_geometry.left() - frame_geometry.left())
        top_margin = max(0, client_geometry.top() - frame_geometry.top())
        right_margin = max(0, frame_geometry.right() - client_geometry.right())
        bottom_margin = max(0, frame_geometry.bottom() - client_geometry.bottom())
        available = screen.availableGeometry()
        client_width = min(
            self.width(), max(1, available.width() - left_margin - right_margin))
        client_height = min(
            self.height(), max(1, available.height() - top_margin - bottom_margin))
        frame_size = QSize(
            client_width + left_margin + right_margin,
            client_height + top_margin + bottom_margin)
        frame_target = self._bounded_geometry(frame_size, center, available)
        self.setGeometry(
            frame_target.left() + left_margin,
            frame_target.top() + top_margin,
            client_width,
            client_height,
        )

    def showEvent(self, event):
        """Revalidate placement in case monitor geometry changed after construction."""
        self._position_within_available_screen()
        super().showEvent(event)

    def _resize_details_region(self):
        """Bound the shared details height while reserving most of the window for plots."""
        line_height = self.status_label.fontMetrics().lineSpacing()
        reserved = self._detail_line_budget * line_height + 8
        cap = max(3 * line_height + 8, int(self.height() * 0.25))
        self.status_label.setFixedHeight(min(reserved, cap))

    def resizeEvent(self, event):
        """Adapt the fixed details region only when the actual window size changes."""
        super().resizeEvent(event)
        if hasattr(self, 'status_label'):
            self._resize_details_region()

    def _selection_changed(self, _row=None):
        """Configure available views for the newly selected model and request its curves."""
        index = self.group_list.currentRow()
        if index < 0:
            return
        record = self.fit_records[index]
        cognitive = record.get('model') in COGNITIVE_MODEL_NAMES
        supported = (record.get('model') == RATCLIFF_MODEL
                     and bool(record.get('specification', {}).get('accuracy_variable'))
                     and record.get('accuracy') is not None)
        self._updating_modes = True
        self.response_mode_button.setEnabled(cognitive)
        self.response_mode_button.setToolTip(
            'Show response-specific curves of the fitted model.' if cognitive else
            'This distribution fits Overall RTs, without response-specific views.')
        self.accuracy_mode_button.setEnabled(supported)
        self.accuracy_mode_button.setToolTip(
            'Show Correct/Error views of the same shared fit.' if supported else
            'Correct/Error diagnostics require a Ratcliff model with Accuracy data.')
        response, accuracy = self._mode_preferences.get(index, (True, supported))
        accuracy = accuracy and supported
        response = cognitive and (response or not accuracy)
        self.response_mode_button.setChecked(response)
        self.accuracy_mode_button.setChecked(accuracy)
        self._updating_modes = False
        self._draw_selected_records()

    def _mode_changed(self, _checked):
        """Keep at least one supported cognitive view selected without a warning popup."""
        if self._updating_modes:
            return
        if not self.response_mode_button.isChecked() and not self.accuracy_mode_button.isChecked():
            self._updating_modes = True
            self.sender().setChecked(True)
            self._updating_modes = False
            return
        self._mode_preferences[self.group_list.currentRow()] = (
            self.response_mode_button.isChecked(), self.accuracy_mode_button.isChecked())
        self._draw_selected_records()

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
        """Request the single selected record immediately, without caching or debounce."""
        if self._closed:
            return
        index = self.group_list.currentRow()
        if index < 0:
            return
        record = self.fit_records[index]
        cognitive = record.get('model') in COGNITIVE_MODEL_NAMES
        modes = tuple(mode for mode, enabled in (
            (False, self.response_mode_button.isChecked()),
            (True, self.accuracy_mode_button.isChecked())) if enabled) if cognitive else (False,)
        self._request_generation += 1
        self._requested_selection = (index, modes)
        self.figure.clear()
        self.canvas.draw_idle()
        self.status_label.setText(html.escape(self._record_label(record)) + '<br>Updating curves…')
        if self._curve_worker is not None:
            self._curve_worker.requestInterruption()
        else:
            self._start_requested_curves()

    def _start_requested_curves(self):
        """Run one application-owned worker for the latest record and selected modes."""
        if self._closed:
            return
        index, modes = self._requested_selection
        worker = DiagnosticCurveThread(
            [((index, mode), self.fit_records[index]) for mode in modes], self._request_generation)
        self._curve_worker = worker
        worker.finished.connect(self._curves_prepared)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _curves_prepared(self):
        """Render only the latest request; discard obsolete numeric payloads."""
        worker = self.sender()
        if worker is not self._curve_worker:
            return
        self._curve_worker = None
        if not self._closed:
            if worker.generation == self._request_generation:
                self._render_selected_records(worker.results)
            else:
                self._start_requested_curves()
        worker.results = {}

    def closeEvent(self, event):
        """Cancel safely without joining numerical work during normal closure."""
        self._closed = True
        self._request_generation += 1
        if self._curve_worker is not None:
            self._curve_worker.requestInterruption()
        super().closeEvent(event)

    def _render_selected_records(self, results):
        """Create a Density/CDF pair for each selected view on the GUI thread."""
        index, modes = self._requested_selection
        record = self.fit_records[index]
        cognitive = record.get('model') in COGNITIVE_MODEL_NAMES
        self.figure.clear()
        axes = self.figure.subplots(len(modes), 2, squeeze=False)
        status = ''
        for row, accuracy_mode in enumerate(modes):
            payload = results[(index, accuracy_mode)]
            # Correct/Error explanations take precedence; the shared-fit summary appears once.
            status = payload['status'].replace(_RECORD_LABEL, html.escape(self._record_label(record)), 1)
            count = (2 if accuracy_mode else max(1, len(record.get('specification', {}).get('response_values', [])))) if cognitive else 1
            palette = self._curve_colors(count)
            for axis_index, method, args, prepared_options in payload['commands']:
                options = dict(prepared_options)
                for property_name in ('color', 'edgecolor'):
                    if property_name in options:
                        options[property_name] = palette[options[property_name]]
                if 'label' in options:
                    options['label'] = options['label'].replace(_CURVE_LABEL, '', 1).lstrip(' /—')
                if method == 'stairs':
                    options.pop('edgecolor', None)
                getattr(axes[row, axis_index], method)(*args, **options)
            view = ('Correct / Error' if accuracy_mode else 'Response') if cognitive else 'Overall'
            density_axis, cdf_axis = axes[row]
            density_axis.set_title(f'{view} RT Density', fontsize=10)
            cdf_axis.set_title(f'{view} RT CDF', fontsize=10)
            density_axis.set_ylabel('Density')
            cdf_axis.set_ylabel('Cumulative probability')
            cdf_axis.set_ylim(-0.02, 1.02)
            for axis in (density_axis, cdf_axis):
                axis.set_xlabel('Reaction time')
                if axis.get_legend_handles_labels()[0]:
                    axis.legend(loc='best', fontsize=8)
        self.status_label.setText(status)
        self.canvas.draw_idle()

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
