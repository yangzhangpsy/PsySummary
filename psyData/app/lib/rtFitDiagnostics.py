import html

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import hsv_to_rgb, to_hex
from matplotlib.figure import Figure
from PyQt5.QtCore import QRect, QSize, Qt, QThread
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QSplitter, QTextBrowser, QFrame, QVBoxLayout, QWidget,
)

from app.cognitiveModelSpec import (
    COGNITIVE_MODEL_NAMES, RATCLIFF_MODEL,
)
from app.rtDiagnosticCurves import _CurveBuilder, _CURVE_LABEL, _RECORD_LABEL
from app.diagnosticCurveCache import (
    DiagnosticCurveCache, attach_cache, peek_curves, prepared_curves,
)


class DiagnosticCurveThread(QThread):
    """Prepare the selected condition and display modes; never access a widget or Matplotlib artist."""

    def __init__(self, jobs, generation):
        # Application ownership keeps a closing/destroyed result window from destroying a running thread.
        super().__init__(QApplication.instance())
        self.jobs = jobs
        fallback_cache = DiagnosticCurveCache()
        for _, record in jobs:
            attach_cache(record, fallback_cache)
        self.generation = generation
        self.results = {}
        QApplication.instance().aboutToQuit.connect(self.finishBeforeQuit)

    def run(self):
        try:
            for key, record in self.jobs:
                if self.isInterruptionRequested():
                    break
                try:
                    payload = prepared_curves(record, key[1], self.isInterruptionRequested)
                except InterruptedError:
                    break
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
        # Old/uncached records acquire a result-owned fallback shared across reopenings.
        fallback_cache = DiagnosticCurveCache()
        for record in fit_records:
            attach_cache(record, fallback_cache)
        self._curve_worker = None
        self._request_generation = 0
        self._rendered_generation = -1
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
        """Render RAM hits immediately; load disk entries or rebuild misses in a worker."""
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
        if self._curve_worker is not None:
            self._curve_worker.requestInterruption()
        results = {(index, mode): peek_curves(record, mode) for mode in modes}
        if all(payload is not None for payload in results.values()):
            self._render_selected_records(results)
            return
        self.figure.clear()
        self.canvas.draw_idle()
        self.status_label.setText(html.escape(self._record_label(record)) + '<br>Updating curves…')
        if self._curve_worker is None:
            self._start_requested_curves()

    def _start_requested_curves(self):
        """Run one application-owned worker for the latest record and selected modes."""
        if self._closed:
            return
        index, modes = self._requested_selection
        results = {(index, mode): peek_curves(self.fit_records[index], mode) for mode in modes}
        if all(payload is not None for payload in results.values()):
            self._render_selected_records(results)
            return
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
            elif self._rendered_generation != self._request_generation:
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
        self._rendered_generation = self._request_generation
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
