import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QListWidget, \
    QListWidgetItem, QPushButton, QSplitter, QVBoxLayout, QWidget
from scipy.stats import gaussian_kde


class DistributionPreviewDialog(QDialog):
    """Compare retained and excluded observations before and after filtering."""

    RETAINED_COLOR = '#0072BD'
    EXCLUDED_COLOR = '#D95319'
    BEFORE_COLOR = '#7F7F7F'
    def __init__(self, data, retained_mask, target_variables, row_facets=None,
                 column_facets=None, parent=None):
        """Initialize the interactive distribution preview.

        :param data: Original unfiltered data frame.
        :param retained_mask: Boolean mask marking rows retained by the active filters.
        :param target_variables: Numeric variables configured in the Summary Data area.
        :param row_facets: Variables configured in the Summary Rows area.
        :param column_facets: Variables configured in the Summary Columns area.
        :param parent: Optional parent widget.
        :return: None.
        """
        super().__init__(parent)
        self.data = data.reset_index(drop=True).copy()
        self.retained_mask = np.asarray(retained_mask, dtype=bool)
        if len(self.data) != len(self.retained_mask):
            raise ValueError('The retained-row mask must have one value per original row.')
        self.target_variables = [
            variable for variable in target_variables
            if variable in self.data.columns
            and pd.to_numeric(self.data[variable], errors='coerce').notna().any()]
        if not self.target_variables:
            raise ValueError('Distribution Preview requires at least one numeric Data variable.')
        self.row_facets = [
            facet for facet in (row_facets or []) if facet in self.data.columns]
        self.column_facets = [
            facet for facet in (column_facets or [])
            if facet in self.data.columns and facet not in self.row_facets]

        self.setWindowTitle('Distribution Preview')
        self.resize(1120, 820)
        self.violin_check = QCheckBox('Violin + Box + Raw Points')
        self.histogram_check = QCheckBox('Histogram + KDE')
        self.ecdf_check = QCheckBox('ECDF')
        self.histogram_check.setToolTip('Histogram + Kernel Density Estimate (KDE)')
        self.ecdf_check.setToolTip('Empirical Cumulative Distribution Function (ECDF)')
        self.violin_check.setChecked(True)
        self.group_list = QListWidget()
        self.group_list.setAlternatingRowColors(True)
        self.group_list.setWordWrap(True)
        self.group_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.group_list.setMinimumWidth(300)
        self.group_list.setToolTip(
            'Select one Rows × Columns combination. Counts use finite observations for the selected variable.')
        self.selected_group_label = QLabel()
        self.selected_group_label.setWordWrap(True)
        self.info_label = QLabel(
            '<span style="color:#0072BD">■</span> Retained &nbsp;&nbsp; '
            '<span style="color:#D95319">■</span> Excluded &nbsp;&nbsp; '
            'Before and After panels use the same scale within each facet.')
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.close_button = QPushButton('Close')

        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel('Plots:'))
        plot_controls.addWidget(self.violin_check)
        plot_controls.addWidget(self.histogram_check)
        plot_controls.addWidget(self.ecdf_check)
        plot_controls.addStretch()

        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_layout.addWidget(self.close_button)

        group_panel = QWidget()
        group_layout = QVBoxLayout(group_panel)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.addWidget(QLabel('Data × Rows × Columns combinations:'))
        group_layout.addWidget(self.group_list, 1)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.selected_group_label)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)

        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.addWidget(group_panel)
        content_splitter.addWidget(plot_panel)
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([330, 790])

        layout = QVBoxLayout(self)
        layout.addLayout(plot_controls)
        layout.addWidget(self.info_label)
        layout.addWidget(content_splitter, 1)
        layout.addLayout(close_layout)

        self.violin_check.toggled.connect(self._plotSelectionChanged)
        self.histogram_check.toggled.connect(self._plotSelectionChanged)
        self.ecdf_check.toggled.connect(self._plotSelectionChanged)
        self.group_list.currentRowChanged.connect(self._drawSelectedGroup)
        self.close_button.clicked.connect(self.close)
        self.refreshPreview()

    def _selectedPlotTypes(self):
        """Return checked plot types in their stable display order."""
        plot_types = []
        if self.violin_check.isChecked():
            plot_types.append('Violin + Box + Raw Points')
        if self.histogram_check.isChecked():
            plot_types.append('Histogram + KDE')
        if self.ecdf_check.isChecked():
            plot_types.append('ECDF')
        return plot_types

    def _plotSelectionChanged(self):
        """Keep at least one plot selected and redraw the active group."""
        if not self._selectedPlotTypes():
            checkbox = self.sender()
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        self._drawSelectedGroup()

    def _facetGroups(self):
        """Return facet labels and integer row positions."""
        row_columns, column_columns = self.row_facets, self.column_facets
        facet_columns = row_columns + column_columns
        if not facet_columns:
            return [('Overall', np.arange(len(self.data), dtype=int))]

        grouping_frame = self.data[facet_columns].copy()
        grouping_frame['_preview_position'] = np.arange(len(self.data), dtype=int)
        groups = []
        grouper = facet_columns[0] if len(facet_columns) == 1 else facet_columns
        for values, frame in grouping_frame.groupby(grouper, sort=False, dropna=False):
            value_tuple = values if isinstance(values, tuple) else (values,)
            row_values = value_tuple[:len(row_columns)]
            column_values = value_tuple[len(row_columns):]
            label_parts = []
            if row_columns:
                label_parts.append('Rows: ' + ', '.join(
                    f'{column}={"NA" if pd.isna(value) else value}'
                    for column, value in zip(row_columns, row_values)))
            if column_columns:
                label_parts.append('Columns: ' + ', '.join(
                    f'{column}={"NA" if pd.isna(value) else value}'
                    for column, value in zip(column_columns, column_values)))
            label = ' | '.join(label_parts)
            groups.append((label, frame['_preview_position'].to_numpy(dtype=int)))
        return groups

    def _groupCounts(self, positions, target):
        """Return finite before, excluded, and retained counts for one group."""
        numeric = pd.to_numeric(self.data.iloc[positions][target], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(numeric)
        retained = int(np.sum(self.retained_mask[positions][valid]))
        before = int(np.sum(valid))
        return before, before - retained, retained

    def refreshPreview(self):
        """Rebuild the group list and redraw the selected Rows × Columns combination."""
        self.groups = self._facetGroups()
        self.preview_items = []
        previous_row = max(0, self.group_list.currentRow())
        self.group_list.blockSignals(True)
        self.group_list.clear()
        for target in self.target_variables:
            for label, positions in self.groups:
                before, excluded, retained = self._groupCounts(positions, target)
                display_label = label.replace(' | ', '\n')
                item = QListWidgetItem(
                    f'Data: {target}\n{display_label}\nN before: {before}   '
                    f'N excluded: {excluded}   N retained: {retained}')
                item.setData(Qt.UserRole, len(self.preview_items))
                item.setToolTip(item.text())
                self.group_list.addItem(item)
                self.preview_items.append((target, label, positions))
        selected_row = min(previous_row, max(0, self.group_list.count() - 1))
        self.group_list.setCurrentRow(selected_row)
        self.group_list.blockSignals(False)
        self._drawSelectedGroup()

    def _drawSelectedGroup(self, _row=None):
        """Draw checked plot types for the group selected in the left list."""
        if not getattr(self, 'preview_items', None) or self.group_list.currentRow() < 0:
            return
        preview_index = self.group_list.currentItem().data(Qt.UserRole)
        target, label, positions = self.preview_items[preview_index]
        before_count, excluded_count, retained_count = self._groupCounts(positions, target)
        self.selected_group_label.setText(
            f'Data: {target}   |   {label}<br>'
            f'N before: {before_count}   N excluded: {excluded_count}   '
            f'N retained: {retained_count}')

        numeric = pd.to_numeric(self.data.iloc[positions][target], errors='coerce').to_numpy(dtype=float)
        valid = np.isfinite(numeric)
        status = self.retained_mask[positions][valid]
        before_values = numeric[valid]
        retained_values = before_values[status]
        excluded_values = before_values[~status]
        plot_types = self._selectedPlotTypes()
        self.figure.clear()
        axes = self.figure.subplots(len(plot_types), 2, squeeze=False)
        for plot_index, plot_type in enumerate(plot_types):
            before_axis, after_axis = axes[plot_index]
            if plot_type == 'Violin + Box + Raw Points':
                self._drawViolin(before_axis, before_values, retained_values, excluded_values, before=True)
                self._drawViolin(after_axis, retained_values, retained_values, np.array([]), before=False)
            elif plot_type == 'Histogram + KDE':
                bins = np.histogram_bin_edges(before_values, bins='auto') if before_values.size else 10
                self._drawHistogram(before_axis, before_values, retained_values, excluded_values, bins, before=True)
                self._drawHistogram(after_axis, retained_values, retained_values, np.array([]), bins, before=False)
            else:
                self._drawEcdf(before_axis, before_values, retained_values, excluded_values, before=True)
                self._drawEcdf(after_axis, retained_values, retained_values, np.array([]), before=False)

            short_plot_name = plot_type.replace(' + Raw Points', '')
            before_axis.set_title(f'{short_plot_name} | Before filtering', fontsize=10)
            after_axis.set_title(f'{short_plot_name} | After filtering', fontsize=10)
            before_axis.set_xlabel(target)
            after_axis.set_xlabel(target)
            if before_values.size:
                lower, upper = float(np.min(before_values)), float(np.max(before_values))
                padding = max((upper - lower) * 0.05, abs(lower) * 0.01, 1e-6)
                before_axis.set_xlim(lower - padding, upper + padding)
                after_axis.set_xlim(lower - padding, upper + padding)

        self.canvas.draw_idle()

    def _drawViolin(self, axis, values, retained, excluded, before):
        """Draw a violin, boxplot, and deterministic jittered observations."""
        if values.size == 0:
            axis.text(0.5, 0.5, 'No valid observations', ha='center', va='center', transform=axis.transAxes)
            return
        body_color = self.BEFORE_COLOR if before else self.RETAINED_COLOR
        if values.size >= 2 and np.unique(values).size >= 2:
            try:
                violin = axis.violinplot(
                    values, positions=[0], orientation='horizontal', showextrema=False, widths=0.75)
            except TypeError:
                violin = axis.violinplot(
                    values, positions=[0], vert=False, showextrema=False, widths=0.75)
            for body in violin['bodies']:
                body.set_facecolor(body_color)
                body.set_edgecolor(body_color)
                body.set_alpha(0.22)
        boxplot_options = {
            'positions': [0],
            'widths': 0.22,
            'patch_artist': True,
            'showfliers': False,
            'boxprops': {'facecolor': 'white', 'alpha': 0.75},
            'medianprops': {'color': 'black'},
        }
        try:
            axis.boxplot(values, orientation='horizontal', **boxplot_options)
        except TypeError:
            axis.boxplot(values, vert=False, **boxplot_options)
        rng = np.random.default_rng(20260822)
        if retained.size:
            axis.scatter(retained, rng.normal(0, 0.045, retained.size), s=12, alpha=0.55,
                         color=self.RETAINED_COLOR, label='Retained')
        if excluded.size:
            axis.scatter(excluded, rng.normal(0, 0.045, excluded.size), s=18, alpha=0.85,
                         color=self.EXCLUDED_COLOR, label='Excluded', zorder=4)
        axis.set_yticks([])
        axis.legend(loc='best', fontsize=8)

    def _drawHistogram(self, axis, values, retained, excluded, bins, before):
        """Draw a density histogram with a kernel-density estimate."""
        if values.size == 0:
            axis.text(0.5, 0.5, 'No valid observations', ha='center', va='center', transform=axis.transAxes)
            return
        if before and excluded.size:
            datasets, labels, colors = [], [], []
            if retained.size:
                datasets.append(retained)
                labels.append('Retained')
                colors.append(self.RETAINED_COLOR)
            datasets.append(excluded)
            labels.append('Excluded')
            colors.append(self.EXCLUDED_COLOR)
            axis.hist(datasets, bins=bins, density=True, stacked=True, alpha=0.42,
                      color=colors, label=labels)
            self._plotKde(axis, values, self.BEFORE_COLOR, 'Before KDE')
        else:
            axis.hist(values, bins=bins, density=True, alpha=0.35,
                      color=self.RETAINED_COLOR, label='Retained')
            self._plotKde(axis, values, self.RETAINED_COLOR, 'Retained KDE')
        axis.set_ylabel('Density')
        axis.legend(loc='best', fontsize=8)

    @staticmethod
    def _plotKde(axis, values, color, label):
        """Plot a KDE when the sample contains enough variation."""
        if values.size < 2 or np.unique(values).size < 2:
            return
        try:
            kde = gaussian_kde(values)
            x_values = np.linspace(float(np.min(values)), float(np.max(values)), 300)
            axis.plot(x_values, kde(x_values), color=color, linewidth=2, label=label)
        except (ValueError, np.linalg.LinAlgError):
            return

    def _drawEcdf(self, axis, values, retained, excluded, before):
        """Draw empirical cumulative distributions for retained and excluded observations."""
        if values.size == 0:
            axis.text(0.5, 0.5, 'No valid observations', ha='center', va='center', transform=axis.transAxes)
            return
        if before:
            self._plotEcdf(axis, values, self.BEFORE_COLOR, 'All before', '-', 1.6)
            self._plotEcdf(axis, retained, self.RETAINED_COLOR, 'Retained', '--', 1.8)
            self._plotEcdf(axis, excluded, self.EXCLUDED_COLOR, 'Excluded', '--', 1.8)
        else:
            self._plotEcdf(axis, retained, self.RETAINED_COLOR, 'Retained', '-', 2.0)
        axis.set_ylabel('Cumulative probability')
        axis.set_ylim(-0.02, 1.02)
        axis.legend(loc='best', fontsize=8)

    @staticmethod
    def _plotEcdf(axis, values, color, label, linestyle, linewidth):
        """Plot one empirical cumulative distribution."""
        if values.size == 0:
            return
        sorted_values = np.sort(values)
        probabilities = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        axis.step(sorted_values, probabilities, where='post', color=color, label=label,
                  linestyle=linestyle, linewidth=linewidth)
