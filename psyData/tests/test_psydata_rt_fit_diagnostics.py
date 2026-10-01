import os
import unittest
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/psysummary-test-matplotlib')

import numpy as np
import pandas as pd
from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton

from app.lib.dataFrameTableWidget import ResultFrameTableWidget
from app.lib.pivotedDataWidget import PivotedDataWidget
from app.cognitiveModelSpec import (
    ACCURACY_CODING, RATCLIFF_MODEL, RESPONSE_CODING, make_model_specification,
)
from app.lib.rtFitDiagnostics import RTFitDiagnosticsDialog
from app.cognitiveModels import fitted_accuracy_pdf, fitted_response_pdf


def make_fit_record(condition, shape, scale):
    """Create a deterministic successful Gamma fit record for plotting tests."""
    data = np.linspace(100.0, 500.0, 40) + (10.0 if condition == 'B' else 0.0)
    return {
        'distribution': 'Gamma (k, θ)',
        'data': data,
        'parameters': np.array([shape, scale]),
        'n_valid': len(data),
        'converged': True,
        'parameter_boundary': 'No',
        'boundary_details': 'None',
        'shift_warning': 'No',
        'shift_warning_details': 'None',
        'log_likelihood': -200.0,
        'aic': 404.0,
        'bic': 407.0,
        'message': 'Synthetic fit',
        'group_vars': ['condition'],
        'group_values': (condition,),
        'result_prefix': 'rt@Gamma (k, θ)',
    }


def make_cognitive_fit_record(with_accuracy=True):
    """Create a deterministic DDM fit record for diagnostics tests."""
    specification = make_model_specification(
        RATCLIFF_MODEL, 'rt', 'response', ['left', 'right'], minimum_rt=0.35,
        accuracy_variable='accuracy' if with_accuracy else None)
    values = {
        'a': 1.2, 'v': 0.5, 't0': 0.2, 'z': 0.6, 'd': 0.0,
        'sz': 0.0, 'sv': 0.0, 'st0': 0.0, 's': 1.0,
    }
    return {
        'model': RATCLIFF_MODEL,
        'distribution': RATCLIFF_MODEL,
        'rt': np.array([0.35, 0.42, 0.51, 0.63, 0.72, 0.81]),
        'response': np.array([2, 1, 2, 1, 2, 1]),
        'accuracy': np.array([1, 1, 1, 1, 0, 0]) if with_accuracy else None,
        'parameters': [values[parameter['name']] for parameter in specification['parameters']],
        'parameter_names': [parameter['name'] for parameter in specification['parameters']],
        'n_valid': 6,
        'response_counts': {'left': 3, 'right': 3},
        'accuracy_rate': 2 / 3 if with_accuracy else None,
        'boundary_coding': specification['boundary_coding'],
        'correct_response_weights': np.array([0.5, 0.5]),
        'converged': True,
        'log_likelihood': -20.0,
        'aic': 46.0,
        'bic': 45.0,
        'specification': specification,
        'group_vars': [],
        'group_values': (),
        'result_prefix': f'rt@{RATCLIFF_MODEL}',
    }


class RTFitDiagnosticsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_opened_result_group_is_the_only_initial_selection(self):
        records = [make_fit_record('A', 2.0, 80.0), make_fit_record('B', 3.0, 60.0)]
        dialog = RTFitDiagnosticsDialog(records, initial_index=1)
        self.assertEqual(dialog.group_selector.checked_indices(), [1])

    def test_window_geometry_is_clamped_to_the_selected_screen(self):
        available = QRect(-1920, 0, 1920, 1080)
        geometry = RTFitDiagnosticsDialog._bounded_geometry(
            QSize(850, 680), QPoint(-1850, 300), available)
        self.assertTrue(available.contains(geometry))
        self.assertEqual(geometry.left(), available.left())
        self.assertEqual(geometry.top(), available.top())

        small_screen = QRect(0, 0, 800, 600)
        reduced = RTFitDiagnosticsDialog._bounded_geometry(
            QSize(850, 680), QPoint(100, 300), small_screen)
        self.assertEqual(reduced, small_screen)

    def test_multiple_groups_use_matlab_colors_and_dashed_empirical_cdfs(self):
        records = [make_fit_record('A', 2.0, 80.0), make_fit_record('B', 3.0, 60.0)]
        dialog = RTFitDiagnosticsDialog(records, initial_index=0)
        dialog.group_selector.model().item(1).setCheckState(Qt.Checked)
        dialog.group_selector._update_summary()
        dialog._draw_selected_records()

        density_axis, cdf_axis = dialog.figure.axes
        self.assertEqual(dialog.group_selector.lineEdit().text(), '2 conditions selected')
        self.assertEqual([line.get_color().lower() for line in density_axis.lines], ['#0072bd', '#d95319'])
        empirical_lines = [line for line in cdf_axis.lines if 'empirical CDF' in line.get_label()]
        self.assertEqual(len(empirical_lines), 2)
        self.assertTrue(all(line.get_linestyle() == '--' for line in empirical_lines))
        self.assertEqual([line.get_color().lower() for line in empirical_lines], ['#0072bd', '#d95319'])

    def test_correct_error_mode_is_disabled_without_accuracy_variable(self):
        dialog = RTFitDiagnosticsDialog([make_cognitive_fit_record(with_accuracy=False)])
        self.assertFalse(dialog.accuracy_mode_button.isEnabled())
        self.assertTrue(dialog.response_mode_button.isChecked())

    def test_coding_selects_the_default_diagnostic_display(self):
        accuracy_record = make_cognitive_fit_record()
        self.assertEqual(accuracy_record['boundary_coding'], ACCURACY_CODING)
        accuracy_dialog = RTFitDiagnosticsDialog([accuracy_record])
        self.assertTrue(accuracy_dialog.accuracy_mode_button.isChecked())
        response_record = make_cognitive_fit_record()
        response_record['boundary_coding'] = RESPONSE_CODING
        response_record['specification']['boundary_coding'] = RESPONSE_CODING
        response_dialog = RTFitDiagnosticsDialog([response_record])
        self.assertTrue(response_dialog.response_mode_button.isChecked())
        accuracy_dialog.close()
        response_dialog.close()

    def test_correct_error_mode_draws_observed_and_fitted_curves(self):
        dialog = RTFitDiagnosticsDialog([make_cognitive_fit_record(with_accuracy=True)])
        self.assertTrue(dialog.accuracy_mode_button.isEnabled())
        self.assertTrue(dialog.accuracy_mode_button.isChecked())
        dialog.accuracy_mode_button.click()

        density_axis, cdf_axis = dialog.figure.axes
        density_labels = [line.get_label() for line in density_axis.lines]
        cdf_labels = [line.get_label() for line in cdf_axis.lines]
        self.assertTrue(any('Correct — observed' in label for label in density_labels))
        self.assertTrue(any('Correct — model-implied' in label for label in density_labels))
        self.assertTrue(any('Error — observed' in label for label in density_labels))
        self.assertTrue(any('Error — model-implied' in label for label in density_labels))
        self.assertTrue(any('Correct — empirical' in label for label in cdf_labels))
        self.assertTrue(any('Error — model-implied' in label for label in cdf_labels))

    def test_correct_error_mode_tracks_selected_records(self):
        records = [make_cognitive_fit_record(with_accuracy=True), make_fit_record('A', 2.0, 80.0)]
        dialog = RTFitDiagnosticsDialog(records)
        self.assertTrue(dialog.accuracy_mode_button.isEnabled())
        dialog.accuracy_mode_button.click()

        dialog.group_selector.model().item(1).setCheckState(Qt.Checked)
        dialog.group_selector._update_summary()
        dialog._selection_changed()
        self.assertFalse(dialog.accuracy_mode_button.isEnabled())
        self.assertTrue(dialog.response_mode_button.isChecked())

    def test_correct_error_diagnostics_derive_oriented_curves_from_one_shared_fit(self):
        """Correct/error curves apply z mirroring without creating separate parameter sets."""
        record = make_cognitive_fit_record()
        record['accuracy'] = np.array([1, 1, 1, 1, 0, 0])
        record['accuracy_rate'] = 2 / 3
        dialog = RTFitDiagnosticsDialog([record])
        response_lines = [line for line in dialog.figure.axes[0].lines if 'fitted' in line.get_label()]
        for index, line in enumerate(response_lines, 1):
            np.testing.assert_allclose(line.get_ydata(), fitted_response_pdf(record, line.get_xdata(), index))
        dialog.accuracy_mode_button.click()
        lines = [
            line for line in dialog.figure.axes[0].lines
            if 'model-implied' in line.get_label()
        ]
        self.assertEqual(len(lines), 2)
        np.testing.assert_allclose(
            lines[0].get_ydata(), fitted_accuracy_pdf(record, lines[0].get_xdata(), True))
        np.testing.assert_allclose(
            lines[1].get_ydata(), fitted_accuracy_pdf(record, lines[1].get_xdata(), False))
        self.assertFalse(np.allclose(lines[0].get_ydata(), lines[1].get_ydata()))
        self.assertIn('same shared parameter set', dialog.status_label.text())
        self.assertIn('does not refit', dialog.status_label.text())
        self.assertNotIn('Independently fitted', dialog.status_label.text())
        dialog.close()

    def test_correct_response_weights_support_binary_and_stable_multichoice_designs(self):
        binary_weights, warning = RTFitDiagnosticsDialog._correct_response_weights(
            np.array([1, 2, 1, 2]), np.array([1, 0, 1, 1]), 2)
        np.testing.assert_allclose(binary_weights, [0.75, 0.25])
        self.assertEqual(warning, '')

        multi_weights, warning = RTFitDiagnosticsDialog._correct_response_weights(
            np.array([2, 1, 3, 2]), np.array([1, 0, 0, 1]), 3)
        np.testing.assert_allclose(multi_weights, [0.0, 1.0, 0.0])
        self.assertEqual(warning, '')

        unavailable, warning = RTFitDiagnosticsDialog._correct_response_weights(
            np.array([1, 2, 3, 2]), np.array([1, 1, 0, 0]), 3)
        self.assertIsNone(unavailable)
        self.assertIn('multi-response model', warning)

    def test_multiple_cognitive_records_receive_unique_response_colors(self):
        first = make_cognitive_fit_record(with_accuracy=True)
        second = make_cognitive_fit_record(with_accuracy=True)
        first['group_vars'] = ['condition']
        first['group_values'] = ('A',)
        second['group_vars'] = ['condition']
        second['group_values'] = ('B',)
        dialog = RTFitDiagnosticsDialog([first, second])
        dialog.response_mode_button.click()
        dialog.group_selector.model().item(1).setCheckState(Qt.Checked)
        dialog.group_selector._update_summary()
        dialog._selection_changed()

        density_axis = dialog.figure.axes[0]
        fitted_lines = [line for line in density_axis.lines if 'fitted' in line.get_label()]
        colors = [line.get_color().lower() for line in fitted_lines]
        self.assertEqual(len(colors), 4)
        self.assertEqual(len(set(colors)), 4)

    def test_curve_palette_expands_without_repeating_colors(self):
        colors = RTFitDiagnosticsDialog._curve_colors(40)
        self.assertEqual(len(colors), 40)
        self.assertEqual(len({color.lower() for color in colors}), 40)

    def test_converged_cell_has_a_single_click_view_fit_button(self):
        record = make_fit_record('A', 2.0, 80.0)
        convergence = pd.DataFrame(
            {'value': ['Yes']}, index=pd.Index(['A'], name='condition'))
        table = ResultFrameTableWidget(
            [convergence], [], ['condition'],
            ['rt@Gamma (k, θ) Converged'], [record])

        buttons = table.findChildren(QPushButton)
        self.assertEqual([button.text() for button in buttons], ['View Fit'])
        self.assertEqual(table.item(2, 1).text(), '')
        self.assertEqual(table.cellWidget(2, 1).findChild(QLabel).text(), 'Yes')
        self.assertEqual(table.cellText(2, 1), 'Yes')
        table.updateTable(2)
        self.assertEqual(table.item(2, 1).text(), '')
        self.assertEqual(table.cellWidget(2, 1).findChild(QLabel).text(), 'Yes')
        clipboard_text = PivotedDataWidget.getTextData(
            SimpleNamespace(filterStr='', table=table))
        self.assertIn('A\tYes\t', clipboard_text)
        activated = []
        table.fitRecordActivated.connect(activated.append)
        buttons[0].click()
        self.assertEqual(len(activated), 1)
        self.assertIs(activated[0], record)


if __name__ == '__main__':
    unittest.main()
