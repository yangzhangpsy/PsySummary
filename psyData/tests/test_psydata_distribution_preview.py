import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/psysummary-test-matplotlib')

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication

from app.lib.distributionPreview import DistributionPreviewDialog
from PsySummary import PsyData


def preview_data():
    """Create two Summary row groups crossed with two column groups."""
    rows = []
    for participant in ['P1', 'P2']:
        for condition in ['A', 'B']:
            for trial in range(4):
                rows.append({
                    'participant': participant,
                    'condition': condition,
                    'trial': trial + 1,
                    'rt': 300.0 + trial * 30 + (40 if condition == 'B' else 0),
                })
    return pd.DataFrame(rows)


class DistributionPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_defaults_to_violin_and_inherits_summary_rows_and_columns(self):
        data = preview_data()
        retained = np.ones(len(data), dtype=bool)
        retained[[0, 7]] = False
        dialog = DistributionPreviewDialog(
            data,
            retained,
            target_variables=['rt'],
            row_facets=['participant'],
            column_facets=['condition'],
        )

        self.assertTrue(dialog.violin_check.isChecked())
        self.assertFalse(dialog.histogram_check.isChecked())
        self.assertFalse(dialog.ecdf_check.isChecked())
        self.assertEqual(
            dialog.histogram_check.toolTip(), 'Histogram + Kernel Density Estimate (KDE)')
        self.assertEqual(
            dialog.ecdf_check.toolTip(), 'Empirical Cumulative Distribution Function (ECDF)')
        self.assertEqual(dialog.target_variables, ['rt'])
        self.assertEqual(dialog.row_facets, ['participant'])
        self.assertEqual(dialog.column_facets, ['condition'])
        self.assertFalse(hasattr(dialog, 'target_combo'))
        groups = dialog._facetGroups()
        self.assertEqual(len(groups), 4)
        self.assertIn('Rows: participant=P1', groups[0][0])
        self.assertIn('Columns: condition=A', groups[0][0])
        self.assertEqual(dialog.group_list.count(), 4)
        self.assertIn('N before: 4', dialog.group_list.item(0).text())
        self.assertIn('N excluded: 1', dialog.group_list.item(0).text())

    def test_checked_plot_types_are_combined_and_paginated(self):
        data = preview_data()
        retained = np.ones(len(data), dtype=bool)
        retained[0] = False
        dialog = DistributionPreviewDialog(
            data,
            retained,
            target_variables=['rt'],
            row_facets=['participant'],
            column_facets=['condition'],
        )
        dialog.histogram_check.setChecked(True)
        dialog.ecdf_check.setChecked(True)

        self.assertEqual(len(dialog._selectedPlotTypes()), 3)
        self.assertEqual(len(dialog.figure.axes), 6)
        self.assertTrue(any('ECDF' in axis.get_title() for axis in dialog.figure.axes))

        dialog.group_list.setCurrentRow(1)
        self.assertIn('Columns: condition=B', dialog.selected_group_label.text())
        self.assertEqual(len(dialog.figure.axes), 6)

    def test_violin_boxplot_does_not_duplicate_outliers_as_black_fliers(self):
        data = pd.DataFrame({'rt': [100.0, 101.0, 102.0, 103.0, 1000.0]})
        dialog = DistributionPreviewDialog(
            data,
            np.ones(len(data), dtype=bool),
            target_variables=['rt'],
        )

        for axis in dialog.figure.axes:
            self.assertFalse(any(line.get_marker() == 'o' for line in axis.lines))

    def test_main_preview_uses_the_actual_filter_to_mark_excluded_rows(self):
        window = PsyData()
        window.data = pd.DataFrame({'rt': [250.0, 400.0, 650.0]})
        window.data_list.addItem('rt@Mean')
        window.filter_list.addItem('rt:< 500')
        with patch('PsySummary.DistributionPreviewDialog') as preview_dialog:
            preview_dialog.return_value.show.return_value = None
            self.assertTrue(window.showDistributionPreview())

        call = preview_dialog.call_args
        np.testing.assert_array_equal(call.args[1], np.array([True, True, False]))
        pd.testing.assert_frame_equal(call.args[0], window.data)
        self.assertEqual(call.kwargs['target_variables'], ['rt'])
        window.close()

    def test_every_data_variable_is_crossed_with_summary_groups(self):
        data = preview_data()
        data['accuracy'] = np.tile([0.0, 1.0], len(data) // 2)
        dialog = DistributionPreviewDialog(
            data,
            np.ones(len(data), dtype=bool),
            target_variables=['rt', 'accuracy'],
            row_facets=['participant'],
            column_facets=['condition'],
        )

        self.assertEqual(dialog.group_list.count(), 8)
        self.assertIn('Data: rt', dialog.group_list.item(0).text())
        self.assertIn('Data: accuracy', dialog.group_list.item(4).text())

    def test_main_preview_requires_a_numeric_data_variable(self):
        window = PsyData()
        window.data = pd.DataFrame({'condition': ['A', 'B']})
        with patch('PsySummary.MessageBox.information') as information:
            self.assertFalse(window.showDistributionPreview())

        self.assertIn('numeric Data variable', information.call_args.args[2])
        window.close()

    def test_main_preview_is_blocked_while_model_fitting(self):
        """Preview must not compete with an active background model fit."""
        window = PsyData()
        window.data = pd.DataFrame({'rt': [250.0, 400.0, 650.0]})
        window.data_list.addItem('rt@Mean')
        window.model_fit_running = True

        with patch.object(window, '_showModelFitBusyMessage') as busy_message, \
                patch('PsySummary.StatisticTool.filterData') as filter_data, \
                patch('PsySummary.DistributionPreviewDialog') as preview_dialog:
            self.assertFalse(window.showDistributionPreview())

        busy_message.assert_called_once_with('previewing filter effects')
        filter_data.assert_not_called()
        preview_dialog.assert_not_called()
        window.model_fit_running = False
        window.close()


if __name__ == '__main__':
    unittest.main()
