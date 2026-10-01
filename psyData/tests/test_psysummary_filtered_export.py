import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/psysummary-test-matplotlib')

import pandas as pd
from PyQt5.QtWidgets import QApplication

from PsySummary import PsyData


class PsySummaryFilteredExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.window = PsyData()
        self.window.data = pd.DataFrame({
            'condition': ['compatible', 'incompatible'],
            'rt': [350.5, 412.0],
        })

    def tearDown(self):
        self.window.close()
        self.temporary_directory.cleanup()

    def test_save_filtered_data_supports_standard_csv_and_adds_extension(self):
        requested_path = os.path.join(self.temporary_directory.name, 'filtered_rows')
        with patch(
                'PsySummary.QFileDialog.getSaveFileName',
                return_value=(requested_path, 'CSV Files (*.csv)')), \
                patch('PsySummary.PsyDataFunc.genScript') as generate_script:
            self.window.saveFilteredData()

        output_path = requested_path + '.csv'
        exported = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(exported, self.window.data)
        self.assertIn("to_csv('" + output_path, generate_script.call_args_list[-1].args[0])
        self.assertNotIn("sep='|'", generate_script.call_args_list[-1].args[0])

    def test_save_filtered_data_retains_psydata_pipe_format(self):
        output_path = os.path.join(self.temporary_directory.name, 'filtered_rows.psydata')
        with patch(
                'PsySummary.QFileDialog.getSaveFileName',
                return_value=(output_path, 'psyData Files (*.psydata)')), \
                patch('PsySummary.PsyDataFunc.genScript') as generate_script:
            self.window.saveFilteredData()

        exported = pd.read_csv(output_path, sep='|')
        pd.testing.assert_frame_equal(exported, self.window.data)
        self.assertIn("sep='|'", generate_script.call_args_list[-1].args[0])

    def test_export_filtered_does_not_replace_or_filter_loaded_data(self):
        original_data = self.window.data.copy(deep=True)
        self.window.filter_list.addItem('rt:< 400')
        output_path = os.path.join(self.temporary_directory.name, 'retained.csv')

        with patch(
                'PsySummary.QFileDialog.getSaveFileName',
                return_value=(output_path, 'CSV Files (*.csv)')), \
                patch('PsySummary.PsyDataFunc.genScript'):
            self.window.export_filtered_button.click()

        exported = pd.read_csv(output_path)
        self.assertEqual(exported['rt'].tolist(), [350.5])
        pd.testing.assert_frame_equal(self.window.data, original_data)

        filtered_for_later_run = self.window.getFilteredDataFrame()
        self.assertEqual(filtered_for_later_run['rt'].tolist(), [350.5])
        pd.testing.assert_frame_equal(self.window.data, original_data)

    def test_main_buttons_make_filtering_and_export_actions_explicit(self):
        self.assertEqual(self.window.filter_button.text(), 'Define Filters')
        self.assertEqual(
            self.window.distribution_preview_button.text(), 'Preview Filter Effects')
        self.assertEqual(self.window.export_filtered_button.text(), 'Export Filtered Data')
        self.assertIn(
            'without changing the currently loaded data',
            self.window.export_filtered_button.toolTip())
        self.assertEqual(
            self.window.run_button.text().replace('&&', '&'), 'Apply Filters & Run')
        self.assertIn('original data', self.window.run_button.toolTip())
        self.assertEqual(self.window.save_filter_button.text(), 'Save Setup')
        self.assertIn('Rows, Columns, Data, and Filters', self.window.save_filter_button.toolTip())
        self.assertEqual(self.window.load_filter_button.text(), 'Load Setup')

        with patch(
                'PsySummary.QFileDialog.getSaveFileName',
                return_value=('', 'CSV Files (*.csv)')) as save_dialog:
            self.window.export_filtered_button.click()

        save_dialog.assert_called_once()


if __name__ == '__main__':
    unittest.main()
