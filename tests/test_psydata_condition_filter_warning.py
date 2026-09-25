import os
import unittest
from unittest.mock import patch

import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt5.QtWidgets import QApplication

from app.lib.pivotedDataWidget import PivotedDataWidget
from app.tool import StatisticTool


class ConditionWiseFilterWarningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_warning_is_logged_after_run_for_multiple_combinations(self):
        data = pd.DataFrame({
            'participant': ['P1', 'P1', 'P2', 'P2'],
            'condition': ['A', 'B', 'A', 'B'],
            'rt': [300.0, 350.0, 400.0, 450.0],
        })
        with patch('app.tool.PsyDataFunc.printOut') as print_out, \
                patch('app.tool.PsyDataFunc.genScript'):
            result_window = PivotedDataWidget(
                data, ['participant'], ['condition'], ['rt@Mean'], ['rt:< 500'])

        warning, information_type = print_out.call_args.args[:2]
        self.assertEqual(information_type, 4)
        self.assertIn('4 observed data cells', warning)
        self.assertIn('participant × condition', warning)
        self.assertIn('exaggerate between-condition differences', warning)
        self.assertIn('https://doi.org/10.1037/xge0001069', warning)
        result_window.close()

    def test_warning_is_not_logged_after_run_for_one_observed_combination(self):
        data = pd.DataFrame({
            'participant': ['P1', 'P1'],
            'condition': ['A', 'A'],
            'rt': [300.0, 350.0],
        })
        with patch('app.tool.PsyDataFunc.printOut') as print_out, \
                patch('app.tool.PsyDataFunc.genScript'):
            result_window = PivotedDataWidget(
                data, ['participant'], ['condition'], ['rt@Mean'], ['rt:< 500'])

        print_out.assert_not_called()
        result_window.close()

    def test_warning_is_not_logged_after_run_without_active_filters(self):
        data = pd.DataFrame({
            'condition': ['A', 'B'],
            'rt': [300.0, 350.0],
        })
        with patch('app.tool.PsyDataFunc.printOut') as print_out, \
                patch('app.tool.PsyDataFunc.genScript'):
            result_window = PivotedDataWidget(
                data, [], ['condition'], ['rt@Mean'], [])

        print_out.assert_not_called()
        result_window.close()

    def test_filtering_outside_run_does_not_log_the_warning(self):
        data = pd.DataFrame({
            'condition': ['A', 'B'],
            'rt': [300.0, 350.0],
        })
        with patch('app.tool.PsyDataFunc.printOut') as print_out, \
                patch('app.tool.PsyDataFunc.genScript'):
            StatisticTool.filterData([], ['condition'], data, ['rt:< 500'])

        print_out.assert_not_called()


if __name__ == '__main__':
    unittest.main()
