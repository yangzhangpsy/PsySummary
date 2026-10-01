import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pandas as pd
from PyQt5.QtWidgets import QApplication

from app.aggregateData import AggregateData
from app.lib.scriptDock import OutputTextEdit
from app.variableCompute import VariableCompute, evaluateVariableExpression


class StandaloneSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_variable_transform_retains_validation_and_reproducible_script(self):
        widget = VariableCompute(pd.DataFrame({'rt': [300.0, 400.0]}))
        widget.target_input.setText('seconds')
        widget.numeric_expression.setText("self.dataFrame['rt'] / 1000")
        with patch('app.variableCompute.PsyDataFunc.genScript') as generate_script:
            widget.on_ok_button_click()
        self.assertEqual(widget.dataFrame['seconds'].tolist(), [0.3, 0.4])
        script = generate_script.call_args.args[0]
        aggregate = AggregateData()
        aggregate.data = pd.DataFrame({'rt': [300.0, 400.0]})
        exec(script, {'aggData': aggregate})
        pd.testing.assert_frame_equal(aggregate.data, widget.dataFrame)
        with self.assertRaises(ValueError):
            evaluateVariableExpression("__import__('os')", widget)
        widget.close()

    def test_aggregate_transform_retains_expression_validation(self):
        aggregate = AggregateData()
        aggregate.data = pd.DataFrame({'rt': [300.0, 400.0]})
        aggregate.calculateVariable('seconds', "self.data['rt'] / 1000")
        self.assertEqual(aggregate.data['seconds'].tolist(), [0.3, 0.4])
        with self.assertRaises(ValueError):
            aggregate.calculateVariable('invalid', "__import__('os')")

    def test_exported_analysis_runs_without_application_package(self):
        with tempfile.TemporaryDirectory() as directory:
            script_path = os.path.join(directory, 'analysis.py')
            editor = OutputTextEdit()
            editor.setPlainText(
                'import pandas as pd\n'
                'from aggregateData import AggregateData\n'
                'aggData = AggregateData()\n'
                "aggData.data = pd.DataFrame({'rt': [300.0, 400.0]})\n"
                'aggData.calculateVariable("seconds", "self.data[\'rt\'] / 1000")\n'
                'assert aggData.data["seconds"].tolist() == [0.3, 0.4]\n'
            )
            with patch('app.lib.scriptDock.QFileDialog.getSaveFileName',
                       return_value=(script_path, 'Python Files (*.py)')):
                editor.export()
            editor.close()
            self.assertFalse(os.path.exists(os.path.join(directory, 'fitCancellation.py')))
            result = subprocess.run(
                [sys.executable, '-I', '-c',
                 'import importlib.abc, runpy, sys\n'
                 'class BlockApp(importlib.abc.MetaPathFinder):\n'
                 '    def find_spec(self, fullname, path=None, target=None):\n'
                 "        if fullname == 'app' or fullname.startswith('app.'):\n"
                 "            raise ModuleNotFoundError('blocked app package')\n"
                 'sys.meta_path.insert(0, BlockApp())\n'
                 'sys.path.insert(0, sys.argv[1])\n'
                 'runpy.run_path(sys.argv[2], run_name="__main__")\n',
                 directory, script_path],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
