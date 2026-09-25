import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from app.aggregateData import AggregateData
from app.lib.scriptDock import OutputTextEdit


class StandaloneIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_exported_variable_expressions_remain_restricted(self):
        aggregate = AggregateData()
        aggregate.data = pd.DataFrame({"rt": [100.0, 200.0]})

        aggregate.calculateVariable("log_rt", "np.log(self.data['rt'])")
        self.assertEqual(aggregate.data["log_rt"].round(6).tolist(), [4.605170, 5.298317])

        with self.assertRaises(ValueError):
            aggregate.calculateVariable("unsafe", "__import__('os').getcwd()")

    def test_script_export_copies_every_runtime_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            script_path = os.path.join(directory, "analysis.py")
            editor = OutputTextEdit()
            with patch(
                "app.lib.scriptDock.QFileDialog.getSaveFileName",
                return_value=(script_path, "Python Files (*.py)"),
            ):
                editor.export()

            self.assertEqual(
                {
                    "aggregateData.py",
                    "analysis.py",
                    "cognitiveModelSpec.py",
                    "cognitiveModels.py",
                    "expression.py",
                    "rtDist.py",
                },
                set(os.listdir(directory)),
            )


if __name__ == "__main__":
    unittest.main()
