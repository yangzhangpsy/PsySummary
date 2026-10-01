import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/psysummary-test-matplotlib')

from PyQt5.QtWidgets import QApplication

from app.info import Info
from PsySummary import PsyData


class PsySummaryOpenRecentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temporary_directory.name, 'config.ini')
        self.config_patch = patch.object(Info, 'ConfigFile', self.config_path)
        self.config_patch.start()
        self.window = PsyData()

    def tearDown(self):
        self.window.close()
        self.config_patch.stop()
        self.temporary_directory.cleanup()

    def _make_file(self, name):
        file_path = os.path.join(self.temporary_directory.name, name)
        with open(file_path, 'w', encoding='utf-8') as data_file:
            data_file.write('placeholder')
        return file_path

    def test_recent_files_are_ordered_deduplicated_and_openable(self):
        first_file = self._make_file('first.psydata')
        second_file = self._make_file('second.psydata')

        self.window.updateRecentFiles([first_file, second_file])
        self.window.updateRecentFiles([second_file])
        actions = [action for action in self.window.open_recent_menu.actions() if not action.isSeparator()]

        self.assertEqual([action.text() for action in actions[:2]], [second_file, first_file])
        with patch.object(self.window, 'openDataFiles') as open_data_files:
            actions[0].trigger()
            open_data_files.assert_called_once_with([second_file])

    def test_missing_recent_file_is_disabled_and_history_can_be_cleared(self):
        missing_file = os.path.join(self.temporary_directory.name, 'missing.csv')
        self.window.updateRecentFiles([missing_file])
        actions = [action for action in self.window.open_recent_menu.actions() if not action.isSeparator()]

        self.assertEqual(actions[0].text(), missing_file)
        self.assertFalse(actions[0].isEnabled())
        self.assertEqual(actions[-1].text(), 'Clear Items')
        self.assertTrue(actions[-1].isEnabled())

        self.window.clearRecentFiles()
        actions = [action for action in self.window.open_recent_menu.actions() if not action.isSeparator()]
        self.assertEqual(actions[0].text(), 'No Recent Files')
        self.assertFalse(actions[0].isEnabled())
        self.assertFalse(actions[-1].isEnabled())

    def test_delimited_data_load_logs_each_file_through_timed_output(self):
        first_file = self._make_file('first.csv')
        second_file = self._make_file('second.csv')
        self.window.import_file = SimpleNamespace(
            files=[first_file, second_file],
            acceptEvent=MagicMock(),
            getFormatAndDelimiter=lambda: ('utf-8', ','),
            getContainHeadStatus=lambda: True,
        )

        with patch.object(self.window, 'printLogInfo') as print_log, \
                patch.object(self.window, 'clearAllListAndSetData'), \
                patch('PsySummary.PsyDataFunc.genScript'):
            self.window.decodingFileDataReady(pd.DataFrame({'rt': [300]}))

        self.assertEqual(
            print_log.call_args_list,
            [
                call(f'Reading file: {first_file}', 0),
                call(f'Reading file: {second_file}', 0),
            ],
        )


if __name__ == '__main__':
    unittest.main()
