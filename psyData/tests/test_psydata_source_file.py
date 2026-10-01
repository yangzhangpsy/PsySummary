import csv
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import numpy as np
from scipy.io import savemat

from app.lib.decoding_files import FinalDataReadWorker
from app.lib.import_mat_thread import ImportMatThread, mapDataFrameElements
from app.lib.source_file import addSourceFileColumn
from PsySummary import readPsyDataFiles
from app.psyDataInfo import PsyDataInfo


class SourceFileColumnTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_csv(self, filename, values):
        path = os.path.join(self.temporary_directory.name, filename)
        pd.DataFrame({'rt': values}).to_csv(path, index=False)
        return path

    def test_dataframe_element_mapping_prefers_current_pandas_api(self):
        """Use DataFrame.map when the current pandas API is available."""
        class CurrentDataFrame:
            def map(self, function):
                return function("current")

        self.assertEqual(
            mapDataFrameElements(CurrentDataFrame(), str.upper),
            "CURRENT",
        )

    def test_dataframe_element_mapping_supports_legacy_pandas_api(self):
        """Fall back to DataFrame.applymap with pandas versions before 2.1."""
        class LegacyDataFrame:
            def applymap(self, function):
                return function("legacy")

        self.assertEqual(
            mapDataFrameElements(LegacyDataFrame(), str.upper),
            "LEGACY",
        )

    def test_source_column_uses_basename_and_preserves_existing_filename(self):
        frames, source_column = addSourceFileColumn(
            [pd.DataFrame({'filename': ['experiment-name'], 'rt': [300]})],
            ['/private/example/participant.csv'],
        )

        self.assertEqual(source_column, 'source_file')
        self.assertEqual(frames[0].loc[0, 'filename'], 'experiment-name')
        self.assertEqual(frames[0].loc[0, 'source_file'], 'participant.csv')

    def test_existing_source_column_is_preserved_without_duplicate(self):
        frames, source_column = addSourceFileColumn(
            [pd.DataFrame({'source_file': ['original.csv'], 'rt': [300]})],
            ['/private/example/imported.psydata'],
        )

        self.assertEqual(source_column, 'source_file')
        self.assertEqual(frames[0].loc[0, 'source_file'], 'original.csv')
        self.assertNotIn('source_file_2', frames[0].columns)

    def test_source_column_is_added_only_to_frames_that_need_it(self):
        frames, source_column = addSourceFileColumn(
            [
                pd.DataFrame({'source_file': ['original.csv'], 'rt': [300]}),
                pd.DataFrame({'rt': [400]}),
            ],
            ['/private/example/first.csv', '/private/example/second.csv'],
        )

        self.assertEqual(source_column, 'source_file')
        self.assertEqual(frames[0].loc[0, 'source_file'], 'original.csv')
        self.assertEqual(frames[1].loc[0, 'source_file'], 'second.csv')

    def test_psysummary_delimited_worker_tags_every_input_file(self):
        first = self._write_csv('participant_1.csv', [300, 350])
        second = self._write_csv('participant_2.csv', [400])
        received = []
        failures = []
        worker = FinalDataReadWorker([first, second], 'utf-8', ',', True, True)
        worker.dataReady.connect(received.append)
        worker.failed.connect(failures.append)

        worker.run()

        self.assertEqual(failures, [])
        self.assertEqual(len(received), 1)
        self.assertEqual(
            received[0]['source_file'].tolist(),
            ['participant_1.csv', 'participant_1.csv', 'participant_2.csv'],
        )

    def test_generic_delimited_worker_does_not_change_other_tools_by_default(self):
        source = self._write_csv('equations.csv', [300])
        received = []
        worker = FinalDataReadWorker([source], 'utf-8', ',', True)
        worker.dataReady.connect(received.append)

        worker.run()

        self.assertNotIn('source_file', received[0].columns)

    def test_psydata_import_does_not_add_a_source_column(self):
        first = os.path.join(self.temporary_directory.name, 'first.psydata')
        second = os.path.join(self.temporary_directory.name, 'second.psydata')
        pd.DataFrame({'rt': [300]}).to_csv(
            first, sep='|', quoting=csv.QUOTE_NONNUMERIC, index=False)
        pd.DataFrame({'rt': [400]}).to_csv(
            second, sep='|', quoting=csv.QUOTE_NONNUMERIC, index=False)
        previous_psydata = PsyDataInfo.PsyData
        PsyDataInfo.PsyData = SimpleNamespace(printLogInfo=lambda *_args: None)
        try:
            with patch('PsySummary.PsyDataFunc.genScript'):
                imported = readPsyDataFiles([first, second])
        finally:
            PsyDataInfo.PsyData = previous_psydata

        self.assertNotIn('source_file', imported.columns)

    def test_psydata_import_preserves_an_existing_source_column(self):
        source = os.path.join(self.temporary_directory.name, 'saved.psydata')
        pd.DataFrame({'source_file': ['original.csv'], 'rt': [300]}).to_csv(
            source, sep='|', quoting=csv.QUOTE_NONNUMERIC, index=False)
        previous_psydata = PsyDataInfo.PsyData
        PsyDataInfo.PsyData = SimpleNamespace(printLogInfo=lambda *_args: None)
        try:
            with patch('PsySummary.PsyDataFunc.genScript'):
                imported = readPsyDataFiles([source])
        finally:
            PsyDataInfo.PsyData = previous_psydata

        self.assertEqual(imported['source_file'].tolist(), ['original.csv'])
        self.assertNotIn('source_file_2', imported.columns)

    def test_mat_import_uses_existing_filename_without_source_file(self):
        paths = []
        for filename, rt_value in [('first.mat', 300.0), ('second.mat', 400.0)]:
            path = os.path.join(self.temporary_directory.name, filename)
            values = np.empty((2, 2), dtype=object)
            values[0, 0] = 'rt'
            values[0, 1] = 'filename'
            values[1, 0] = rt_value
            values[1, 1] = 'experiment-defined-name'
            savemat(path, {'allResults_APL': values})
            paths.append(path)

        imported = ImportMatThread(paths).readMatlabFiles()

        self.assertEqual(
            imported['filename'].tolist(),
            ['experiment-defined-name', 'experiment-defined-name'],
        )
        self.assertNotIn('source_file', imported.columns)

    def test_mat_import_adds_source_file_when_filename_is_missing(self):
        path = os.path.join(self.temporary_directory.name, 'without_filename.mat')
        values = np.empty((2, 1), dtype=object)
        values[0, 0] = 'rt'
        values[1, 0] = 300.0
        savemat(path, {'allResults_APL': values})

        imported = ImportMatThread([path]).readMatlabFiles()

        self.assertEqual(imported['source_file'].tolist(), ['without_filename.mat'])


if __name__ == '__main__':
    unittest.main()
