"""Background coordination of isolated, atomic CSV/psydata writing."""

import os
import threading
from PyQt5.QtCore import QThread, pyqtSignal
from app.lib.fitProcess import _write_packet
from app.lib.writeProcess import run_write_process, stage_frame


class ExportCancelled(Exception):
    """An interrupted export must leave the existing destination unchanged."""


class _InterruptibleWriter:
    """Check cooperative cancellation as pandas writes its bounded CSV chunks."""

    def __init__(self, stream, check_cancelled):
        self.stream = stream
        self.check_cancelled = check_cancelled

    def write(self, text):
        self.check_cancelled()
        return self.stream.write(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


class DataExportThread(QThread):
    """Retain a read-only frame reference and publish only a complete export."""

    stageChanged = pyqtSignal(str)

    def __init__(self, dataframe, file_path, csv_options, script_lines, parent=None):
        super().__init__(parent)
        self.dataframe = dataframe
        self.file_path = os.path.abspath(file_path)
        self.csv_options = dict(csv_options)
        self.script_lines = list(script_lines)
        self.row_count = len(dataframe)
        self.error = ''
        self.cancelled = False
        self.succeeded = False
        self.process_id = None
        self._cancel = threading.Event()

    def requestInterruption(self):
        """Retain cancellation even if requested before start or during native completion."""
        self._cancel.set()
        super().requestInterruption()

    def isInterruptionRequested(self):
        return self._cancel.is_set()

    def _check_cancelled(self):
        """Stop at a safe write boundary instead of terminating the native thread."""
        if self.isInterruptionRequested():
            raise ExportCancelled()

    def run(self):
        """Coordinate the isolated writer; publish only after it exits successfully."""
        try:
            def prepare(directory, check):
                _write_packet(directory / 'job.zip', {
                    'kind': 'data', 'psydata': os.path.splitext(self.file_path)[1].lower() == '.psydata',
                    'csv_options': self.csv_options}, check)
                stage_frame(directory, self.dataframe, check)
            run_write_process(self, self.file_path, prepare)
            self.succeeded = True
        except (ExportCancelled, InterruptedError):
            self.cancelled = True
        except Exception as error:
            self.error = str(error)
        finally:
            self.dataframe = None
