"""Background, atomic CSV/psydata writing without accessing GUI objects."""

import os
import tempfile

from PyQt5.QtCore import QThread
from app.dataPreparation import write_psydata_stream


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

    def _check_cancelled(self):
        """Stop at a safe write boundary instead of terminating the native thread."""
        if self.isInterruptionRequested():
            raise ExportCancelled()

    def run(self):
        """Write beside the destination, close the stream, and atomically replace it."""
        temporary_path = None
        try:
            self._check_cancelled()
            if os.path.lexists(self.file_path) and (
                    os.path.islink(self.file_path) or not os.path.isfile(self.file_path)):
                raise ValueError('Cannot replace a directory or symbolic link with exported data.')
            descriptor, temporary_path = tempfile.mkstemp(
                prefix='.psysummary-data-', suffix='.tmp',
                dir=os.path.dirname(self.file_path))
            with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:
                writer = _InterruptibleWriter(stream, self._check_cancelled)
                chunk_rows = max(1, min(65536, 100000 // max(1, len(self.dataframe.columns))))
                if os.path.splitext(self.file_path)[1].lower() == '.psydata':
                    write_psydata_stream(self.dataframe, writer, chunk_rows, self._check_cancelled)
                else:
                    self.dataframe.to_csv(writer, chunksize=chunk_rows, **self.csv_options)
            self._check_cancelled()
            os.replace(temporary_path, self.file_path)
            temporary_path = None
            self.succeeded = True
        except ExportCancelled:
            self.cancelled = True
        except Exception as error:
            self.error = str(error)
        finally:
            self.dataframe = None
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError as error:
                    self.error += f' Temporary file could not be removed: {temporary_path} ({error})'
