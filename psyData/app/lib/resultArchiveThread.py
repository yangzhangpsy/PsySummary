"""Owned background jobs for result archives; publish only on native QThread completion."""

import threading

from PyQt5.QtCore import QThread, pyqtSignal
from app.resultArchive import load_result_archive, validate_snapshot
from app.lib.fitProcess import _write_packet
from app.lib.writeProcess import run_write_process


class ResultArchiveThread(QThread):
    """Load snapshots in-thread and coordinate isolated saves without touching widgets."""

    stageChanged = pyqtSignal(str)

    def __init__(self, path, snapshot=None, parent=None):
        super().__init__(parent)
        self.path, self.snapshot = path, snapshot
        self.loading = snapshot is None
        self.result = None
        self.warnings = []
        self.error = ''
        self.succeeded = False
        self.cancelled = False
        self.process_id = None
        self._cancel = threading.Event()

    def requestInterruption(self):
        """Keep a cancellation request until the write commit point or read completion."""
        self._cancel.set()
        super().requestInterruption()

    def isInterruptionRequested(self):
        return self._cancel.is_set()

    def _check(self):
        if self.isInterruptionRequested(): raise InterruptedError('Result archive operation cancelled.')

    def run(self):
        try:
            self._check()
            if self.loading:
                self.result, self.warnings = load_result_archive(self.path, self._check)
            else:
                def prepare(directory, check):
                    validate_snapshot(self.snapshot)
                    _write_packet(directory / 'job.zip', {'kind': 'result'}, check)
                    snapshot = {k: v for k, v in self.snapshot.items() if k != 'fit_records'}
                    _write_packet(directory / 'snapshot.zip', (snapshot,), check,
                                  self.snapshot['fit_records'])
                run_write_process(self, self.path, prepare)
            self.succeeded = True
        except InterruptedError:
            self.cancelled = True
        except Exception as error:
            self.error = str(error)
        finally:
            self.snapshot = None
