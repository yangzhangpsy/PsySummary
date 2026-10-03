"""Read PsySummary data in an owned worker without accessing GUI state."""

import pandas as pd
from PyQt5.QtCore import QThread
from app.dataPreparation import read_psydata


def read_psydata_files(files, check_cancelled=None):
    """Read an ordered batch, restoring saved types or inferring legacy files."""
    frames = []
    for path in files:
        if check_cancelled:
            check_cancelled()
        frames.append(read_psydata(path))
    if check_cancelled:
        check_cancelled()
    if not frames:
        raise ValueError('No data files were selected.')
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True, copy=False)


class ImportPsyDataThread(QThread):
    """Stage a complete import; the GUI publishes it only after native completion."""

    def __init__(self, files, parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.data = None
        self.error = ''
        self.cancelled = False

    def _check_cancelled(self):
        """Cancel between reads without terminating an active parser."""
        if self.isInterruptionRequested():
            raise InterruptedError('Data import cancelled.')

    def run(self):
        """Perform all file reading and concatenation away from the GUI thread."""
        try:
            self.data = read_psydata_files(self.files, self._check_cancelled)
            self._check_cancelled()
        except InterruptedError:
            self.data = None
            self.cancelled = True
        except Exception as error:
            self.data = None
            self.error = str(error)
