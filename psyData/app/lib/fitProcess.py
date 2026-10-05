"""Owned fit subprocesses; the Qt thread only stages data, relays events and reaps children.

The private job directory contains typed JSON/NPY packets, never executable pickle.
It is not a user-facing result format and is removed after the child has exited.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

from PyQt5.QtCore import QCoreApplication, QThread, Qt, pyqtSignal, pyqtSlot

from app.fitCancellation import FitCancelled, raise_if_fit_cancelled


def worker_command(directory, flag='--psysummary-fit-worker'):
    """Use the same Python or frozen application with an early, non-GUI entry point."""
    args = [flag, str(directory)]
    if getattr(sys, 'frozen', False):
        return [sys.executable, *args]
    return [sys.executable, str(Path(__file__).resolve().parents[2] / 'PsySummary.py'), *args]


def worker_environment():
    """Avoid nested numerical thread oversubscription and select a non-GUI plot backend."""
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    if getattr(sys, 'frozen', False):
        env['PYINSTALLER_RESET_ENVIRONMENT'] = '1'
    return env


def _write_packet(path, data, check, records=None):
    """Encode data and optional curves incrementally in an owned, non-pickle packet."""
    from app.resultArchive import _Codec
    from app.diagnosticCurveCache import cache_key

    temporary = path.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED) as archive:
        codec = _Codec(archive, check)
        manifest = {'version': 1, 'data': codec.encode(data)}
        if records is not None:
            encoded, curves = [], []
            for record in records:
                check()
                encoded.append(codec.encode({k: v for k, v in record.items()
                                             if not k.startswith('_diagnostic_')}))
                views = {}
                cache = record.get('_diagnostic_curve_cache')
                for mode in (False, True):
                    payload = cache.get(cache_key(record, mode)) if cache else None
                    if payload is not None and not payload.get('failed'):
                        views[str(int(mode))] = codec.encode(payload)
                curves.append(views)
            manifest.update(records=encoded, curves=curves)
        archive.writestr('manifest.json', json.dumps(manifest, allow_nan=False))
    check()
    os.replace(temporary, path)


def _read_packet(path, check, cache=None):
    """Decode typed input or reattach prepared curves to the parent's bounded result cache."""
    from app.resultArchive import _Codec, MAX_MANIFEST_BYTES, MAX_EXPANDED_BYTES
    from app.diagnosticCurveCache import attach_cache, cache_key

    with zipfile.ZipFile(path) as archive:
        if (archive.getinfo('manifest.json').file_size > MAX_MANIFEST_BYTES
                or sum(info.file_size for info in archive.infolist()) > MAX_EXPANDED_BYTES):
            raise ValueError('Fit process data exceeds the supported transfer size.')
        check()
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('version') != 1:
            raise ValueError('Unsupported fit process packet.')
        codec = _Codec(archive, check)
        data = codec.decode(manifest['data'])
        if cache is None:
            return data
        records = []
        for encoded, views in zip(manifest['records'], manifest['curves'], strict=True):
            check()
            record = codec.decode(encoded)
            attach_cache(record, cache)
            for mode in (False, True):
                if str(int(mode)) in views:
                    payload = codec.decode(views[str(int(mode))])
                    try:
                        cache.put(cache_key(record, mode), payload,
                                  lambda: False)  # check() around each bounded view transfer.
                    except OSError:
                        # put() remembers in RAM first; fitting remains valid if disk caching fails.
                        pass
                    check()
            records.append(record)
        return (*data, records)


class IsolatedFitThread(QThread):
    """Coordinate one subprocess without doing likelihood/curve work in the GUI process."""

    fitStatus = pyqtSignal(int, str, bool)
    resultReady = pyqtSignal(object, list, list, list, object)
    cancelled = pyqtSignal()
    conditionProgress = pyqtSignal(int, int)
    curvePreparationProgress = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel = threading.Event()
        self._outcome = None
        self.external_cancel_check = None
        self.process_id = None
        self.finished.connect(self._publish_result)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown)

    def requestInterruption(self):
        """Retain cancellation through native completion and result publication."""
        self._cancel.set()
        super().requestInterruption()

    def isInterruptionRequested(self):
        return self._cancel.is_set() or bool(self.external_cancel_check and self.external_cancel_check())

    def _check(self):
        raise_if_fit_cancelled(self.isInterruptionRequested)

    def run(self):
        """Stage only required columns, monitor progress, and reap before native finished."""
        try:
            self._outcome = ('result', self._run_process())
        except (FitCancelled, InterruptedError):
            self._outcome = ('cancelled', None)
        except Exception as error:
            self._outcome = ('error', str(error))

    def _run_process(self):
        self._check()
        job = self._process_job()
        with tempfile.TemporaryDirectory(prefix='psysummary-fit-') as name:
            directory = Path(name)
            _write_packet(directory / 'input.zip', job, self._check)
            del job
            self._check()
            (directory / 'heartbeat').touch()
            (directory / 'events.jsonl').touch()
            process = None
            with (directory / 'worker.log').open('wb') as log, \
                    (directory / 'events.jsonl').open('r', encoding='utf-8') as events:
                try:
                    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
                    process = subprocess.Popen(worker_command(directory), env=worker_environment(),
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=log, **options)
                    self.process_id = process.pid
                    cancel_time = None
                    heartbeat_time = 0
                    pending = ''
                    while True:
                        exit_code = process.poll()
                        now = time.monotonic()
                        if now - heartbeat_time >= 1:
                            (directory / 'heartbeat').touch()
                            heartbeat_time = now
                        if self.isInterruptionRequested() and cancel_time is None:
                            (directory / 'cancel').touch()
                            cancel_time = now
                        if cancel_time is not None and now - cancel_time > 5 and process.poll() is None:
                            # Only the owned process is stopped; never terminate a QThread.
                            process.terminate()
                            try:
                                process.wait(timeout=1)
                            except subprocess.TimeoutExpired:
                                process.kill()
                        chunk = events.read(65536)
                        pending += chunk
                        while '\n' in pending:
                            line, pending = pending.split('\n', 1)
                            event, args = json.loads(line)
                            if not self.isInterruptionRequested():
                                if event == 'status': self.fitStatus.emit(*args)
                                elif event == 'condition': self.conditionProgress.emit(*args)
                                elif event == 'curves': self.curvePreparationProgress.emit(*args)
                        if exit_code is not None and len(chunk) < 65536:
                            break
                        time.sleep(.05)  # Coordinator only, never the GUI event loop.
                    self._check()
                    if process.returncode != 0 or not (directory / 'output.zip').is_file():
                        log.flush()
                        with (directory / 'worker.log').open('rb') as error_log:
                            error_log.seek(max(0, error_log.seek(0, 2) - 8192))
                            detail = error_log.read().decode('utf-8', errors='replace').strip()
                        raise RuntimeError(f'Fit process exited with code {process.returncode}. {detail}')
                    result = _read_packet(directory / 'output.zip', self._check, self.curve_cache)
                    frame, names, rows, columns, series_name, records = result
                    self._check()
                    return (frame.iloc[:, 0].rename(series_name), names, rows, columns, records)
                finally:
                    if process is not None:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=1)
                            except subprocess.TimeoutExpired:
                                process.kill()
                        process.wait()

    @pyqtSlot()
    def _publish_result(self):
        """Deliver terminal signals on the GUI thread only after native completion/reaping."""
        outcome, payload = self._outcome or ('error', 'Fit coordinator did not return a result.')
        self._outcome = None
        if self.isInterruptionRequested() or outcome == 'cancelled':
            self.cancelled.emit()
        elif outcome == 'error':
            self.fitStatus.emit(2, f'Model fitting error: {payload}', False)
            self.resultReady.emit(None, [], self.row_vars, self.col_vars, [])
        else:
            self.resultReady.emit(*payload)

    @pyqtSlot()
    def _shutdown(self):
        """Join only at final application shutdown, after requesting child cancellation."""
        if self.isRunning():
            self.requestInterruption()
            self.wait()


def worker_main(name):
    """Run the existing grouped numerical implementation without a GUI application."""
    directory = Path(name).resolve()
    cancellation = threading.Event()
    finished = threading.Event()

    def watch_owner():
        """Cancel on request and stop an orphan if its coordinator heartbeat disappears."""
        while not finished.wait(.1):
            try:
                lost = time.time() - (directory / 'heartbeat').stat().st_mtime > 15
            except OSError:
                lost = True
            if (directory / 'cancel').exists() or lost:
                cancellation.set()
            if lost and not finished.wait(5):
                os._exit(3)

    threading.Thread(target=watch_owner, daemon=True).start()
    cache = None
    try:
        check = lambda: raise_if_fit_cancelled(cancellation.is_set)
        job = _read_packet(directory / 'input.zip', check)
        from app.diagnosticCurveCache import DiagnosticCurveCache
        from app.lib.fitCognitiveModelThread import FitCognitiveModelThread
        from app.lib.fitRTsDistThread import FitRTsDistThread
        cache = DiagnosticCurveCache(directory_parent=directory)
        worker_type = {'cognitive': FitCognitiveModelThread, 'distribution': FitRTsDistThread}[job['kind']]
        worker = worker_type(job['frame'], **job['arguments'], curve_cache=cache)
        worker.external_cancel_check = cancellation.is_set
        outcomes = []
        with (directory / 'events.jsonl').open('a', encoding='utf-8', buffering=1) as events:
            def emit(event, *args):
                events.write(json.dumps([event, args], ensure_ascii=True) + '\n')
            worker.fitStatus.connect(lambda *args: emit('status', *args), Qt.DirectConnection)
            worker.conditionProgress.connect(lambda *args: emit('condition', *args), Qt.DirectConnection)
            worker.curvePreparationProgress.connect(lambda *args: emit('curves', *args), Qt.DirectConnection)
            worker.resultReady.connect(lambda *args: outcomes.append(args), Qt.DirectConnection)
            worker.cancelled.connect(cancellation.set, Qt.DirectConnection)
            worker._run_local()
            check()
            if not outcomes or outcomes[0][0] is None:
                raise RuntimeError('Model fitting failed; see the preceding fit diagnostics.')
            result, names, rows, columns, records = outcomes[0]
            _write_packet(directory / 'output.zip', (result.to_frame(), names, rows, columns, result.name), check, records)
        return 0
    except (FitCancelled, InterruptedError):
        return 2
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    finally:
        finished.set()
        if cache is not None:
            cache.close()
