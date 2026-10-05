"""Isolate export/archive writing while keeping atomic publication owned by the coordinator."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from app.lib.fitProcess import (
    _read_packet, _write_packet, worker_command, worker_environment,
)


def _validate_destination(path):
    """Never follow a destination symlink or replace a directory."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError('Cannot replace a directory or symbolic link with exported data.')


def stage_frame(directory, frame, check):
    """Spool bounded column chunks without deep-copying the parent source table."""
    import pandas as pd

    chunk_rows = 65536
    chunks = max(1, (len(frame) + chunk_rows - 1) // chunk_rows)
    index = (('range', frame.index.start, frame.index.stop, frame.index.step, frame.index.name)
             if isinstance(frame.index, pd.RangeIndex) else ('index', frame.index))
    _write_packet(directory / 'axes.zip', (index, frame.columns, chunks), check)
    for column in range(len(frame.columns)):
        for chunk in range(chunks):
            check()
            start = chunk * chunk_rows
            part = frame.iloc[start:start + chunk_rows, column]
            # Series metadata is supplied separately; preserve its dtype without its repeated index.
            _write_packet(directory / f'column-{column}-{chunk}.zip', part, check)
            # Yield between staging chunks, never busy-poll from the GUI thread.
            time.sleep(0)


def restore_frame(directory, check):
    """Reconstruct one child-owned table, with at most one column's assembly overhead."""
    import pandas as pd

    encoded_index, columns, chunks = _read_packet(directory / 'axes.zip', check)
    index = (pd.RangeIndex(*encoded_index[1:4], name=encoded_index[4])
             if encoded_index[0] == 'range' else encoded_index[1])
    frame = pd.DataFrame(index=index)
    for column in range(len(columns)):
        parts = []
        for chunk in range(chunks):
            check()
            parts.append(_read_packet(directory / f'column-{column}-{chunk}.zip', check))
        series = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        del parts
        if len(series) != len(index):
            raise ValueError('Incomplete export column transfer.')
        # Use positional arrays, never align a source with duplicated index labels.
        frame.insert(column, column, series.array)
        del series
    frame.columns = columns
    return frame


def run_write_process(owner, destination, prepare):
    """Stage inputs off-GUI, reap the writer, then atomically publish on the same filesystem."""
    destination = Path(destination).absolute()
    check = owner._check_cancelled if hasattr(owner, '_check_cancelled') else owner._check
    check()
    _validate_destination(destination)
    with tempfile.TemporaryDirectory(prefix='.psysummary-write-', dir=destination.parent) as name:
        directory = Path(name)
        owner.stageChanged.emit('Preparing data for writing…')
        prepare(directory, check)
        check()
        (directory / 'heartbeat').touch()
        (directory / 'events.jsonl').touch()
        process = None
        with (directory / 'worker.log').open('wb') as log, \
                (directory / 'events.jsonl').open('r', encoding='utf-8') as events:
            try:
                options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
                process = subprocess.Popen(
                    worker_command(directory, '--psysummary-write-worker'), env=worker_environment(),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, **options)
                owner.process_id = process.pid
                cancel_time = None
                heartbeat_time = 0
                pending = ''
                while True:
                    exit_code = process.poll()
                    now = time.monotonic()
                    if now - heartbeat_time >= 1:
                        (directory / 'heartbeat').touch()
                        heartbeat_time = now
                    if owner.isInterruptionRequested() and cancel_time is None:
                        (directory / 'cancel').touch()
                        cancel_time = now
                    if cancel_time is not None and now - cancel_time > 5 and process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    chunk = events.read(65536)
                    pending += chunk
                    while '\n' in pending:
                        line, pending = pending.split('\n', 1)
                        if not owner.isInterruptionRequested():
                            owner.stageChanged.emit(json.loads(line))
                    if exit_code is not None and len(chunk) < 65536:
                        break
                    time.sleep(.05)
                check()
                if exit_code != 0 or not (directory / 'ready').is_file():
                    log.flush()
                    with (directory / 'worker.log').open('rb') as error_log:
                        error_log.seek(max(0, error_log.seek(0, 2) - 8192))
                        detail = error_log.read().decode('utf-8', errors='replace').strip()
                    raise RuntimeError(f'Writing process exited with code {exit_code}. {detail}')
                _validate_destination(destination)
                check()
                # This is the commit point. A later cancel must not misreport a published file.
                os.replace(directory / 'output.tmp', destination)
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    process.wait()


def worker_main(name):
    """Write only an owned temporary output; the child never receives the destination path."""
    directory = Path(name).resolve()
    cancellation, finished = threading.Event(), threading.Event()

    def watch_owner():
        while not finished.wait(.1):
            try:
                lost = time.time() - (directory / 'heartbeat').stat().st_mtime > 15
            except OSError:
                lost = True
            if lost or (directory / 'cancel').exists():
                cancellation.set()
            if lost and not finished.wait(5):
                os._exit(3)

    def check():
        if cancellation.is_set():
            raise InterruptedError('Writing cancelled.')

    threading.Thread(target=watch_owner, daemon=True).start()
    cache = None
    try:
        job = _read_packet(directory / 'job.zip', check)
        with (directory / 'events.jsonl').open('a', encoding='utf-8', buffering=1) as events:
            def stage(message):
                check()
                events.write(json.dumps(message) + '\n')

            stage('Receiving data…')
            if job['kind'] == 'data':
                from app.lib.dataExportThread import _InterruptibleWriter
                from app.dataPreparation import write_psydata_stream
                frame = restore_frame(directory, check)
                stage('Writing data…')
                with (directory / 'output.tmp').open('w', encoding='utf-8', newline='') as stream:
                    writer = _InterruptibleWriter(stream, check)
                    rows = max(1, min(65536, 100000 // max(1, len(frame.columns))))
                    if job['psydata']:
                        write_psydata_stream(frame, writer, rows, check)
                    else:
                        frame.to_csv(writer, chunksize=rows, **job['csv_options'])
            elif job['kind'] == 'result':
                from app.diagnosticCurveCache import DiagnosticCurveCache
                from app.resultArchive import save_result_archive
                cache = DiagnosticCurveCache(directory_parent=directory)
                snapshot, records = _read_packet(directory / 'snapshot.zip', check, cache)
                snapshot['fit_records'] = records
                stage('Saving results…')
                save_result_archive(directory / 'output.tmp', snapshot, check)
            else:
                raise ValueError('Unknown writing operation.')
            check()
            (directory / 'ready').touch()
        return 0
    except InterruptedError:
        return 2
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    finally:
        finished.set()
        if cache is not None:
            cache.close()
