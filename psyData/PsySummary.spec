# -*- mode: python ; coding: utf-8 -*-
# Windows and Linux use the same one-folder application layout.
from pathlib import Path

root = Path(SPECPATH).resolve()
a = Analysis(
    [str(root / 'PsySummary.py')], pathex=[str(root)], binaries=[],
    datas=[
        (str(root / 'app' / 'images'), 'app/images'),
        (str(root / 'app' / 'demoData'), 'app/demoData'),
        (str(root / 'app' / 'exportFiles'), 'app/exportFiles'),
        # Script export needs source files, not just bundled PYZ bytecode.
        (str(root / 'app' / 'cognitiveModels.py'), 'app'),
        (str(root / 'app' / 'cognitiveModelSpec.py'), 'app'),
    ],
    hiddenimports=['matplotlib.backends.backend_qt5agg'],
    hookspath=[], runtime_hooks=[], excludes=['PyQt6', 'PySide6', 'PySide2'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name='PsySummary',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, icon=str(root / 'app' / 'images' / 'psybuilder.ico'),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='PsySummary')
