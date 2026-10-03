# -*- mode: python ; coding: utf-8 -*-
import os
from importlib.util import find_spec
from pathlib import Path

root = Path(SPECPATH).resolve()
codesign_identity = os.environ.get('PSYSUMMARY_CODESIGN_IDENTITY', '').strip() or None
entitlements_file = str(root / 'entitlements.plist') if codesign_identity else None
if entitlements_file and not Path(entitlements_file).is_file():
    raise FileNotFoundError(f'Entitlements file not found: {entitlements_file}')
# Some SciPy versions no longer provide the extension requested by older hooks.
scipy_excludes = [] if find_spec('scipy.special._cdflib') else ['scipy.special._cdflib']
a = Analysis(
    [str(root / 'PsySummary.py')], pathex=[str(root)], binaries=[],
    datas=[
        (str(root / 'app' / 'images'), 'app/images'),
        (str(root / 'app' / 'demoData'), 'app/demoData'),
        (str(root / 'app' / 'exportFiles'), 'app/exportFiles'),
        # Script export needs source files, not just bundled PYZ bytecode.
        (str(root / 'app' / 'cognitiveModels.py'), 'app'),
        (str(root / 'app' / 'cognitiveModelSpec.py'), 'app'),
        (str(root / 'app' / 'expression.py'), 'app'),
        (str(root / 'app' / 'dataPreparation.py'), 'app'),
    ],
    hiddenimports=['matplotlib.backends.backend_qt5agg'],
    hookspath=[], runtime_hooks=[],
    excludes=['PyQt6', 'PySide6', 'PySide2'] + scipy_excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name='PsySummary',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, icon=str(root / 'app' / 'images' / 'icon.icns'),
    codesign_identity=codesign_identity, entitlements_file=entitlements_file,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='PsySummary')
app = BUNDLE(
    coll, name='PsySummary.app', icon=str(root / 'app' / 'images' / 'icon.icns'),
    bundle_identifier='org.yangzhangpsy.PsySummary',
    info_plist={
        'CFBundleName': 'PsySummary',
        'CFBundleDisplayName': 'PsySummary',
        'CFBundleVersion': '0.1',
        'CFBundleShortVersionString': '0.1',
        'NSRequiresAquaSystemAppearance': True,
        'NSHighResolutionCapable': True,
    },
)
