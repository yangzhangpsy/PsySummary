# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['PsySummary.py'],
    pathex=['C:\\Users\\Yang\\PycharmProjects\\psyData'],
    binaries=[],
    datas=[('.\\app\\exportFiles', '.\\app\\exportFiles'), ('.\\app\\images', '.\\app\\images'), ('.\\app\\demoData', '.\\app\\demoData')],
    hiddenimports=['pkg_resources.py2_warn', 'pkg_resources.extern'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PsySummary',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['.\\app\\images\\psybuilder.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PsySummary',
)
