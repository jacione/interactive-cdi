# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\cdi_live.py'],
    pathex=['src', 'venv/Lib/site-packages'],
    binaries=[],
    datas=[
        ('example_data/*.tif', 'example_data'),
        ('LICENSE.txt', '.'),
        ('README.md', '.')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='cdi_live',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cdi_live',
)
