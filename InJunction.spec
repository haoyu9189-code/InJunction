# Reproducible Windows single-file GUI build, invoked from the repository root.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('icon.ico', '.'), ('licenses', 'licenses')]
          + collect_data_files('tslearn', include_py_files=True),
    hiddenimports=['matplotlib.backends.backend_agg', 'matplotlib.backends.backend_ps',
                   'pymsgbox', 'tkinter', 'tkinter.ttk', 'h5py']
                  + collect_submodules('sklearn', filter=lambda name: '.tests' not in name)
                  + collect_submodules('tslearn', filter=lambda name: '.tests' not in name),
    hookspath=[],
    hooksconfig={'matplotlib': {'backends': ['Agg']}},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'tensorflow', 'torch', 'IPython', 'notebook', 'pytest'],
    module_collection_mode={'numba': 'pyz+py', 'umap': 'pyz+py', 'pynndescent': 'pyz+py'},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='InJunction', debug=False,
          bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          disable_windowed_traceback=False, icon=['icon.ico'])
