"""Exercise the actual GUI callback with lightweight controls, without Qt/widgets assets."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


class Control:
    def __init__(self, value=1):
        self.number = value
    def value(self): return self.number
    def property(self, _): return self.number
    def isChecked(self): return False
    def currentText(self): return 'jet'
    def setValue(self, value): self.number = value


@pytest.mark.parametrize('failure', ['worker', 'plot'])
def test_gui_callback_handles_failure_and_closes_pool(tmp_path, failure):
    import glob
    import os
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MainWindow')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run_file_button_iv')
    callback = ast.Module(body=[method], type_ignores=[])
    calls, messages = [], []

    class Result:
        def __init__(self, error=False): self.error = error
        def get(self):
            if self.error: raise ValueError('test failure')
            return ([[1, 2, 3]],) * 10

    class Pool:
        def __init__(self, workers):
            assert workers == 1  # Single-core machines must still work.
        def __enter__(self): return self
        def __exit__(self, *args): calls.append('closed')
        def apply_async(self, func, args, kwds):
            calls.append(func)
            return Result(error=failure == 'worker' or func == 'plot')

    scope = dict(os=os, glob=glob, multiprocessing=SimpleNamespace(Pool=Pool, cpu_count=lambda: 1),
                 IVDataProcessUtils=SimpleNamespace(iv_process='worker'), cacu_3fig='plot',
                 cacu=SimpleNamespace(signal_window=messages.append))
    exec(compile(callback, 'main.py', 'exec'), scope)
    controls = {f'doubleSpinBox_{i}': Control() for i in range(4, 15)}
    controls.update({f'spinBox_{i}': Control() for i in [5, 8, 9, 10]})
    controls.update(progressBar_5=Control(), checkBox_2=Control(), color_style_2d=Control())
    (tmp_path / 'data.TDMS').touch()
    window = SimpleNamespace(file_path=str(tmp_path), ui=SimpleNamespace(**controls),
                             showMinimized=lambda: calls.append('minimized'),
                             showNormal=lambda: calls.append('restored'),
                             his_For='old', his_Reve='old', data_For='old', data_Reve='old')
    scope['run_file_button_iv'](window)
    assert calls[-2:] == ['closed', 'restored']
    assert messages and 'failed' in messages[-1]
    assert window.his_For is window.data_For is window.his_Reve is window.data_Reve is None
    assert window.ui.progressBar_5.value() == 0
    if failure == 'worker': assert 'plot' not in calls
