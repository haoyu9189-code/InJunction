"""Verify the actual frozen GUI, spawned processing workers, exports and optional imports."""
import json
from pathlib import Path


def run(app, window, output):
    import numpy as np
    from nptdms import TdmsWriter, ChannelObject
    import CaculateHistogram as cacu
    from PySide6.QtCore import QFile
    from modules.ui_functions import UIFunctions

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'synthetic-input'
    source.mkdir(exist_ok=True)
    sweep = np.r_[np.linspace(0, .5, 51), np.linspace(.49, -.5, 100), np.linspace(-.49, 0, 50)]
    for name, plateau, names in [
        ('new', 39, ['Bias (V)', 'Current (mA)', 'Log(G/G0)']),
        ('old', 450, ['RT-IO:AO1-Bias Volt. [V]', 'RT-PV:Current [mA]', 'RT-PV:Conductance LogG']),
    ]:
        v = np.r_[np.full(20, .1), np.full(plateau, .2), np.zeros(5), sweep,
                  np.zeros(5), np.full(plateau, .2), np.full(220, .1)]
        g = np.full(len(v), -4.)
        if name == 'old':
            boundaries = np.flatnonzero(np.diff(np.r_[False, v == .2, False]))
            for start, stop in zip(boundaries[::2], boundaries[1::2]):
                g[start:start + 100] = 0
                g[stop - 100:stop] = 0
        with TdmsWriter(source / f'{name}.tdms') as writer:
            writer.write_segment([ChannelObject('STM', key, data)
                                  for key, data in zip(names, [v, v * 1e-5, g])])

    messages = []
    cacu.signal_window = messages.append
    assert QFile.exists(':/icons/images/icons/icon_close.png'), 'Missing embedded Qt icons'
    assert window.left_grip and window.right_grip
    assert QFile.exists(':/icons/images/icons/cil-layers.png')
    assert 'cil-layers.png' in window.ui.btn_numpy_merge.styleSheet()
    window.resize(1280, 900)
    UIFunctions.maximize_restore(window)
    UIFunctions.maximize_restore(window)
    window.file_path = str(source)
    window.ui.lineEdit_iv.setText(str(source))
    window.ui.doubleSpinBox_4.setValue(.1)
    window.ui.doubleSpinBox_5.setValue(-2.5)
    window.ui.doubleSpinBox_6.setValue(-5.)
    window.ui.stackedWidget.setCurrentWidget(window.ui.IV_page)
    app.processEvents()
    window.run_file_button_iv()
    assert window.his_For is not None and window.his_Reve is not None, messages
    assert np.asarray(window.data_For[0]).shape == (2, 1000)
    assert np.asarray(window.data_Reve[0]).shape == (2, 1000)
    assert window.ui.progressBar_5.value() == 100
    window.Save_iv()
    exports = list(source.rglob('*.npz'))
    assert len(exports) == 8, [str(p) for p in exports]
    for path in exports:
        with np.load(path, allow_pickle=False) as data:
            assert data['distance_array'].shape == (2, 1000)
            assert data['conductance_array'].shape == (2, 1000)
    app.processEvents()
    assert window.grab().save(str(output / 'gui-iv.png'))

    # Open the real menu button and drive the modal dialog and its background worker.
    import shutil
    from PySide6.QtCore import QTimer
    from numpy_merge_dialog import NumpyMergeDialog
    first = exports[0]
    second = output / 'merge-copy.npz'
    shutil.copyfile(first, second)
    merged_path = output / 'merged.npz'
    dialog_errors = []

    def drive_merge_dialog():
        dialog = app.activeModalWidget()
        try:
            assert isinstance(dialog, NumpyMergeDialog)
            dialog.add_paths([str(first), str(second), str(first)])
            assert dialog.files.count() == 2
            dialog.files.setCurrentRow(1)
            dialog.move_current(-1)
            assert dialog.files.item(0).text() == str(second.resolve())
            dialog.begin_merge([str(second), str(first)], str(merged_path))

            def finish():
                try:
                    assert '合并完成' in dialog.status.text(), dialog.status.text()
                    app.processEvents()
                    assert dialog.grab().save(str(output / 'gui-numpy-merge.png'))
                except Exception as exc:
                    dialog_errors.append(str(exc))
                finally:
                    dialog.accept()
            dialog.worker.finished.connect(finish)
        except Exception as exc:
            dialog_errors.append(str(exc))
            if dialog:
                dialog.reject()

    QTimer.singleShot(0, drive_merge_dialog)
    window.ui.btn_numpy_merge.click()
    assert not dialog_errors, dialog_errors
    with np.load(merged_path, allow_pickle=False) as merged, np.load(first) as original:
        for key in ('distance_array', 'conductance_array', 'length_array'):
            np.testing.assert_array_equal(merged[key], np.concatenate([original[key], original[key]]))
        np.testing.assert_array_equal(merged['additional_length'], original['additional_length'])

    # Exercise lazy imports and compiled numerical dependencies inside the frozen EXE.
    import umap
    from tslearn.clustering import TimeSeriesKMeans, KShape
    from numba import njit
    assert njit(lambda x: x + 1)(2) == 3
    rng = np.random.RandomState(42)
    samples = rng.normal(size=(16, 12))
    embedding = umap.UMAP(n_neighbors=4, n_epochs=20, random_state=42).fit_transform(samples)
    assert embedding.shape == (16, 2)
    assert TimeSeriesKMeans(n_clusters=2, n_init=1, max_iter=2, random_state=42).fit_predict(samples).shape == (16,)

    report = {'status': 'passed', 'gui_constructed': True, 'embedded_icons': True,
              'window_grips': True, 'spawn_processing': True, 'old_and_new_synthetic_scans': 2,
              'forward_shape': [2, 1000], 'reverse_shape': [2, 1000],
              'numpy_merge_dialog': True, 'numpy_merge_rows': 4,
              'npz_exports': len(exports), 'numba_jit': True, 'umap': True,
              'tslearn_import': True, 'tslearn_kmeans': True, 'messages': messages}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
