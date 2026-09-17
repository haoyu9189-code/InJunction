"""Modal NumPy merge dialog; disk and array work runs off the GUI thread."""
from pathlib import Path
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
                               QPushButton, QFileDialog, QAbstractItemView, QComboBox)
from numpy_merge import merge_numpy_files


class MergeWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, paths, output, mode, parent=None):
        super().__init__(parent)
        self.paths, self.output, self.mode = paths, output, mode

    def run(self):
        try:
            self.completed.emit(merge_numpy_files(self.paths, self.output, self.mode))
        except Exception as exc:
            self.failed.emit(str(exc))


class NumpyMergeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('合并 NumPy')
        self.resize(760, 500)
        self.worker = None
        layout = QVBoxLayout(self)
        hint = QLabel('添加同格式 .npz 或 .npy 文件，按下方顺序沿第一维合并。\n'
                      'NPZ 字段、数据类型和每条曲线点数须一致；其他标量参数须相同。\n'
                      '请仅选择相同物理量、单位和扫描方向的数据。程序不会自动转换或去重曲线。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        options = QHBoxLayout()
        options.addWidget(QLabel('additional_length 含义：'))
        self.additional_mode = QComboBox()
        for label, mode in [('自动识别', 'auto'), ('曲线数量（相加）', 'count'),
                            ('每条曲线点数（保留）', 'points'), ('固定参数（须相同）', 'constant')]:
            self.additional_mode.addItem(label, mode)
        options.addWidget(self.additional_mode)
        options.addStretch()
        layout.addLayout(options)
        self.files = QListWidget()
        self.files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.files)
        row = QHBoxLayout()
        self.add_button = QPushButton('添加文件…')
        self.remove_button = QPushButton('移除选中')
        self.up_button = QPushButton('上移')
        self.down_button = QPushButton('下移')
        self.merge_button = QPushButton('合并并另存为…')
        self.close_button = QPushButton('关闭')
        self.controls = [self.additional_mode, self.add_button, self.remove_button, self.up_button,
                         self.down_button, self.merge_button, self.close_button]
        for button in self.controls[1:]:
            row.addWidget(button)
        layout.addLayout(row)
        self.status = QLabel('请选择至少两个文件。')
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(76)
        self.status.setTextFormat(Qt.PlainText)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.add_button.clicked.connect(self.choose_files)
        self.remove_button.clicked.connect(self.remove_selected)
        self.up_button.clicked.connect(lambda: self.move_current(-1))
        self.down_button.clicked.connect(lambda: self.move_current(1))
        self.merge_button.clicked.connect(self.start_merge)
        self.close_button.clicked.connect(self.reject)

    def add_paths(self, paths):
        existing = {self.files.item(i).text() for i in range(self.files.count())}
        for path in paths:
            path = str(Path(path).resolve())
            if path not in existing:
                self.files.addItem(path)
                existing.add(path)
        self.status.setText(f'已选择 {self.files.count()} 个文件。')

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, '选择同格式 NumPy 文件', '',
                                               'NumPy 文件 (*.npz *.npy)')
        self.add_paths(paths)

    def remove_selected(self):
        for item in self.files.selectedItems():
            self.files.takeItem(self.files.row(item))
        self.status.setText(f'已选择 {self.files.count()} 个文件。')

    def move_current(self, offset):
        row = self.files.currentRow()
        if 0 <= row + offset < self.files.count() and row >= 0:
            self.files.insertItem(row + offset, self.files.takeItem(row))
            self.files.setCurrentRow(row + offset)

    def start_merge(self):
        paths = [self.files.item(i).text() for i in range(self.files.count())]
        if len(paths) < 2:
            self.status.setText('请至少选择两个文件。')
            return
        suffix = Path(paths[0]).suffix.lower()
        if suffix not in ('.npy', '.npz') or any(Path(p).suffix.lower() != suffix for p in paths):
            self.status.setText('请分别合并 .npy 和 .npz，不能混用。')
            return
        output, _ = QFileDialog.getSaveFileName(self, '保存合并数据',
                                               str(Path(paths[0]).with_name('merged' + suffix)),
                                               f'NumPy 文件 (*{suffix})')
        if not output:
            return
        # QFileDialog handles extension and overwrite confirmation before this point.
        if not Path(output).suffix:
            self.status.setText(f'请在保存文件名后加上 {suffix} 扩展名。')
            return
        self.begin_merge(paths, output)

    def begin_merge(self, paths, output):
        if self.worker is not None and self.worker.isRunning():
            return
        for button in self.controls:
            button.setEnabled(False)
        self.files.setEnabled(False)
        self.status.setText('正在检查格式并合并，请稍候…')
        self.worker = MergeWorker(paths, output, self.additional_mode.currentData(), self)
        self.worker.completed.connect(self.show_result)
        self.worker.failed.connect(lambda error: self.status.setText('未合并：' + error))
        self.worker.finished.connect(self.finish_merge)
        self.worker.start()

    def show_result(self, result):
        self.status.setText(f"合并完成：{result['files']} 个文件，共 {result['rows']} 条/行。\n"
                            f"已保存：{result['output']}")

    def finish_merge(self):
        for button in self.controls:
            button.setEnabled(True)
        self.files.setEnabled(True)

    def reject(self):
        if self.worker is None or not self.worker.isRunning():
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
        else:
            super().closeEvent(event)
