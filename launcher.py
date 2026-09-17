"""Frozen Windows entry point. Dispatch spawn workers before importing Qt/NumPy."""
import multiprocessing


def main():
    import argparse
    import ctypes
    import os
    from pathlib import Path
    import sys
    import traceback

    parser = argparse.ArgumentParser(description='InJunction desktop application')
    parser.add_argument('--self-test', type=Path, metavar='OUTPUT', help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.self_test.resolve() if args.self_test else None
    state = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / '.local' / 'share'))) / 'InJunction'
    state.mkdir(parents=True, exist_ok=True)
    # Relative plot caches must be writable even if the EXE lives in Program Files.
    os.chdir(state)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('NUMBA_CACHE_DIR', str(state / 'numba-cache'))
    log = open(state / 'InJunction.log', 'a', encoding='utf-8', buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from PySide6.QtGui import QIcon
        from main import MainWindow
        app = QApplication(sys.argv)
        if sys.platform == 'win32':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('InJunction.IVCompatibility')
        window = MainWindow(show_startup_message=not bool(output))
        window.setWindowTitle('InJunction — IV compatibility 2026.09')
        icon = Path(__file__).resolve().parent / 'icon.ico'
        window.setWindowIcon(QIcon(str(icon)))
        if output:
            from build_support.frozen_smoke import run
            run(app, window, output)
            window.close()
            app.processEvents()
            return 0

        def handle_exception(kind, value, tb):
            traceback.print_exception(kind, value, tb, file=log)
            log.flush()
            QMessageBox.critical(window, 'InJunction', f'{value}\n\nLog: {state / "InJunction.log"}')
        sys.excepthook = handle_exception
        window.show()
        return app.exec()
    except Exception:
        traceback.print_exc(file=log)
        log.flush()
        if output:
            output.mkdir(parents=True, exist_ok=True)
            (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        elif sys.platform == 'win32':
            ctypes.windll.user32.MessageBoxW(None, f'启动失败，请查看日志：\n{state / "InJunction.log"}',
                                           'InJunction', 0x10)
        return 1


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
