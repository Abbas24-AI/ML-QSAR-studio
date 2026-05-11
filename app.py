from __future__ import annotations
import os
import sys
import platform

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontDatabase

from .ui.main_window import MainWindow


def _set_platform_font(app: QApplication) -> None:
    system = platform.system()
    if system == "Windows":
        candidates = ["Segoe UI", "Arial"]
    elif system == "Darwin":
        candidates = ["SF Pro Display", "Helvetica Neue", "Arial"]
    else:  # Linux / others
        candidates = ["Ubuntu", "Noto Sans", "Liberation Sans", "DejaVu Sans", "Arial"]

    available = QFontDatabase.families()
    for name in candidates:
        if name in available:
            app.setFont(QFont(name, 13))
            return
    # Fall back to system default
    f = app.font()
    f.setPointSize(13)
    app.setFont(f)


def main():
    # High-DPI support
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("ML-QSAR Studio")
    app.setApplicationDisplayName("ML-QSAR Studio")
    app.setOrganizationName("Agouni Lab")

    _set_platform_font(app)

    logo_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "logo.png")
    )
    if os.path.exists(logo_path):
        from PyQt6.QtGui import QIcon
        app.setWindowIcon(QIcon(logo_path))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
