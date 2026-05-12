from __future__ import annotations
import os
import json
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QCheckBox, QListWidget, QListWidgetItem,
    QSpinBox, QTextEdit, QComboBox, QMessageBox, QStackedWidget, QGroupBox,
    QFormLayout, QScrollArea, QFrame, QSizePolicy, QDoubleSpinBox,
    QAbstractItemView, QSplitter, QSpacerItem,
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QPixmap, QFont, QFontDatabase

import platform as _platform

from .worker import TrainWorker, WorkerThread
from ..backend.config import QSARConfig
from ..backend.predictor import predict_from_smiles


def _mono_font() -> str:
    """Return a monospace font stack that works on Windows, macOS, and Linux."""
    system = _platform.system()
    if system == "Windows":
        return "'Consolas', 'Courier New', monospace"
    elif system == "Darwin":
        return "'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace"
    else:
        return "'Ubuntu Mono', 'DejaVu Sans Mono', 'Liberation Mono', 'Courier New', monospace"


MONO = _mono_font()

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as _plt
    _plt.rcParams.update({"figure.dpi": 100})
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_CHOICES = [
    "RandomForest", "XGBoost", "LightGBM", "GradientBoosting", "HistGB",
    "ExtraTrees", "SVM", "KNN", "MLP", "ElasticNet", "PLS", "Ridge", "Stacking",
]
FP_CHOICES = ["ECFP4", "ECFP6", "FCFP4", "FCFP6", "MACCS", "RDKIT"]

BG_MAIN    = "#0f172a"
BG_SIDEBAR = "#0b1120"
BG_CARD    = "#1e293b"
BG_INPUT   = "#162032"
BORDER     = "#334155"
TEXT       = "#e2e8f0"
SUBTEXT    = "#94a3b8"
MUTED      = "#475569"
BLUE       = "#3b82f6"
TEAL       = "#14b8a6"
PURPLE     = "#8b5cf6"
GREEN      = "#10b981"
ORANGE     = "#f59e0b"
RED        = "#ef4444"
SB_ACTIVE  = "#1e3a5f"

CHART_BG   = "#1a2744"
CHART_AX   = "#94a3b8"
CHART_GRID = "#253452"

STYLESHEET = f"""
QMainWindow, QDialog {{
    background-color: {BG_MAIN};
    color: {TEXT};
}}
QWidget {{
    background-color: transparent;
    color: {TEXT};
    font-size: 13px;
}}
/* ── Scroll bars ── */
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {BG_CARD}; width: 7px; border-radius: 3px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {MUTED}; border-radius: 3px; min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar:horizontal {{
    background: {BG_CARD}; height: 7px; border-radius: 3px;
}}
QScrollBar::handle:horizontal {{
    background: {MUTED}; border-radius: 3px; min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
/* ── Inputs ── */
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 10px;
    selection-background-color: {BLUE};
}}
QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {BLUE};
}}
QComboBox {{
    background-color: {BG_INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 10px;
    min-width: 120px;
}}
QComboBox:focus {{ border-color: {BLUE}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {SUBTEXT};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    color: {TEXT};
    selection-background-color: {SB_ACTIVE};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {BG_CARD}; border: none; width: 18px;
    border-radius: 3px;
}}
/* ── Buttons ── */
QPushButton {{
    background-color: {BG_CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{ background-color: {BG_INPUT}; border-color: {BLUE}; }}
QPushButton:pressed {{ background-color: {SB_ACTIVE}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: {BORDER}; }}
QPushButton#btn_primary {{
    background-color: {BLUE}; color: #fff; border: none; font-weight: 600;
}}
QPushButton#btn_primary:hover {{ background-color: #2563eb; }}
QPushButton#btn_primary:disabled {{ background-color: {MUTED}; color: #8899aa; }}
QPushButton#btn_success {{
    background-color: {GREEN}; color: #fff; border: none; font-weight: 600;
}}
QPushButton#btn_success:hover {{ background-color: #059669; }}
QPushButton#btn_danger {{
    background-color: {RED}; color: #fff; border: none; font-weight: 600;
}}
QPushButton#btn_start {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {PURPLE}, stop:1 {BLUE});
    color: #fff; border: none; border-radius: 8px;
    font-weight: 700; font-size: 14px; padding: 12px 28px;
    min-height: 44px;
}}
QPushButton#btn_start:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c3aed, stop:1 #2563eb); }}
QPushButton#btn_start:disabled {{ background: {MUTED}; color: #8899aa; }}
/* ── Nav buttons ── */
QPushButton#nav_btn {{
    background: transparent; color: {SUBTEXT};
    border: none; border-radius: 8px;
    padding: 10px 12px; text-align: left;
    font-size: 13px;
}}
QPushButton#nav_btn:hover {{ background-color: {BG_CARD}; color: {TEXT}; }}
QPushButton#nav_btn[active="true"] {{
    background-color: {SB_ACTIVE}; color: {BLUE}; font-weight: 600;
}}
/* ── Checkboxes ── */
QCheckBox {{ color: {TEXT}; spacing: 8px; }}
QCheckBox::indicator {{
    width: 17px; height: 17px;
    border: 2px solid {BORDER}; border-radius: 4px;
    background: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {BLUE}; border-color: {BLUE};
}}
/* ── Lists ── */
QListWidget {{
    background-color: {BG_INPUT}; border: 1px solid {BORDER};
    border-radius: 6px; color: {TEXT}; padding: 4px;
}}
QListWidget::item {{ padding: 6px 8px; border-radius: 4px; }}
QListWidget::item:selected {{ background-color: {SB_ACTIVE}; color: {BLUE}; }}
QListWidget::item:hover:!selected {{ background-color: {BG_CARD}; }}
/* ── Group boxes ── */
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 8px;
    margin-top: 14px; padding: 12px 14px;
    color: {SUBTEXT}; font-size: 11px; font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: {SUBTEXT}; text-transform: uppercase; letter-spacing: 0.5px;
}}
/* ── Status bar ── */
QStatusBar {{
    background-color: {BG_SIDEBAR};
    color: {SUBTEXT};
    border-top: 1px solid {BORDER};
    font-size: 12px;
}}
"""


# ── Small helper widgets ───────────────────────────────────────────────────────
def _label(text: str, size: int = 13, bold: bool = False, color: str = TEXT) -> QLabel:
    lbl = QLabel(text)
    f = lbl.font()
    f.setPointSize(size)
    if bold:
        f.setBold(True)
    lbl.setFont(f)
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    return lbl


def _card(parent_layout, stretch: int = 0) -> QFrame:
    f = QFrame()
    f.setObjectName("card")
    f.setStyleSheet(
        f"QFrame#card {{ background-color: {BG_CARD}; border: 1px solid {BORDER};"
        f"border-radius: 10px; }}"
    )
    if parent_layout is not None:
        parent_layout.addWidget(f, stretch)
    return f


def _divider() -> QFrame:
    d = QFrame()
    d.setFrameShape(QFrame.Shape.HLine)
    d.setStyleSheet(f"color: {BORDER}; background: {BORDER}; max-height: 1px;")
    return d


def _metric_card(value: str, label: str, accent: str = BLUE) -> QFrame:
    card = QFrame()
    card.setStyleSheet(
        f"QFrame {{ background-color: {BG_CARD}; border: 1px solid {BORDER};"
        f"border-radius: 10px; border-left: 3px solid {accent}; }}"
    )
    v_lay = QVBoxLayout(card)
    v_lay.setContentsMargins(16, 14, 16, 14)
    v_lay.setSpacing(4)
    val_lbl = QLabel(value)
    val_lbl.setStyleSheet(f"color: {TEXT}; font-size: 26px; font-weight: 700; background: transparent;")
    lbl_lbl = QLabel(label)
    lbl_lbl.setStyleSheet(f"color: {SUBTEXT}; font-size: 11px; text-transform: uppercase; background: transparent;")
    v_lay.addWidget(val_lbl)
    v_lay.addWidget(lbl_lbl)
    return card


def _section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color: {TEXT}; font-size: 16px; font-weight: 700; background: transparent;"
        f"padding-bottom: 2px; border-bottom: 2px solid {BLUE};"
    )
    return lbl


def _scrollable(widget: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidget(widget)
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.Shape.NoFrame)
    return sa


def _browse_row(line_edit: QLineEdit, btn: QPushButton) -> QWidget:
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)
    h.addWidget(line_edit)
    h.addWidget(btn)
    return row


# ── Chart helpers ──────────────────────────────────────────────────────────────
def _new_figure(w: float = 5, h: float = 3.5) -> tuple:
    fig = Figure(figsize=(w, h), facecolor=CHART_BG, tight_layout=True)
    canvas = FigureCanvas(fig)
    canvas.setStyleSheet("background: transparent;")
    return fig, canvas


def _style_ax(ax):
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=CHART_AX, labelsize=9)
    ax.xaxis.label.set_color(CHART_AX)
    ax.yaxis.label.set_color(CHART_AX)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_GRID)
    ax.grid(color=CHART_GRID, linestyle="--", linewidth=0.5, alpha=0.7)


def _placeholder_canvas(message: str = "Run training to see chart") -> QWidget:
    w = QWidget()
    w.setStyleSheet(f"background-color: {CHART_BG}; border-radius: 8px;")
    lay = QVBoxLayout(w)
    lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    ico = _label("📊", 28)
    ico.setAlignment(Qt.AlignmentFlag.AlignCenter)
    msg = _label(message, 12, color=MUTED)
    msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(ico)
    lay.addWidget(msg)
    return w


# ── MainWindow ─────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    # page indices
    PAGE_DASH   = 0
    PAGE_DATA   = 1
    PAGE_FEAT   = 2
    PAGE_TRAIN  = 3
    PAGE_RES    = 4
    PAGE_PRED   = 5
    PAGE_AD     = 6
    PAGE_ABOUT  = 7

    _NAV_ITEMS = [
        ("  Dashboard",          "⊞", PAGE_DASH),
        ("  Data Acquisition",   "⬇", PAGE_DATA),
        ("  Feature Engineering","⚙", PAGE_FEAT),
        ("  Model Training",     "▶", PAGE_TRAIN),
        ("  Results & Analysis", "📈",PAGE_RES),
        ("  Prediction",         "⚡", PAGE_PRED),
        ("  Applicability Domain","🎯",PAGE_AD),
        ("  About",              "ℹ", PAGE_ABOUT),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML-QSAR Studio")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        self._project_dir      = ""
        self._notebook_path    = None
        self._use_notebook     = False
        self._train_results    = None
        self._nav_buttons: list[QPushButton] = []
        # Figure refs (UI dark versions) and publication plot data
        self._fig_obs    = None
        self._fig_resid  = None
        self._fig_yscr   = None
        self._fig_dist   = None
        self._fig_ad     = None
        self._fig_mdlcmp = None
        self._fig_pvr    = None
        self._plot_data: dict = {}

        self.setStyleSheet(STYLESHEET)
        self._setup_central()
        self._update_status("Ready", "")

    # ── Layout scaffolding ─────────────────────────────────────────────────────
    def _setup_central(self):
        root = QWidget()
        root.setStyleSheet(f"background-color: {BG_MAIN};")
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        # top header bar
        root_lay.addWidget(self._build_header())

        # body = sidebar + stack
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)
        body_lay.addWidget(self._build_sidebar())

        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background: transparent;")

        pages = [
            self._build_dashboard_page(),
            self._build_data_page(),
            self._build_features_page(),
            self._build_training_page(),
            self._build_results_page(),
            self._build_prediction_page(),
            self._build_ad_page(),
            self._build_about_page(),
        ]
        for p in pages:
            self.stack.addWidget(p)

        body_lay.addWidget(self.stack, 1)
        root_lay.addWidget(body, 1)

        # status bar
        self.status_bar = self._build_status_bar()
        root_lay.addWidget(self.status_bar)

        self._nav_to(self.PAGE_DASH)

    def _build_header(self) -> QFrame:
        hdr = QFrame()
        hdr.setFixedHeight(54)
        hdr.setStyleSheet(
            f"QFrame {{ background-color: {BG_SIDEBAR}; border-bottom: 1px solid {BORDER}; }}"
        )
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(16, 0, 16, 0)

        # Logo + title on the left
        logo_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "logo.png")
        )
        if os.path.exists(logo_path):
            pm = QPixmap(logo_path).scaled(
                34, 34, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_lbl = QLabel()
            logo_lbl.setPixmap(pm)
            logo_lbl.setStyleSheet("background: transparent;")
            lay.addWidget(logo_lbl)
            lay.addSpacing(8)

        title = _label("ML-QSAR Studio", 15, bold=True, color=TEXT)
        subtitle = _label("Machine Learning for Quantitative Structure-Activity Relationships",
                          10, color=SUBTEXT)
        txt_col = QVBoxLayout()
        txt_col.setSpacing(0)
        txt_col.addWidget(title)
        txt_col.addWidget(subtitle)
        lay.addLayout(txt_col)
        lay.addStretch(1)

        # Step pills
        step_labels = ["① Data", "② Features", "③ Models", "④ Results"]
        step_pages  = [self.PAGE_DATA, self.PAGE_FEAT, self.PAGE_TRAIN, self.PAGE_RES]
        self._step_btns: list[QPushButton] = []
        for txt, pg in zip(step_labels, step_pages):
            b = QPushButton(txt)
            b.setFixedHeight(30)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                f"QPushButton {{ background: {BG_CARD}; color: {SUBTEXT};"
                f"border: 1px solid {BORDER}; border-radius: 14px; padding: 0 14px;"
                f"font-size: 12px; }}"
                f"QPushButton:hover {{ border-color: {BLUE}; color: {TEXT}; }}"
            )
            b.clicked.connect(lambda _, p=pg: self._nav_to(p))
            lay.addWidget(b)
            self._step_btns.append(b)
            lay.addSpacing(4)

        return hdr

    def _build_sidebar(self) -> QFrame:
        sb = QFrame()
        sb.setFixedWidth(218)
        sb.setStyleSheet(
            f"QFrame {{ background-color: {BG_SIDEBAR}; border-right: 1px solid {BORDER}; }}"
        )
        lay = QVBoxLayout(sb)
        lay.setContentsMargins(10, 16, 10, 12)
        lay.setSpacing(2)

        # Nav buttons
        for (label, icon, page_idx) in self._NAV_ITEMS:
            btn = QPushButton(f"{icon}  {label.strip()}")
            btn.setObjectName("nav_btn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(40)
            btn.setCheckable(False)
            btn.clicked.connect(lambda _, p=page_idx: self._nav_to(p))
            lay.addWidget(btn)
            self._nav_buttons.append(btn)

        lay.addStretch(1)
        lay.addWidget(_divider())
        lay.addSpacing(10)

        # Lab logo + developer credit
        logo_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "logo.png")
        )
        if os.path.exists(logo_path):
            pm = QPixmap(logo_path).scaled(
                76, 76, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_lbl = QLabel()
            logo_lbl.setPixmap(pm)
            logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_lbl.setStyleSheet("background: transparent; margin-bottom: 6px;")
            lay.addWidget(logo_lbl)

        credit = QLabel(
            "<center><span style='color:#64748b;font-size:10px;'>"
            "Developed by<br>"
            "<b style='color:#94a3b8;'>Dr. Abbas Khan</b><br>"
            "<span style='color:#64748b;'>&amp;</span><br>"
            "<b style='color:#94a3b8;'>Dr. Abdelali Agouni</b>"
            "</span></center>"
        )
        credit.setWordWrap(True)
        credit.setStyleSheet("background: transparent; line-height: 160%;")
        lay.addWidget(credit)

        ver = _label("v1.0.0", 10, color=MUTED)
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(ver)
        return sb

    def _build_status_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(28)
        bar.setStyleSheet(
            f"QFrame {{ background-color: {BG_SIDEBAR}; border-top: 1px solid {BORDER}; }}"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        self._status_project = _label("No project", 11, color=MUTED)
        self._status_state   = _label("Ready", 11, color=GREEN)
        self._status_outdir  = _label("", 11, color=MUTED)
        lay.addWidget(self._status_project)
        lay.addSpacing(16)
        lay.addWidget(self._status_outdir)
        lay.addStretch(1)
        lay.addWidget(self._status_state)
        return bar

    def _update_status(self, state: str, outdir: str):
        color = GREEN if state == "Completed" else (ORANGE if state == "Training..." else SUBTEXT)
        self._status_state.setText(f"● {state}")
        self._status_state.setStyleSheet(f"color: {color}; background: transparent;")
        self._status_outdir.setText(outdir[:60] + ("…" if len(outdir) > 60 else ""))

    def _nav_to(self, page_idx: int):
        self.stack.setCurrentIndex(page_idx)
        for i, btn in enumerate(self._nav_buttons):
            is_active = (i == page_idx)
            btn.setProperty("active", "true" if is_active else "false")
            # Force style refresh
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ── Dashboard page ─────────────────────────────────────────────────────────
    def _build_dashboard_page(self) -> QWidget:
        outer = QWidget()
        outer.setStyleSheet(f"background: {BG_MAIN};")
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(24, 20, 24, 20)
        outer_lay.setSpacing(16)

        outer_lay.addWidget(_section_header("Dashboard"))

        # metric cards row
        self._dash_cards_row = QHBoxLayout()
        self._dash_cards_row.setSpacing(12)
        self._dash_metric_mol  = _metric_card("—", "Compounds Retrieved", TEAL)
        self._dash_metric_fs   = _metric_card("—", "Feature Sets", BLUE)
        self._dash_metric_mdl  = _metric_card("—", "Models Trained", PURPLE)
        self._dash_metric_r2   = _metric_card("—", "Best R² (Test)", GREEN)
        for w in [self._dash_metric_mol, self._dash_metric_fs,
                  self._dash_metric_mdl, self._dash_metric_r2]:
            self._dash_cards_row.addWidget(w)
        outer_lay.addLayout(self._dash_cards_row)

        # charts row
        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)

        # pIC50 distribution placeholder
        dist_card = _card(None)
        dist_lay  = QVBoxLayout(dist_card)
        dist_lay.setContentsMargins(12, 12, 12, 12)
        dist_lay.addWidget(_label("pIC50 Distribution", 12, bold=True, color=SUBTEXT))
        self._dash_dist_holder = QVBoxLayout()
        self._dash_dist_holder.addWidget(_placeholder_canvas("Run training to see distribution"))
        dist_lay.addLayout(self._dash_dist_holder, 1)
        charts_row.addWidget(dist_card, 1)

        # best model summary panel
        bms_card = _card(None)
        bms_card.setFixedWidth(260)
        bms_lay  = QVBoxLayout(bms_card)
        bms_lay.setContentsMargins(16, 14, 16, 14)
        bms_lay.setSpacing(8)
        bms_lay.addWidget(_label("Best Model Summary", 13, bold=True))
        bms_lay.addWidget(_divider())

        self._bms_labels: dict[str, QLabel] = {}
        for key in ["Model", "Feature Set", "Test R²", "Test RMSE",
                    "CV R² (Mean±Std)", "AD Mean Dist"]:
            row = QHBoxLayout()
            row.addWidget(_label(key + ":", 12, color=SUBTEXT))
            val = _label("—", 12, bold=True)
            self._bms_labels[key] = val
            row.addWidget(val)
            row.addStretch(1)
            bms_lay.addLayout(row)

        bms_lay.addStretch(1)
        btn_results = QPushButton("View Full Results →")
        btn_results.setObjectName("btn_primary")
        btn_results.clicked.connect(lambda: self._nav_to(self.PAGE_RES))
        bms_lay.addWidget(btn_results)
        btn_pred = QPushButton("Make Predictions →")
        btn_pred.clicked.connect(lambda: self._nav_to(self.PAGE_PRED))
        bms_lay.addWidget(btn_pred)

        charts_row.addWidget(bms_card)
        outer_lay.addLayout(charts_row, 1)

        # welcome note (hidden after training)
        self._welcome_widget = QLabel(
            "<center>"
            f"<p style='color:{SUBTEXT};font-size:13px;margin-top:8px;'>"
            "Configure your project in <b>Data Acquisition</b>, select features and models, "
            "then start training to see results here.</p>"
            "</center>"
        )
        self._welcome_widget.setWordWrap(True)
        outer_lay.addWidget(self._welcome_widget)

        return outer

    # ── Data Acquisition page ──────────────────────────────────────────────────
    def _build_data_page(self) -> QWidget:
        outer, lay = self._page_scaffold("Data Acquisition",
                                         "Configure ChEMBL target and project settings.")

        # Main form card
        card = _card(None)
        form_lay = QFormLayout(card)
        form_lay.setContentsMargins(20, 16, 20, 16)
        form_lay.setSpacing(14)
        form_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("Select output directory…")
        btn_dir = QPushButton("Browse…")
        btn_dir.setFixedWidth(90)
        btn_dir.clicked.connect(self._pick_dir)
        form_lay.addRow("Output directory:", _browse_row(self.dir_edit, btn_dir))

        self.chembl_edit = QLineEdit("CHEMBL3629")
        self.chembl_edit.setPlaceholderText("e.g. CHEMBL3629")
        form_lay.addRow("ChEMBL Target ID:", self.chembl_edit)

        self.std_type = QComboBox()
        self.std_type.addItems(["IC50", "Ki", "EC50", "Kd", "Potency"])
        form_lay.addRow("Standard Type:", self.std_type)

        self.std_units = QComboBox()
        self.std_units.addItems(["nM", "uM", "pM", "M"])
        form_lay.addRow("Units:", self.std_units)

        form_lay.addRow("", _divider())

        self.use_notebook_chk = QCheckBox("Use custom notebook (.ipynb) instead of built-in pipeline")
        self.use_notebook_chk.stateChanged.connect(self._toggle_notebook_mode)
        form_lay.addRow("", self.use_notebook_chk)

        self.nb_path_edit = QLineEdit()
        self.nb_path_edit.setEnabled(False)
        self.nb_path_edit.setPlaceholderText("Select .ipynb file…")
        btn_nb = QPushButton("Select…")
        btn_nb.setFixedWidth(90)
        btn_nb.setEnabled(False)
        btn_nb.clicked.connect(self._pick_notebook)
        self._btn_nb = btn_nb
        form_lay.addRow("Notebook:", _browse_row(self.nb_path_edit, btn_nb))

        lay.addWidget(card)

        # nav
        lay.addStretch(1)
        nav = self._nav_row(back=None, forward=("Next: Feature Engineering →", self.PAGE_FEAT))
        lay.addLayout(nav)

        return _scrollable(outer)

    # ── Feature Engineering page ───────────────────────────────────────────────
    def _build_features_page(self) -> QWidget:
        outer, lay = self._page_scaffold("Feature Engineering",
                                         "Select molecular representations to use as features.")

        # descriptor toggles card
        tog_card = _card(None)
        tog_lay  = QVBoxLayout(tog_card)
        tog_lay.setContentsMargins(20, 16, 20, 16)
        tog_lay.setSpacing(10)
        tog_lay.addWidget(_label("Descriptor Blocks", 12, bold=True, color=SUBTEXT))

        self.chk_2d = QCheckBox("2D Descriptors  (MolWt, LogP, TPSA, HBA, HBD … 19 features)")
        self.chk_2d.setChecked(True)
        self.chk_3d = QCheckBox("3D Descriptors  (Asphericity, PMI, RadiusOfGyration … 10 features)")
        self.chk_3d.setChecked(True)
        self.chk_fp = QCheckBox("Fingerprints  (ECFP, FCFP, MACCS, RDKIT)")
        self.chk_fp.setChecked(True)
        for ck in [self.chk_2d, self.chk_3d, self.chk_fp]:
            tog_lay.addWidget(ck)
        lay.addWidget(tog_card)

        # fingerprint config card
        fp_card = _card(None)
        fp_lay  = QVBoxLayout(fp_card)
        fp_lay.setContentsMargins(20, 16, 20, 16)
        fp_lay.setSpacing(10)
        fp_lay.addWidget(_label("Fingerprint Configuration", 12, bold=True, color=SUBTEXT))

        fp_lay.addWidget(_label("Select fingerprint types (multi-select):", 12, color=SUBTEXT))
        self.fp_list = QListWidget()
        self.fp_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.fp_list.setMaximumHeight(160)
        for fp in FP_CHOICES:
            it = QListWidgetItem(fp)
            self.fp_list.addItem(it)
            it.setSelected(True)
        fp_lay.addWidget(self.fp_list)

        nbits_row = QHBoxLayout()
        nbits_row.addWidget(_label("Fingerprint bits (nBits):", 12, color=SUBTEXT))
        self.fp_nbits = QSpinBox()
        self.fp_nbits.setRange(256, 4096)
        self.fp_nbits.setSingleStep(256)
        self.fp_nbits.setValue(1024)
        self.fp_nbits.setFixedWidth(100)
        nbits_row.addWidget(self.fp_nbits)
        nbits_row.addStretch(1)
        fp_lay.addLayout(nbits_row)
        lay.addWidget(fp_card)

        lay.addStretch(1)
        nav = self._nav_row(back=("← Data Acquisition", self.PAGE_DATA),
                            forward=("Next: Model Training →", self.PAGE_TRAIN))
        lay.addLayout(nav)

        return _scrollable(outer)

    # ── Model Training page ────────────────────────────────────────────────────
    def _build_training_page(self) -> QWidget:
        outer = QWidget()
        outer.setStyleSheet(f"background: {BG_MAIN};")
        main_lay = QVBoxLayout(outer)
        main_lay.setContentsMargins(24, 20, 24, 20)
        main_lay.setSpacing(14)

        main_lay.addWidget(_section_header("Model Training"))
        main_lay.addWidget(_label("Select algorithms and configure hyperparameter search.",
                                  12, color=SUBTEXT))

        # top row: model list + settings
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        # model list card
        ml_card = _card(None)
        ml_lay  = QVBoxLayout(ml_card)
        ml_lay.setContentsMargins(16, 14, 16, 14)
        ml_lay.setSpacing(8)
        ml_lay.addWidget(_label("Algorithms", 12, bold=True, color=SUBTEXT))
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for m in MODEL_CHOICES:
            it = QListWidgetItem(m)
            self.model_list.addItem(it)
            it.setSelected(True)
        ml_lay.addWidget(self.model_list)

        btn_row = QHBoxLayout()
        btn_all  = QPushButton("Select All")
        btn_none = QPushButton("Clear")
        btn_all.clicked.connect(lambda: [self.model_list.item(i).setSelected(True)
                                          for i in range(self.model_list.count())])
        btn_none.clicked.connect(lambda: [self.model_list.item(i).setSelected(False)
                                           for i in range(self.model_list.count())])
        btn_row.addWidget(btn_all); btn_row.addWidget(btn_none); btn_row.addStretch(1)
        ml_lay.addLayout(btn_row)
        top_row.addWidget(ml_card, 1)

        # settings card
        cfg_card = _card(None)
        cfg_card.setFixedWidth(260)
        cfg_lay  = QFormLayout(cfg_card)
        cfg_lay.setContentsMargins(16, 14, 16, 14)
        cfg_lay.setSpacing(12)
        cfg_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.trials = QSpinBox()
        self.trials.setRange(1, 500); self.trials.setValue(30)
        self.folds = QSpinBox()
        self.folds.setRange(2, 10); self.folds.setValue(5)
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.05, 0.5); self.test_size.setSingleStep(0.05)
        self.test_size.setValue(0.2); self.test_size.setDecimals(2)
        self.val_size = QDoubleSpinBox()
        self.val_size.setRange(0.05, 0.4); self.val_size.setSingleStep(0.05)
        self.val_size.setValue(0.1); self.val_size.setDecimals(2)
        self.use_gpu = QCheckBox("Use GPU if available")
        self.use_gpu.setChecked(True)

        self.select_by = QComboBox()
        self.select_by.addItems(["test_r2", "cv_mean"])

        cfg_lay.addRow(_label("Optuna trials:", 12, color=SUBTEXT), self.trials)
        cfg_lay.addRow(_label("CV folds:", 12, color=SUBTEXT), self.folds)
        cfg_lay.addRow(_label("Test size:", 12, color=SUBTEXT), self.test_size)
        cfg_lay.addRow(_label("Val size:", 12, color=SUBTEXT), self.val_size)
        cfg_lay.addRow(_label("Select best by:", 12, color=SUBTEXT), self.select_by)
        cfg_lay.addRow("", self.use_gpu)
        top_row.addWidget(cfg_card)
        main_lay.addLayout(top_row)

        # log area
        log_card = _card(None)
        log_lay  = QVBoxLayout(log_card)
        log_lay.setContentsMargins(14, 12, 14, 12)
        log_lay.setSpacing(6)
        log_lay.addWidget(_label("Training Log", 12, bold=True, color=SUBTEXT))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(
            f"background-color: #0d1525; color: {GREEN};"
            f"border: 1px solid {BORDER}; border-radius: 6px;"
            f"font-family: {MONO}; font-size: 12px;"
            f"padding: 8px;"
        )
        self.log.setMinimumHeight(200)
        log_lay.addWidget(self.log)
        main_lay.addWidget(log_card, 1)

        # action buttons
        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        back_btn = QPushButton("← Feature Engineering")
        back_btn.clicked.connect(lambda: self._nav_to(self.PAGE_FEAT))

        self.btn_start = QPushButton("▶  Run Training")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setMinimumWidth(180)
        self.btn_start.clicked.connect(self._start_training)

        self.btn_to_results = QPushButton("View Results →")
        self.btn_to_results.setObjectName("btn_primary")
        self.btn_to_results.setEnabled(False)
        self.btn_to_results.clicked.connect(lambda: self._nav_to(self.PAGE_RES))

        action_row.addWidget(back_btn)
        action_row.addStretch(1)
        action_row.addWidget(self.btn_start)
        action_row.addWidget(self.btn_to_results)
        main_lay.addLayout(action_row)

        return outer

    # ── Results & Analysis page ────────────────────────────────────────────────
    def _build_results_page(self) -> QWidget:
        outer = QWidget()
        outer.setStyleSheet(f"background: {BG_MAIN};")
        main_lay = QVBoxLayout(outer)
        main_lay.setContentsMargins(24, 20, 24, 20)
        main_lay.setSpacing(14)
        main_lay.addWidget(_section_header("Results & Analysis"))

        # Model ranking card
        ranking_card = _card(None)
        ranking_card.setFixedHeight(180)
        rl = QVBoxLayout(ranking_card)
        rl.setContentsMargins(14, 12, 14, 12)
        rl.addWidget(_label("Model Leaderboard  (all feature sets)", 12, bold=True, color=SUBTEXT))
        self._results_log = QTextEdit()
        self._results_log.setReadOnly(True)
        self._results_log.setStyleSheet(
            f"background: #0d1525; color: {TEXT}; border: none;"
            f"font-family: {MONO}; font-size: 11px;"
        )
        self._results_log.setPlaceholderText("Training results will appear here…")
        rl.addWidget(self._results_log)
        main_lay.addWidget(ranking_card)

        # ── Chart row 1: obs-vs-pred | residuals | y-scrambling ─────────────────
        charts_row = QHBoxLayout()
        charts_row.setSpacing(12)

        self._res_obs_card = _card(None)
        obs_lay = QVBoxLayout(self._res_obs_card)
        obs_lay.setContentsMargins(12, 10, 12, 10)
        obs_lay.addWidget(_label("Observed vs Predicted", 11, bold=True, color=SUBTEXT))
        self._obs_chart_holder = QVBoxLayout()
        self._obs_chart_holder.addWidget(_placeholder_canvas())
        obs_lay.addLayout(self._obs_chart_holder, 1)
        charts_row.addWidget(self._res_obs_card, 1)

        self._res_resid_card = _card(None)
        res_lay = QVBoxLayout(self._res_resid_card)
        res_lay.setContentsMargins(12, 10, 12, 10)
        res_lay.addWidget(_label("Residuals Distribution", 11, bold=True, color=SUBTEXT))
        self._resid_chart_holder = QVBoxLayout()
        self._resid_chart_holder.addWidget(_placeholder_canvas())
        res_lay.addLayout(self._resid_chart_holder, 1)
        charts_row.addWidget(self._res_resid_card, 1)

        self._res_yscr_card = _card(None)
        yscr_lay = QVBoxLayout(self._res_yscr_card)
        yscr_lay.setContentsMargins(12, 10, 12, 10)
        yscr_lay.addWidget(_label("Y-Scrambling Validation (R²)", 11, bold=True, color=SUBTEXT))
        self._yscr_chart_holder = QVBoxLayout()
        self._yscr_chart_holder.addWidget(_placeholder_canvas())
        yscr_lay.addLayout(self._yscr_chart_holder, 1)
        charts_row.addWidget(self._res_yscr_card, 1)

        main_lay.addLayout(charts_row, 1)

        # ── Chart row 2: model comparison | predicted vs residuals ─────────────
        charts_row2 = QHBoxLayout()
        charts_row2.setSpacing(12)

        self._res_mdlcmp_card = _card(None)
        mc_lay = QVBoxLayout(self._res_mdlcmp_card)
        mc_lay.setContentsMargins(12, 10, 12, 10)
        mc_lay.addWidget(_label("Model Comparison  (Train vs Test R²)", 11, bold=True, color=SUBTEXT))
        self._mdlcmp_chart_holder = QVBoxLayout()
        self._mdlcmp_chart_holder.addWidget(_placeholder_canvas())
        mc_lay.addLayout(self._mdlcmp_chart_holder, 1)
        charts_row2.addWidget(self._res_mdlcmp_card, 1)

        self._res_pvr_card = _card(None)
        pvr_lay = QVBoxLayout(self._res_pvr_card)
        pvr_lay.setContentsMargins(12, 10, 12, 10)
        pvr_lay.addWidget(_label("Predicted vs Residuals", 11, bold=True, color=SUBTEXT))
        self._pvr_chart_holder = QVBoxLayout()
        self._pvr_chart_holder.addWidget(_placeholder_canvas())
        pvr_lay.addLayout(self._pvr_chart_holder, 1)
        charts_row2.addWidget(self._res_pvr_card, 1)

        main_lay.addLayout(charts_row2, 1)

        nav_row = QHBoxLayout()
        btn_back = QPushButton("← Model Training")
        btn_back.clicked.connect(lambda: self._nav_to(self.PAGE_TRAIN))
        btn_ad = QPushButton("Applicability Domain →")
        btn_ad.setObjectName("btn_primary")
        btn_ad.clicked.connect(lambda: self._nav_to(self.PAGE_AD))
        nav_row.addWidget(btn_back); nav_row.addStretch(1); nav_row.addWidget(btn_ad)
        main_lay.addLayout(nav_row)

        return outer

    # ── Prediction page ────────────────────────────────────────────────────────
    def _build_prediction_page(self) -> QWidget:
        outer, lay = self._page_scaffold("Prediction",
                                         "Predict pIC50 values for novel compounds.")

        # model files card
        files_card = _card(None)
        files_lay  = QFormLayout(files_card)
        files_lay.setContentsMargins(20, 16, 20, 16)
        files_lay.setSpacing(12)
        files_lay.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("best_model.pkl")
        btn_model = QPushButton("Browse…")
        btn_model.setFixedWidth(90)
        btn_model.clicked.connect(self._pick_model)
        files_lay.addRow("Model (.pkl):", _browse_row(self.model_path, btn_model))

        self.meta_path = QLineEdit()
        self.meta_path.setPlaceholderText("best_model_meta.json")
        btn_meta = QPushButton("Browse…")
        btn_meta.setFixedWidth(90)
        btn_meta.clicked.connect(self._pick_meta)
        files_lay.addRow("Metadata (.json):", _browse_row(self.meta_path, btn_meta))

        btn_autoload = QPushButton("Auto-load from training output")
        btn_autoload.clicked.connect(self._autoload_model)
        files_lay.addRow("", btn_autoload)
        lay.addWidget(files_card)

        # SMILES input card
        smiles_card = _card(None)
        sm_lay = QVBoxLayout(smiles_card)
        sm_lay.setContentsMargins(16, 14, 16, 14)
        sm_lay.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.addWidget(_label("SMILES Input", 12, bold=True, color=SUBTEXT))
        top_row.addStretch(1)
        btn_load_smiles = QPushButton("Load file…")
        btn_load_smiles.clicked.connect(self._pick_smiles_file)
        top_row.addWidget(btn_load_smiles)
        sm_lay.addLayout(top_row)

        self.smiles_text = QTextEdit()
        self.smiles_text.setPlaceholderText(
            "Paste SMILES here, one per line:\n"
            "CC(=O)Oc1ccccc1C(=O)O\n"
            "c1ccc(cc1)CC(=O)O"
        )
        self.smiles_text.setStyleSheet(
            f"background: #0d1525; color: {TEXT}; border: 1px solid {BORDER};"
            f"border-radius: 6px; font-family: {MONO}; font-size: 12px; padding: 8px;"
        )
        self.smiles_text.setMaximumHeight(160)
        sm_lay.addWidget(self.smiles_text)
        lay.addWidget(smiles_card)

        self.btn_predict = QPushButton("⚡  Run Prediction")
        self.btn_predict.setObjectName("btn_primary")
        self.btn_predict.clicked.connect(self._predict)
        lay.addWidget(self.btn_predict)

        # output card
        out_card = _card(None)
        out_lay  = QVBoxLayout(out_card)
        out_lay.setContentsMargins(14, 12, 14, 12)
        out_lay.addWidget(_label("Results", 12, bold=True, color=SUBTEXT))
        self.pred_out = QTextEdit()
        self.pred_out.setReadOnly(True)
        self.pred_out.setStyleSheet(
            f"background: #0d1525; color: {GREEN}; border: none;"
            f"font-family: {MONO}; font-size: 12px;"
        )
        self.pred_out.setMinimumHeight(160)
        out_lay.addWidget(self.pred_out)
        lay.addWidget(out_card, 1)

        lay.addStretch(0)
        nav = self._nav_row(back=("← Results", self.PAGE_RES), forward=None)
        lay.addLayout(nav)

        return _scrollable(outer)

    # ── Applicability Domain page ──────────────────────────────────────────────
    def _build_ad_page(self) -> QWidget:
        outer = QWidget()
        outer.setStyleSheet(f"background: {BG_MAIN};")
        main_lay = QVBoxLayout(outer)
        main_lay.setContentsMargins(24, 20, 24, 20)
        main_lay.setSpacing(14)
        main_lay.addWidget(_section_header("Applicability Domain"))
        main_lay.addWidget(
            _label("K-nearest neighbour (k-NN) based applicability domain analysis. "
                   "Lower mean distance = test compounds are more similar to training set.",
                   12, color=SUBTEXT)
        )

        # stats card
        stats_card = _card(None)
        stats_lay  = QHBoxLayout(stats_card)
        stats_lay.setContentsMargins(20, 14, 20, 14)
        stats_lay.setSpacing(30)

        self._ad_stat_labels: dict[str, QLabel] = {}
        for key in ["Best Model", "Feature Set", "AD Mean k-NN Dist",
                    "90% PI Coverage", "PI Width (Mean)"]:
            col = QVBoxLayout()
            col.setSpacing(4)
            val = _label("—", 18, bold=True)
            lbl = _label(key, 11, color=SUBTEXT)
            self._ad_stat_labels[key] = val
            col.addWidget(val)
            col.addWidget(lbl)
            stats_lay.addLayout(col)

        main_lay.addWidget(stats_card)

        # chart + explanation row
        chart_row = QHBoxLayout()
        chart_row.setSpacing(12)

        ad_chart_card = _card(None)
        ac_lay = QVBoxLayout(ad_chart_card)
        ac_lay.setContentsMargins(12, 10, 12, 10)
        ac_lay.addWidget(_label("AD Mean Distance per Model (by Feature Set)", 11, bold=True, color=SUBTEXT))
        self._ad_chart_holder = QVBoxLayout()
        self._ad_chart_holder.addWidget(_placeholder_canvas("Run training to see AD analysis"))
        ac_lay.addLayout(self._ad_chart_holder, 1)
        chart_row.addWidget(ad_chart_card, 2)

        # explanation card
        exp_card = _card(None)
        exp_card.setFixedWidth(260)
        exp_lay = QVBoxLayout(exp_card)
        exp_lay.setContentsMargins(16, 14, 16, 14)
        exp_lay.setSpacing(8)
        exp_lay.addWidget(_label("What is AD?", 13, bold=True))
        explanation = QLabel(
            "<p style='color:#94a3b8;font-size:12px;line-height:160%;'>"
            "The <b>Applicability Domain</b> defines the chemical space where "
            "predictions are reliable.<br><br>"
            "We use <b>k-NN mean distance</b> in transformed feature space "
            "(after imputation + scaling + optional feature selection).<br><br>"
            "<b>Interpretation:</b><br>"
            "• Low distance → test molecules similar to training set → reliable predictions<br>"
            "• High distance → extrapolation region → less reliable<br><br>"
            "A distance threshold can be set as mean + 3×std of training pairwise distances."
            "</p>"
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("background: transparent;")
        exp_lay.addWidget(explanation)
        exp_lay.addStretch(1)
        chart_row.addWidget(exp_card)

        main_lay.addLayout(chart_row, 1)

        nav_row = QHBoxLayout()
        btn_back = QPushButton("← Results & Analysis")
        btn_back.clicked.connect(lambda: self._nav_to(self.PAGE_RES))
        nav_row.addWidget(btn_back); nav_row.addStretch(1)
        main_lay.addLayout(nav_row)

        return outer

    # ── About page ─────────────────────────────────────────────────────────────
    def _build_about_page(self) -> QWidget:
        outer, lay = self._page_scaffold("About ML-QSAR Studio", "")

        card = _card(None)
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(24, 20, 24, 24)
        card_lay.setSpacing(14)

        # Logo centred
        logo_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "logo.png")
        )
        if os.path.exists(logo_path):
            pm = QPixmap(logo_path).scaled(
                100, 100, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_lbl = QLabel()
            logo_lbl.setPixmap(pm)
            logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_lbl.setStyleSheet("background: transparent;")
            card_lay.addWidget(logo_lbl)

        title = QLabel("<center><b style='font-size:18px;'>ML-QSAR Studio</b><br>"
                       "<span style='color:#94a3b8;font-size:13px;'>v1.0.0</span></center>")
        title.setStyleSheet("background: transparent;")
        card_lay.addWidget(title)
        card_lay.addWidget(_divider())

        body = QLabel(
            "<div style='color:#94a3b8;font-size:13px;line-height:170%;'>"
            "<p><b style='color:#e2e8f0;'>ML-QSAR Studio</b> is an end-to-end AutoML platform "
            "for building Quantitative Structure-Activity Relationship (QSAR) models "
            "from ChEMBL bioactivity data.</p>"
            "<p><b style='color:#e2e8f0;'>Key Features:</b><br>"
            "• Automated data retrieval from ChEMBL API<br>"
            "• 2D/3D molecular descriptors + multiple fingerprint types<br>"
            "• Scaffold-based train/test splitting (prevents data leakage)<br>"
            "• 13 ML algorithms with cross-validation<br>"
            "• Applicability domain (k-NN) analysis<br>"
            "• Y-scrambling validation (guards against chance correlations)<br>"
            "• Conformal prediction intervals (MAPIE)<br>"
            "• Deployable sklearn pipeline (.pkl)</p>"
            "<p><b style='color:#e2e8f0;'>Developed by:</b><br>"
            "Dr. Abbas Khan<br>"
            "Dr. Abdelali Agouni<br>"
            "<i>Agouni Lab — Precision Pharmacology &amp; AI-Drug Discovery</i></p>"
            "<p><b style='color:#e2e8f0;'>Dependencies:</b> "
            "PyQt6, scikit-learn, XGBoost, LightGBM, RDKit, molfeat, datamol, "
            "ChEMBL webresource client, Optuna, MAPIE, matplotlib</p>"
            "</div>"
        )
        body.setWordWrap(True)
        body.setStyleSheet("background: transparent;")
        card_lay.addWidget(body)
        card_lay.addStretch(1)

        lay.addWidget(card)
        lay.addStretch(1)

        return _scrollable(outer)

    # ── Page scaffold helper ───────────────────────────────────────────────────
    def _page_scaffold(self, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        outer = QWidget()
        outer.setStyleSheet(f"background: {BG_MAIN};")
        lay = QVBoxLayout(outer)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)
        lay.addWidget(_section_header(title))
        if subtitle:
            lay.addWidget(_label(subtitle, 12, color=SUBTEXT))
        return outer, lay

    def _nav_row(self, back=None, forward=None) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        if back:
            txt, pg = back
            b = QPushButton(txt)
            b.clicked.connect(lambda: self._nav_to(pg))
            row.addWidget(b)
        row.addStretch(1)
        if forward:
            txt, pg = forward
            b = QPushButton(txt)
            b.setObjectName("btn_primary")
            b.clicked.connect(lambda: self._nav_to(pg))
            row.addWidget(b)
        return row

    # ── Training logic ─────────────────────────────────────────────────────────
    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self._project_dir = d
            self.dir_edit.setText(d)
            self._status_project.setText(os.path.basename(d))

    def _toggle_notebook_mode(self):
        self._use_notebook = self.use_notebook_chk.isChecked()
        self.nb_path_edit.setEnabled(self._use_notebook)
        self._btn_nb.setEnabled(self._use_notebook)

    def _pick_notebook(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select notebook",
                                              filter="Jupyter Notebook (*.ipynb)")
        if path:
            self._notebook_path = path
            self.nb_path_edit.setText(path)

    def _gather_config(self) -> QSARConfig:
        outdir = self.dir_edit.text().strip()
        if not outdir:
            raise ValueError("Output directory is required.")
        chembl = self.chembl_edit.text().strip().upper()
        if not chembl:
            raise ValueError("ChEMBL Target ID is required.")
        fp_types = [it.text() for it in self.fp_list.selectedItems()] or ["ECFP4"]
        models   = [it.text() for it in self.model_list.selectedItems()] or ["RandomForest"]
        return QSARConfig(
            output_dir=outdir,
            target_chembl_id=chembl,
            standard_type=self.std_type.currentText(),
            standard_units=self.std_units.currentText(),
            test_size=self.test_size.value(),
            val_size=self.val_size.value(),
            use_2d=self.chk_2d.isChecked(),
            use_3d=self.chk_3d.isChecked(),
            use_fp=self.chk_fp.isChecked(),
            fp_types=fp_types,
            fp_nbits=int(self.fp_nbits.value()),
            enabled_models=models,
            n_trials=int(self.trials.value()),
            n_folds=int(self.folds.value()),
            use_gpu=self.use_gpu.isChecked(),
            select_best_by=self.select_by.currentText(),
        )

    def _start_training(self):
        try:
            cfg = self._gather_config()
            if self._use_notebook and not self._notebook_path:
                raise ValueError("Notebook mode is enabled but no .ipynb file selected.")
        except Exception as exc:
            QMessageBox.critical(self, "Configuration Error", str(exc))
            return

        self.log.clear()
        self.btn_start.setEnabled(False)
        self.btn_to_results.setEnabled(False)
        self._update_status("Training…", cfg.output_dir)

        worker = TrainWorker(cfg, use_notebook=self._use_notebook,
                             notebook_path=self._notebook_path)
        thread = WorkerThread(worker)
        self._thread = thread
        self._worker = worker

        worker.log.connect(self._append_log)
        worker.finished.connect(self._train_finished)
        worker.failed.connect(self._train_failed)

        thread.start()

    def _append_log(self, msg: str):
        self.log.append(msg)

    def _train_finished(self, res: dict):
        self._train_results = res
        self._append_log("\n✅ Training complete.")
        self.btn_start.setEnabled(True)
        self.btn_to_results.setEnabled(True)
        self._update_status("Completed", self.dir_edit.text().strip())
        self._populate_results(res)
        outdir = self.dir_edit.text().strip()
        if outdir and HAS_MPL:
            saved = self._save_all_figures(outdir)
            for path in saved:
                self._append_log(f"  Figure saved → {path}")

    def _train_failed(self, tb: str):
        self.btn_start.setEnabled(True)
        self._update_status("Failed", "")
        QMessageBox.critical(self, "Training Failed", tb[:2000])

    # ── Populate result pages after training ───────────────────────────────────
    def _populate_results(self, res: dict):
        best = res.get("best", {})
        all_results = res.get("results", {})
        residuals   = res.get("residuals", {})
        yscr_viz    = res.get("yscramble_viz", [])
        n_mol       = res.get("n_molecules", 0)
        n_fs        = res.get("n_feature_sets", 0)

        best_model = best.get("best_model", "—")
        best_fs    = best.get("best_feature_set", "—")
        best_r2    = best.get("best_score", float("nan"))

        # ── Count models ──────────────────────────────────────────────────────
        n_models = sum(
            1 for mods in all_results.values()
            for m, v in mods.items()
            if isinstance(v, dict) and "test" in v
        )

        # ── Dashboard cards ──────────────────────────────────────────────────
        self._dash_metric_mol.findChildren(QLabel)[0].setText(str(n_mol))
        self._dash_metric_fs.findChildren(QLabel)[0].setText(str(n_fs))
        self._dash_metric_mdl.findChildren(QLabel)[0].setText(str(n_models))
        self._dash_metric_r2.findChildren(QLabel)[0].setText(
            f"{best_r2:.3f}" if np.isfinite(best_r2) else "—"
        )

        # ── Best model summary ────────────────────────────────────────────────
        best_metrics = all_results.get(best_fs, {}).get(best_model, {})
        test_r2   = best_metrics.get("test", {}).get("r2", float("nan"))
        test_rmse = best_metrics.get("test", {}).get("rmse", float("nan"))
        cv_mean   = best_metrics.get("cv_mean", float("nan"))
        cv_std    = best_metrics.get("cv_std", float("nan"))
        ad_dist   = best_metrics.get("ad_mean_dist", float("nan"))
        cov90     = best_metrics.get("coverage_90pi", float("nan"))
        piw       = best_metrics.get("pi_width_mean", float("nan"))

        def _fmt(v): return f"{v:.3f}" if np.isfinite(float(v)) else "—"

        self._bms_labels["Model"].setText(best_model)
        self._bms_labels["Feature Set"].setText(best_fs)
        self._bms_labels["Test R²"].setText(_fmt(test_r2))
        self._bms_labels["Test RMSE"].setText(_fmt(test_rmse))
        cv_txt = f"{cv_mean:.3f} ± {cv_std:.3f}" if np.isfinite(cv_mean) else "—"
        self._bms_labels["CV R² (Mean±Std)"].setText(cv_txt)
        self._bms_labels["AD Mean Dist"].setText(_fmt(ad_dist))
        self._welcome_widget.hide()

        # ── AD page stats ─────────────────────────────────────────────────────
        self._ad_stat_labels["Best Model"].setText(best_model)
        self._ad_stat_labels["Feature Set"].setText(best_fs)
        self._ad_stat_labels["AD Mean k-NN Dist"].setText(_fmt(ad_dist))
        self._ad_stat_labels["90% PI Coverage"].setText(_fmt(cov90))
        self._ad_stat_labels["PI Width (Mean)"].setText(_fmt(piw))

        # ── Results log table ─────────────────────────────────────────────────
        lines = [f"{'FeatureSet':<12} {'Model':<20} {'TestR2':>8} {'TrainR2':>8} "
                 f"{'CV_Mean':>8} {'CV_Std':>7} {'AD_Dist':>8} {'Yscr_R2':>8}"]
        lines.append("─" * 85)
        rows = []
        for fs, mods in all_results.items():
            for mname, met in mods.items():
                if not isinstance(met, dict) or "test" not in met:
                    continue
                rows.append((
                    fs, mname,
                    met["test"].get("r2", float("nan")),
                    met["train"].get("r2", float("nan")),
                    met.get("cv_mean", float("nan")),
                    met.get("cv_std", float("nan")),
                    met.get("ad_mean_dist", float("nan")),
                    met.get("yscr_r2", float("nan")),
                ))
        rows.sort(key=lambda r: r[2] if np.isfinite(r[2]) else -999, reverse=True)
        for r in rows:
            marker = " ★" if (r[0] == best_fs and r[1] == best_model) else ""
            lines.append(
                f"{r[0]:<12} {r[1]:<20} {r[2]:>8.3f} {r[3]:>8.3f} "
                f"{r[4]:>8.3f} {r[5]:>7.3f} {r[6]:>8.3f} {r[7]:>8.3f}{marker}"
            )
        self._results_log.setPlainText("\n".join(lines))

        # ── Charts ────────────────────────────────────────────────────────────
        if not HAS_MPL:
            return

        y_test  = residuals.get("y_test", [])
        y_pred  = residuals.get("y_pred", [])

        y_train     = residuals.get("y_train", [])
        y_pred_train = residuals.get("y_pred_train", [])

        # Store for publication-quality save
        self._plot_data = {
            "y_test": y_test, "y_pred": y_pred,
            "y_train": y_train, "y_pred_train": y_pred_train,
            "yscr_viz": yscr_viz,
            "best_r2": best_r2, "best_model": best_model, "best_fs": best_fs,
            "all_results": all_results,
        }

        if y_test and y_pred:
            self._draw_obs_vs_pred(y_test, y_pred, y_train, y_pred_train,
                                   best_model, best_fs, best_r2)
            self._draw_residuals(y_test, y_pred, y_train, y_pred_train)
            self._draw_dist_dashboard(y_test, y_pred, y_train)
            self._draw_pred_vs_resid(y_test, y_pred, y_train, y_pred_train)

        if yscr_viz:
            self._draw_yscrambling(yscr_viz, best_r2)

        self._draw_model_comparison(all_results, best_fs)
        self._draw_ad_chart(all_results)

    def _clear_holder(self, holder: QVBoxLayout):
        while holder.count():
            item = holder.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _draw_obs_vs_pred(self, y_test, y_pred, y_train, y_pred_train,
                          model, fs, r2):
        self._clear_holder(self._obs_chart_holder)
        fig, canvas = _new_figure(4.5, 3.2)
        self._fig_obs = fig
        ax = fig.add_subplot(111)
        _style_ax(ax)
        arr_t = np.array(y_test, dtype=float)
        arr_p = np.array(y_pred, dtype=float)
        all_vals = list(arr_t) + list(arr_p)
        if y_train and y_pred_train:
            tr_t = np.array(y_train, dtype=float)
            tr_p = np.array(y_pred_train, dtype=float)
            all_vals += list(tr_t) + list(tr_p)
            ax.scatter(tr_t, tr_p, s=10, alpha=0.30, c="#94a3b8",
                       edgecolors="none", zorder=2, label="Train")
        ax.scatter(arr_t, arr_p, s=20, alpha=0.75, c=BLUE,
                   edgecolors="none", zorder=3, label="Test")
        mn, mx = min(all_vals), max(all_vals)
        pad = (mx - mn) * 0.05
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "--",
                color=ORANGE, lw=1.5, alpha=0.9, zorder=2)
        ax.set_xlabel("Observed pIC50", fontsize=9)
        ax.set_ylabel("Predicted pIC50", fontsize=9)
        ax.set_title(f"{model}  [{fs}]", fontsize=10, fontweight="bold", color=TEXT, pad=6)
        ax.text(0.06, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes,
                color=GREEN, fontsize=10, fontweight="bold", va="top")
        ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT,
                  loc="lower right")
        fig.tight_layout(pad=0.8)
        self._obs_chart_holder.addWidget(canvas)

    def _draw_residuals(self, y_test, y_pred, y_train, y_pred_train):
        self._clear_holder(self._resid_chart_holder)
        fig, canvas = _new_figure(4.5, 3.2)
        self._fig_resid = fig
        ax = fig.add_subplot(111)
        _style_ax(ax)
        resid_te = np.array(y_test, dtype=float) - np.array(y_pred, dtype=float)
        nbins = min(30, max(10, len(resid_te) // 8))
        if y_train and y_pred_train:
            resid_tr = np.array(y_train, dtype=float) - np.array(y_pred_train, dtype=float)
            ax.hist(resid_tr, bins=nbins, color="#94a3b8", alpha=0.45,
                    edgecolor="none", label=f"Train (σ={np.std(resid_tr):.3f})")
        ax.hist(resid_te, bins=nbins, color=BLUE, alpha=0.75,
                edgecolor="none", label=f"Test  (σ={np.std(resid_te):.3f})")
        ax.axvline(0, color=ORANGE, lw=1.5, linestyle="--", alpha=0.9)
        ax.set_xlabel("Residual (Actual − Predicted)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Residuals Distribution", fontsize=10, fontweight="bold", color=TEXT, pad=6)
        ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT)
        fig.tight_layout(pad=0.8)
        self._resid_chart_holder.addWidget(canvas)

    def _draw_yscrambling(self, yscr_vals, original_r2):
        self._clear_holder(self._yscr_chart_holder)
        fig, canvas = _new_figure(4.5, 3.2)
        self._fig_yscr = fig
        ax = fig.add_subplot(111)
        _style_ax(ax)
        vals = np.array(yscr_vals, dtype=float)
        xs = np.arange(len(vals))
        colors = [RED if v > 0 else "#334155" for v in vals]
        ax.bar(xs, vals, color=colors, alpha=0.85, width=0.7, zorder=3)
        ax.axhline(original_r2, color=ORANGE, lw=2, linestyle="--",
                   label=f"Original R² = {original_r2:.3f}", zorder=4)
        ax.axhline(0, color=SUBTEXT, lw=0.8, alpha=0.5, zorder=2)
        ax.set_xlabel("Scrambling Run", fontsize=9)
        ax.set_ylabel("R² (scrambled)", fontsize=9)
        ax.set_title("Y-Scrambling Validation", fontsize=10, fontweight="bold", color=TEXT, pad=6)
        ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper right")
        mean_scr = np.nanmean(vals)
        ax.text(0.06, 0.08,
                f"Mean scrambled R² = {mean_scr:.3f}",
                transform=ax.transAxes, color=SUBTEXT, fontsize=8)
        fig.tight_layout(pad=0.8)
        self._yscr_chart_holder.addWidget(canvas)

    def _draw_dist_dashboard(self, y_test, y_pred, y_train):
        self._clear_holder(self._dash_dist_holder)
        fig, canvas = _new_figure(5, 3)
        self._fig_dist = fig
        ax = fig.add_subplot(111)
        _style_ax(ax)
        if y_train:
            ax.hist(y_train, bins=30, color="#94a3b8", alpha=0.45,
                    edgecolor="none", label="Train")
        ax.hist(y_test, bins=30, color=TEAL, alpha=0.75,
                edgecolor="none", label="Test")
        ax.set_xlabel("pIC50", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("pIC50 Distribution (Train vs Test)", fontsize=10,
                     fontweight="bold", color=TEXT, pad=6)
        ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT)
        fig.tight_layout(pad=0.8)
        self._dash_dist_holder.addWidget(canvas)

    def _draw_ad_chart(self, all_results: dict):
        self._clear_holder(self._ad_chart_holder)
        labels, values, fs_tags = [], [], []
        for fs, mods in all_results.items():
            for mname, met in mods.items():
                if not isinstance(met, dict):
                    continue
                dist = met.get("ad_mean_dist", float("nan"))
                if np.isfinite(dist):
                    labels.append(f"{mname}\n({fs})")
                    values.append(dist)
                    fs_tags.append(fs)

        if not values:
            self._ad_chart_holder.addWidget(_placeholder_canvas("No AD data available"))
            return

        fig, canvas = _new_figure(6, max(3.5, len(values) * 0.35))
        self._fig_ad = fig
        ax = fig.add_subplot(111)
        _style_ax(ax)
        ys = np.arange(len(values))
        cmap_colors = [BLUE, TEAL, PURPLE, GREEN, ORANGE, RED]
        unique_fs = list(dict.fromkeys(fs_tags))
        bar_colors = [cmap_colors[unique_fs.index(t) % len(cmap_colors)] for t in fs_tags]
        bars = ax.barh(ys, values, color=bar_colors, alpha=0.85, height=0.65)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Mean k-NN Distance (AD Score)", fontsize=9)
        ax.set_title("Applicability Domain — k-NN Mean Distance per Model",
                     fontsize=10, fontweight="bold", color=TEXT, pad=6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7, color=TEXT)
        fig.tight_layout(pad=0.8)
        self._ad_chart_holder.addWidget(canvas)

    def _draw_model_comparison(self, all_results: dict, best_fs: str):
        self._clear_holder(self._mdlcmp_chart_holder)
        # Collect metrics from the best feature set only (or all if best_fs missing)
        source = all_results.get(best_fs) or next(iter(all_results.values()), {})
        names, tr_r2s, te_r2s, cv_r2s = [], [], [], []
        for mname, met in source.items():
            if not isinstance(met, dict) or "test" not in met:
                continue
            tr = met["train"].get("r2", float("nan"))
            te = met["test"].get("r2", float("nan"))
            cv = met.get("cv_mean", float("nan"))
            if np.isfinite(te):
                names.append(mname)
                tr_r2s.append(tr); te_r2s.append(te); cv_r2s.append(cv)

        if not names:
            self._mdlcmp_chart_holder.addWidget(_placeholder_canvas("No model data"))
            return

        # sort by test R²
        order = np.argsort(te_r2s)
        names   = [names[i] for i in order]
        tr_r2s  = [tr_r2s[i] for i in order]
        te_r2s  = [te_r2s[i] for i in order]
        cv_r2s  = [cv_r2s[i] for i in order]

        fig, canvas = _new_figure(4.5, max(3.0, len(names) * 0.38))
        self._fig_mdlcmp = fig
        ax = fig.add_subplot(111)
        _style_ax(ax)
        ys = np.arange(len(names))
        h  = 0.25
        ax.barh(ys - h, tr_r2s,  height=h, color="#94a3b8", alpha=0.7, label="Train R²")
        ax.barh(ys,     te_r2s,  height=h, color=BLUE,      alpha=0.85, label="Test R²")
        ax.barh(ys + h, cv_r2s,  height=h, color=GREEN,     alpha=0.75, label="CV R²")
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("R²", fontsize=9)
        ax.set_title(f"Model Comparison  [{best_fs}]", fontsize=10,
                     fontweight="bold", color=TEXT, pad=6)
        ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER,
                  labelcolor=TEXT, loc="lower right")
        fig.tight_layout(pad=0.8)
        self._mdlcmp_chart_holder.addWidget(canvas)

    def _draw_pred_vs_resid(self, y_test, y_pred, y_train, y_pred_train):
        self._clear_holder(self._pvr_chart_holder)
        fig, canvas = _new_figure(4.5, 3.2)
        self._fig_pvr = fig
        ax = fig.add_subplot(111)
        _style_ax(ax)
        if y_train and y_pred_train:
            tr_p = np.array(y_pred_train, dtype=float)
            tr_r = np.array(y_train, dtype=float) - tr_p
            ax.scatter(tr_p, tr_r, s=10, alpha=0.25, c="#94a3b8",
                       edgecolors="none", zorder=2, label="Train")
        te_p = np.array(y_pred, dtype=float)
        te_r = np.array(y_test, dtype=float) - te_p
        ax.scatter(te_p, te_r, s=18, alpha=0.70, c=BLUE,
                   edgecolors="none", zorder=3, label="Test")
        ax.axhline(0, color=ORANGE, lw=1.5, linestyle="--", alpha=0.9)
        ax.set_xlabel("Predicted pIC50", fontsize=9)
        ax.set_ylabel("Residual", fontsize=9)
        ax.set_title("Predicted vs Residuals", fontsize=10,
                     fontweight="bold", color=TEXT, pad=6)
        ax.legend(fontsize=8, facecolor=BG_CARD, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper right")
        fig.tight_layout(pad=0.8)
        self._pvr_chart_holder.addWidget(canvas)

    # ── Publication-quality figure save (white background) ────────────────────
    def _save_all_figures(self, outdir: str) -> list[str]:
        """Re-draw every chart with white background and save to outdir/figures/."""
        from matplotlib.figure import Figure as _Fig

        pd = self._plot_data
        if not pd:
            return []

        fig_dir = os.path.join(outdir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        saved: list[str] = []

        # ── publication palette ───────────────────────────────────────────────
        TR_C   = "#4472C4"   # train – blue
        TE_C   = "#ED7D31"   # test  – orange
        DIAG_C = "#C00000"   # diagonal / zero line
        SCRAM_C= "#4472C4"
        GR_C   = "#70AD47"   # CV / green

        def _wb_ax(ax):
            ax.set_facecolor("white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#aaaaaa")
            ax.tick_params(colors="#333333", labelsize=9)
            ax.xaxis.label.set_color("#333333")
            ax.yaxis.label.set_color("#333333")
            ax.title.set_color("#111111")
            ax.grid(color="#dddddd", linestyle="--", linewidth=0.5, alpha=0.8)

        def _savefig(fig: _Fig, name: str):
            path = os.path.join(fig_dir, name)
            try:
                fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
                saved.append(path)
            except Exception as exc:
                self._append_log(f"  Warning: could not save {name}: {exc}")

        y_test      = pd["y_test"];       y_pred      = pd["y_pred"]
        y_train     = pd["y_train"];      y_pred_train = pd["y_pred_train"]
        best_r2     = pd["best_r2"];      best_model  = pd["best_model"]
        best_fs     = pd["best_fs"];      yscr_viz    = pd["yscr_viz"]
        all_results = pd["all_results"]

        # 1. Observed vs Predicted
        if y_test and y_pred:
            fig = _Fig(figsize=(6, 5), facecolor="white", tight_layout=True)
            ax  = fig.add_subplot(111)
            _wb_ax(ax)
            arr_t = np.array(y_test); arr_p = np.array(y_pred)
            if y_train and y_pred_train:
                ax.scatter(np.array(y_train), np.array(y_pred_train),
                           s=12, alpha=0.25, c=TR_C, edgecolors="none",
                           zorder=2, label="Train")
            ax.scatter(arr_t, arr_p, s=22, alpha=0.80, c=TE_C,
                       edgecolors="none", zorder=3, label="Test")
            all_v = list(arr_t) + list(arr_p)
            mn, mx = min(all_v), max(all_v)
            pad = (mx - mn) * 0.05
            ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "--",
                    color=DIAG_C, lw=1.5, alpha=0.85)
            ax.set_xlabel("Observed pIC50", fontsize=10)
            ax.set_ylabel("Predicted pIC50", fontsize=10)
            ax.set_title(f"Observed vs Predicted — {best_model} [{best_fs}]",
                         fontsize=11, fontweight="bold")
            ax.text(0.06, 0.94, f"R² = {best_r2:.3f}",
                    transform=ax.transAxes, fontsize=11,
                    fontweight="bold", color="#16a34a", va="top")
            ax.legend(fontsize=9)
            _savefig(fig, "obs_vs_predicted.png")

        # 2. Residuals distribution
        if y_test and y_pred:
            fig = _Fig(figsize=(6, 4.5), facecolor="white", tight_layout=True)
            ax  = fig.add_subplot(111)
            _wb_ax(ax)
            resid_te = np.array(y_test) - np.array(y_pred)
            nbins = min(30, max(10, len(resid_te) // 8))
            if y_train and y_pred_train:
                resid_tr = np.array(y_train) - np.array(y_pred_train)
                ax.hist(resid_tr, bins=nbins, color=TR_C, alpha=0.45,
                        edgecolor="none",
                        label=f"Train  (σ={np.std(resid_tr):.3f})")
            ax.hist(resid_te, bins=nbins, color=TE_C, alpha=0.80,
                    edgecolor="none",
                    label=f"Test  (σ={np.std(resid_te):.3f})")
            ax.axvline(0, color=DIAG_C, lw=1.5, linestyle="--", alpha=0.85)
            ax.set_xlabel("Residual (Actual − Predicted)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title("Residuals Distribution", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            _savefig(fig, "residuals_distribution.png")

        # 3. Predicted vs Residuals
        if y_test and y_pred:
            fig = _Fig(figsize=(6, 4.5), facecolor="white", tight_layout=True)
            ax  = fig.add_subplot(111)
            _wb_ax(ax)
            te_p = np.array(y_pred)
            te_r = np.array(y_test) - te_p
            if y_train and y_pred_train:
                tr_p = np.array(y_pred_train)
                tr_r = np.array(y_train) - tr_p
                ax.scatter(tr_p, tr_r, s=12, alpha=0.25, c=TR_C,
                           edgecolors="none", zorder=2, label="Train")
            ax.scatter(te_p, te_r, s=22, alpha=0.80, c=TE_C,
                       edgecolors="none", zorder=3, label="Test")
            ax.axhline(0, color=DIAG_C, lw=1.5, linestyle="--", alpha=0.85)
            ax.set_xlabel("Predicted pIC50", fontsize=10)
            ax.set_ylabel("Residual", fontsize=10)
            ax.set_title("Predicted vs Residuals", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            _savefig(fig, "pred_vs_residuals.png")

        # 4. pIC50 distribution (train vs test)
        if y_test:
            fig = _Fig(figsize=(6, 4.5), facecolor="white", tight_layout=True)
            ax  = fig.add_subplot(111)
            _wb_ax(ax)
            if y_train:
                ax.hist(np.array(y_train), bins=30, color=TR_C, alpha=0.45,
                        edgecolor="none", label="Train")
            ax.hist(np.array(y_test), bins=30, color=TE_C, alpha=0.80,
                    edgecolor="none", label="Test")
            ax.set_xlabel("pIC50", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title("pIC50 Distribution (Train vs Test)",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            _savefig(fig, "pic50_distribution.png")

        # 5. Y-scrambling
        if yscr_viz:
            fig = _Fig(figsize=(6, 4.5), facecolor="white", tight_layout=True)
            ax  = fig.add_subplot(111)
            _wb_ax(ax)
            vals = np.array(yscr_viz, dtype=float)
            xs   = np.arange(len(vals))
            ax.bar(xs, vals, color=SCRAM_C, alpha=0.75, width=0.7)
            ax.axhline(best_r2, color=DIAG_C, lw=2, linestyle="--",
                       label=f"Original R² = {best_r2:.3f}")
            ax.axhline(0, color="#aaaaaa", lw=0.8)
            ax.set_xlabel("Scrambling Run", fontsize=10)
            ax.set_ylabel("Scrambled R²", fontsize=10)
            ax.set_title("Y-Scrambling Validation", fontsize=11, fontweight="bold")
            ax.text(0.06, 0.06,
                    f"Mean scrambled R² = {np.nanmean(vals):.3f}",
                    transform=ax.transAxes, fontsize=9, color="#555555")
            ax.legend(fontsize=9)
            _savefig(fig, "y_scrambling.png")

        # 6. Model comparison
        source = all_results.get(best_fs) or next(iter(all_results.values()), {})
        names, tr_r2s, te_r2s, cv_r2s = [], [], [], []
        for mname, met in source.items():
            if not isinstance(met, dict) or "test" not in met:
                continue
            tr = met["train"].get("r2", float("nan"))
            te = met["test"].get("r2", float("nan"))
            cv = met.get("cv_mean", float("nan"))
            if np.isfinite(te):
                names.append(mname); tr_r2s.append(tr)
                te_r2s.append(te);   cv_r2s.append(cv)
        if names:
            order   = np.argsort(te_r2s)
            names   = [names[i] for i in order]
            tr_r2s  = [tr_r2s[i] for i in order]
            te_r2s  = [te_r2s[i] for i in order]
            cv_r2s  = [cv_r2s[i] for i in order]
            fig = _Fig(figsize=(7, max(4, len(names) * 0.42)),
                       facecolor="white", tight_layout=True)
            ax  = fig.add_subplot(111)
            _wb_ax(ax)
            ys = np.arange(len(names)); h = 0.25
            ax.barh(ys - h, tr_r2s,  height=h, color=TR_C,  alpha=0.65, label="Train R²")
            ax.barh(ys,     te_r2s,  height=h, color=TE_C,  alpha=0.85, label="Test R²")
            ax.barh(ys + h, cv_r2s,  height=h, color=GR_C,  alpha=0.75, label="CV R²")
            ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel("R²", fontsize=10)
            ax.set_title(f"Model Comparison  [{best_fs}]",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=9, loc="lower right")
            _savefig(fig, "model_comparison.png")

        # 7. Applicability Domain
        labels, values, fs_tags = [], [], []
        for fs, mods in all_results.items():
            for mname, met in mods.items():
                if not isinstance(met, dict):
                    continue
                dist = met.get("ad_mean_dist", float("nan"))
                if np.isfinite(dist):
                    labels.append(f"{mname} ({fs})")
                    values.append(dist)
                    fs_tags.append(fs)
        if values:
            order2 = np.argsort(values)
            labels = [labels[i] for i in order2]
            values = [values[i] for i in order2]
            fig = _Fig(figsize=(7, max(4, len(values) * 0.38)),
                       facecolor="white", tight_layout=True)
            ax  = fig.add_subplot(111)
            _wb_ax(ax)
            ys2 = np.arange(len(values))
            ax.barh(ys2, values, color=TR_C, alpha=0.80, height=0.65)
            ax.set_yticks(ys2); ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Mean k-NN Distance (AD Score)", fontsize=10)
            ax.set_title("Applicability Domain — k-NN Mean Distance per Model",
                         fontsize=11, fontweight="bold")
            for i, val in enumerate(values):
                ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=8)
            _savefig(fig, "applicability_domain.png")

        return saved

    # ── Prediction logic ───────────────────────────────────────────────────────
    def _pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select best_model.pkl",
                                              filter="Pickle (*.pkl)")
        if path:
            self.model_path.setText(path)

    def _pick_meta(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select best_model_meta.json",
                                              filter="JSON (*.json)")
        if path:
            self.meta_path.setText(path)

    def _autoload_model(self):
        outdir = self.dir_edit.text().strip()
        if not outdir:
            QMessageBox.warning(self, "No project dir",
                                "Set the output directory in Data Acquisition first.")
            return
        pkl  = os.path.join(outdir, "best_model.pkl")
        meta = os.path.join(outdir, "best_model_meta.json")
        if os.path.exists(pkl):
            self.model_path.setText(pkl)
        if os.path.exists(meta):
            self.meta_path.setText(meta)

    def _pick_smiles_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SMILES file",
                                              filter="Text/CSV (*.txt *.csv)")
        if not path:
            return
        import pandas as pd
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
                col = next((c for c in df.columns if c.lower() == "smiles"), None)
                if col is None:
                    QMessageBox.critical(self, "Error",
                                         "CSV must contain a 'smiles' or 'SMILES' column.")
                    return
                smiles = df[col].astype(str).tolist()
            else:
                with open(path, encoding="utf-8") as fh:
                    smiles = [l.strip() for l in fh if l.strip()]
            self.smiles_text.setPlainText("\n".join(smiles))
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    def _predict(self):
        model_path = self.model_path.text().strip()
        meta_path  = self.meta_path.text().strip()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.critical(self, "Missing model",
                                  "Please select a valid best_model.pkl.")
            return
        if not meta_path or not os.path.exists(meta_path):
            QMessageBox.critical(self, "Missing metadata",
                                  "Please select a valid best_model_meta.json.")
            return
        smiles = [s.strip() for s in self.smiles_text.toPlainText().splitlines() if s.strip()]
        if not smiles:
            QMessageBox.critical(self, "No input", "Paste at least one SMILES string.")
            return
        try:
            df = predict_from_smiles(model_path, meta_path, smiles)
            out_dir  = os.path.dirname(model_path)
            out_csv  = os.path.join(out_dir, "predictions.csv")
            df.to_csv(out_csv, index=False)
            self.pred_out.setPlainText(
                df.head(40).to_string(index=False) + f"\n\nSaved → {out_csv}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Prediction failed", str(exc))
