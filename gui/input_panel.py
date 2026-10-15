"""
gui/input_panel.py
==================
Panel KIRI: form input, tombol [Hitung Akar] & [Reset], dan petunjuk penggunaan.

Aturan tombol:
  - [Hitung Akar]  aktif hanya bila SEMUA isian terisi.
  - [Reset]        aktif bila ada isian atau ada hasil yang bisa dihapus.
  Tombol non-aktif tampil dengan opacity 25% (diatur di theme.py).
"""

from __future__ import annotations

from typing import Dict

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

# (kunci, label, placeholder, contoh)
FIELDS = [
    ("function", "Fungsi F(X)", "Masukkan Fungsi F(X)", "Contoh: x**2 - 2x + 1"),
    ("a", "Batas Bawah (a)", "Masukkan Nilai Batas Bawah", "Contoh: 2"),
    ("b", "Batas Atas (b)", "Masukkan Nilai Batas Atas", "Contoh: 5"),
    ("eps", "Nilai Galat (ε)", "Masukkan Nilai Galat", "Contoh: 0,0001"),
    ("n", "Iterasi Maksimum (N)", "Masukkan Iterasi Maksimum", "Contoh: 20 (Maks : 100)"),
    ("decimals", "Jumlah Angka Desimal", "Masukkan Jumlah Angka Desimal", "Contoh: 4 (Maks : 4)"),
]

GUIDE = [
    "· Gunakan x untuk variabel dan (*) sebagai perkalian",
    "· Gunakan / untuk pembagian dan (**) untuk pangkat",
    "· Pemisah desimal boleh koma (,) atau titik (.)",
]


class InputPanel(QFrame):
    calculate_requested = pyqtSignal()   # tombol Hitung Akar ditekan
    reset_requested = pyqtSignal()       # tombol Reset ditekan

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setFixedWidth(292)
        self._has_result = False
        self.edits: Dict[str, QLineEdit] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 24, 22, 22)
        layout.setSpacing(2)

        title = QLabel("PANEL INPUT")
        title.setObjectName("cardTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(10)

        # --- field input ---
        for key, label, placeholder, example in FIELDS:
            lab = QLabel(label)
            lab.setObjectName("fieldLabel")
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.textChanged.connect(self.update_buttons)
            edit.returnPressed.connect(self._on_enter)
            hint = QLabel(example)
            hint.setObjectName("hint")
            layout.addWidget(lab)
            layout.addWidget(edit)
            layout.addWidget(hint)
            self.edits[key] = edit

        # --- tombol aksi (jajar horizontal) ---
        layout.addSpacing(10)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.btn_calc = QPushButton("Hitung Akar")
        self.btn_reset = QPushButton("Reset")
        for btn in (self.btn_calc, self.btn_reset):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(btn)
        self.btn_calc.clicked.connect(self.calculate_requested.emit)
        self.btn_reset.clicked.connect(self.reset_requested.emit)
        layout.addLayout(row)

        # --- petunjuk penggunaan ---
        layout.addSpacing(14)
        guide_title = QLabel("Cara Penggunaan:")
        guide_title.setObjectName("guideTitle")
        layout.addWidget(guide_title)
        for line in GUIDE:
            g = QLabel(line)
            g.setObjectName("guide")
            g.setWordWrap(True)
            layout.addWidget(g)

        self.update_buttons()

    # ------------------------------------------------------------------ API
    def values(self) -> Dict[str, str]:
        """Isi semua field (kunci sama dengan argumen parse_inputs)."""
        return {key: edit.text() for key, edit in self.edits.items()}

    def is_complete(self) -> bool:
        return all(edit.text().strip() for edit in self.edits.values())

    def has_text(self) -> bool:
        return any(edit.text().strip() for edit in self.edits.values())

    def set_has_result(self, value: bool) -> None:
        """Diberi tahu oleh jendela utama bila hasil sedang ditampilkan."""
        self._has_result = value
        self.update_buttons()

    def clear(self) -> None:
        """Kosongkan seluruh textfield."""
        for edit in self.edits.values():
            edit.blockSignals(True)
            edit.clear()
            edit.blockSignals(False)
        self.update_buttons()
        self.edits["function"].setFocus()

    def update_buttons(self) -> None:
        self.btn_calc.setEnabled(self.is_complete())
        self.btn_reset.setEnabled(self.has_text() or self._has_result)

    def apply_theme(self, theme) -> None:
        """Placeholder memakai warna dengan opacity 25% (lewat QPalette)."""
        for edit in self.edits.values():
            palette = edit.palette()
            palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(*theme.placeholder_rgba))
            edit.setPalette(palette)

    # ------------------------------------------------------------- internal
    def _on_enter(self) -> None:
        if self.btn_calc.isEnabled():
            self.calculate_requested.emit()
