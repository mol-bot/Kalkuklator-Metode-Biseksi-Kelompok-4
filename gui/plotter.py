"""
gui/plotter.py
==============
Logika penggambaran grafik fungsi dengan Matplotlib.

Sengaja dipisah dari PyQt (hanya memakai objek Figure) agar:
  - graph_panel.py cukup menanamkan Figure ke canvas Qt, dan
  - kode penggambaran bisa diuji tanpa GUI.

Fitur interaktif: scroll mouse = zoom, drag = geser, klik ganda = reset tampilan.

[PERUBAHAN 3] Kurva tidak lagi "terpotong": setiap kali tampilan berubah
(zoom / geser / reset) kurva dihitung ulang untuk rentang x yang sedang terlihat,
sehingga grafik selalu tersambung sampai x -> tak hingga.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle          # [PERUBAHAN 3] bingkai area plot
from matplotlib.ticker import FuncFormatter

from backend.bisection_solver import ParsedFunction, format_number

# Tampilan awal (grid kosong) sesuai desain
EMPTY_XLIM = (-15.0, 15.0)
EMPTY_YLIM = (-20.0, 20.0)

CURVE_SAMPLES = 1500     # [PERUBAHAN 3] jumlah titik sampel kurva pada rentang x yang terlihat


def _comma_tick(value: float, _pos) -> str:
    """Label sumbu dengan koma sebagai pemisah desimal (mis. 2,5)."""
    text = f"{value:.6g}"
    if text in ("-0", "0"):
        text = "0"
    return text.replace(".", ",")


class BisectionPlotter:
    def __init__(self, figure: Figure, theme) -> None:
        self.fig = figure
        self.theme = theme
        self.ax = self.fig.add_subplot(111)
        self._home: Tuple[Tuple[float, float], Tuple[float, float]] = (EMPTY_XLIM, EMPTY_YLIM)
        self._drag: Optional[dict] = None
        self._state: Optional[dict] = None      # data hasil terakhir (untuk ganti tema)
        self._fn = None                          # [PERUBAHAN 3] fungsi vektor f(x) yang sedang digambar
        self._line: Optional[Line2D] = None      # [PERUBAHAN 3] objek garis kurva

        # [PERUBAHAN 3] area plot dibuat hampir memenuhi canvas (margin dikecilkan)
        self.fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.14)
        canvas = self.fig.canvas
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)
        self.clear()

    # ------------------------------------------------------------------ API
    def set_theme(self, theme) -> None:
        """Ganti tema lalu gambar ulang kondisi terakhir (posisi zoom/geser dipertahankan)."""
        self.theme = theme
        if self._state is None:
            self.clear()
            return
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.plot(**self._state)
        self.ax.set_xlim(*xlim)                  # [PERUBAHAN 3] jangan reset zoom saat ganti tema
        self.ax.set_ylim(*ylim)
        self._finish()

    def clear(self) -> None:
        """Kembali ke grid kosong awal."""
        self._state = None
        self._fn = None                          # [PERUBAHAN 3]
        self._line = None                        # [PERUBAHAN 3]
        self._reset_axes()
        self.ax.set_xlim(*EMPTY_XLIM)
        self.ax.set_ylim(*EMPTY_YLIM)
        self._home = (EMPTY_XLIM, EMPTY_YLIM)
        self._finish()

    def plot(self, function: ParsedFunction, a: float, b: float,
             root: float, f_root: float, decimals: int) -> None:
        """Gambar kurva f(x), interval [a, b], dan titik akar."""
        t = self.theme
        self._state = dict(function=function, a=a, b=b, root=root,
                           f_root=f_root, decimals=decimals)
        self._reset_axes()

        # ---- rentang x awal: sedikit melebar dari [a, b], usahakan memuat x = 0 ----
        span = b - a
        x_lo, x_hi = a - 0.5 * span, b + 0.5 * span
        if 0 < x_lo < 2 * span:
            x_lo = -0.15 * span
        if x_hi < 0 and -x_hi < 2 * span:
            x_hi = 0.15 * span

        # ---- rentang y awal: berdasarkan nilai di sekitar interval [a, b] ----
        # (abaikan nilai ekstrem, mis. asimtot, dengan persentil)
        xs = np.linspace(x_lo, x_hi, 900)
        ys = function.vector(xs)
        inside = ys[(xs >= a) & (xs <= b) & np.isfinite(ys)]
        finite = inside if inside.size else ys[np.isfinite(ys)]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
        else:
            lo, hi = -1.0, 1.0
        lo, hi = min(lo, 0.0, f_root), max(hi, 0.0, f_root)
        pad = 0.30 * ((hi - lo) or 1.0)
        y_lo, y_hi = lo - pad, hi + pad

        # ---- kurva (data diisi oleh _update_curve) & titik akar ----
        self._fn = function.vector                                         # [PERUBAHAN 3]
        (self._line,) = self.ax.plot([], [], color=t.plot_curve,           # [PERUBAHAN 3]
                                     linewidth=2.2, zorder=3)
        self.ax.plot([root], [f_root], "o", color=t.plot_root, markersize=9, zorder=6)
        self.ax.annotate(
            f"({format_number(root, decimals)}; {format_number(f_root, decimals)})",
            xy=(root, f_root), xytext=(10, 14), textcoords="offset points",
            color=t.plot_label, fontsize=9, zorder=7,
        )

        self.ax.set_xlim(x_lo, x_hi)
        self.ax.set_ylim(y_lo, y_hi)
        self._home = ((x_lo, x_hi), (y_lo, y_hi))

        # ---- legend di bawah plot ----
        handles = [
            Line2D([0], [0], color=t.plot_curve, linewidth=2.2),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=t.plot_root,
                   markeredgecolor=t.plot_root, markersize=9),
        ]
        labels = [f"f(x) = {function.display}", f"Akar = {format_number(root, decimals)}"]
        legend = self.fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0.0),
                                 frameon=False, fontsize=9, labelcolor=t.plot_label)
        for text in legend.get_texts():
            text.set_fontweight("bold")
        self._finish()

    # ------------------------------------------------------------- internal
    def _reset_axes(self) -> None:
        """Bersihkan axes + legend lalu terapkan gaya sesuai tema."""
        t = self.theme
        for leg in list(self.fig.legends):
            leg.remove()
        self.ax.clear()
        self.fig.patch.set_facecolor(t.plot_bg)
        self.ax.set_facecolor(t.plot_bg)

        self.ax.minorticks_on()
        self.ax.set_axisbelow(True)
        self.ax.xaxis.set_major_formatter(FuncFormatter(_comma_tick))
        self.ax.yaxis.set_major_formatter(FuncFormatter(_comma_tick))
        self.ax.grid(which="major", color=t.plot_axis, alpha=0.55, linewidth=0.8)
        self.ax.grid(which="minor", color=t.plot_axis, alpha=0.20, linewidth=0.5)
        self.ax.tick_params(which="major", colors=t.plot_axis, labelcolor=t.plot_label,
                            labelsize=8, length=3)
        self.ax.tick_params(which="minor", length=0)
        for side in ("top", "right"):
            self.ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            self.ax.spines[side].set_color(t.plot_axis)
            self.ax.spines[side].set_linewidth(1.3)

        # [PERUBAHAN 3] bingkai tipis di sekeliling area plot, supaya jelas bahwa
        # kurva hanya "terpotong" oleh jendela tampilan (kurva aslinya terus berlanjut)
        self.ax.add_patch(Rectangle((0, 0), 1, 1, transform=self.ax.transAxes, fill=False,
                                    edgecolor=t.plot_axis, linewidth=1.0, zorder=20,
                                    clip_on=False))

    def _place_axes(self) -> None:
        """Sumbu melewati titik (0,0) bila terlihat; jika tidak, menempel di tepi."""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.spines["bottom"].set_position(("data", 0) if y0 <= 0 <= y1 else ("axes", 0))
        self.ax.spines["left"].set_position(("data", 0) if x0 <= 0 <= x1 else ("axes", 0))

    def _update_curve(self) -> None:
        """
        [PERUBAHAN 3] Hitung ulang kurva untuk rentang x yang SEDANG terlihat
        (ditambah sedikit margin). Dipanggil setiap tampilan berubah, sehingga
        kurva tidak pernah berhenti di batas rentang lama -> seolah sampai tak hingga.
        """
        if self._line is None or self._fn is None:
            return
        x0, x1 = self.ax.get_xlim()
        pad = 0.05 * (x1 - x0)
        xs = np.linspace(x0 - pad, x1 + pad, CURVE_SAMPLES)
        ys = np.array(self._fn(xs), dtype=float)

        # putus garis di asimtot (mis. tan x): lompatan antar titik > 3x tinggi tampilan
        y0, y1 = self.ax.get_ylim()
        with np.errstate(invalid="ignore"):
            jump = np.abs(np.diff(ys)) > 3 * ((y1 - y0) or 1.0)
        ys[:-1][jump] = np.nan
        self._line.set_data(xs, ys)

    def _finish(self) -> None:
        self._place_axes()
        self._update_curve()                     # [PERUBAHAN 3]
        self.fig.canvas.draw_idle()

    # ------------------------------------------------ interaksi mouse (zoom/pan)
    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(event.xdata - (event.xdata - x0) * scale,
                         event.xdata + (x1 - event.xdata) * scale)
        self.ax.set_ylim(event.ydata - (event.ydata - y0) * scale,
                         event.ydata + (y1 - event.ydata) * scale)
        self._finish()

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1:
            return
        if getattr(event, "dblclick", False):               # klik ganda = reset tampilan
            self.ax.set_xlim(*self._home[0])
            self.ax.set_ylim(*self._home[1])
            self._finish()
            return
        self._drag = dict(px=event.x, py=event.y,
                          xlim=self.ax.get_xlim(), ylim=self.ax.get_ylim())

    def _on_motion(self, event) -> None:
        if self._drag is None or event.x is None:
            return
        bbox = self.ax.bbox
        x0, x1 = self._drag["xlim"]
        y0, y1 = self._drag["ylim"]
        dx = (event.x - self._drag["px"]) * (x1 - x0) / bbox.width
        dy = (event.y - self._drag["py"]) * (y1 - y0) / bbox.height
        self.ax.set_xlim(x0 - dx, x1 - dx)
        self.ax.set_ylim(y0 - dy, y1 - dy)
        self._finish()

    def _on_release(self, event) -> None:
        self._drag = None
