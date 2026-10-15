"""
backend/bisection_solver.py
===========================
Logika bisnis (backend) Metode Biseksi.

Modul ini TIDAK bergantung pada GUI sama sekali, sehingga bisa diuji / dipakai
ulang secara terpisah. Isinya:
  1. Parser fungsi f(x) berbasis SymPy (dengan whitelist karakter & nama).
  2. Validasi seluruh input pengguna.
  3. Algoritma biseksi (mengikuti slide algoritma pada materi).
  4. Pemformatan angka (koma desimal + notasi ilmiah otomatis).
  5. Penyusun teks ringkasan / kesimpulan.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# --------------------------------------------------------------------------
# Konstanta batas input
# --------------------------------------------------------------------------
MAX_ITERATIONS = 100   # iterasi maksimum yang boleh kita masokkan
MAX_DECIMALS = 4       # jumlah angka desimal maksimum yang boleh tarok


class InputError(ValueError):
    """Dilempar bila input pengguna / evaluasi fungsi tidak valid."""


# ==========================================================================
# 1. PARSER FUNGSI f(x)
# ==========================================================================
_X = sp.Symbol("x")

# Hanya karakter berikut yang boleh muncul di teks fungsi (setelah dinormalkan).
_ALLOWED_CHARS = re.compile(r"^[0-9a-z+\-*/^().,\s]+$")

# Namespace terbatas untuk parse_expr (tanpa __builtins__ => lebih aman).
_GLOBALS = {
    "__builtins__": {},
    # kelas-kelas yang dibutuhkan transformasi SymPy
    "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
    "Symbol": sp.Symbol, "Function": sp.Function,
    # fungsi matematika yang diizinkan
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "log10": lambda v: sp.log(v, 10),
    "sqrt": sp.sqrt, "abs": sp.Abs,
}
_LOCALS = {"x": _X, "e": sp.E, "pi": sp.pi}

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


# --------------------------------------------------------------------------
# Pangkat dinampakkan sebagai superscript anjay (tanpa tanda ^ / **)
# --------------------------------------------------------------------------
_SUP_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_TO_SUP = str.maketrans("0123456789-", _SUP_DIGITS + "⁻")
_FROM_SUP = str.maketrans(_SUP_DIGITS + "⁻", "0123456789-")
_SUP_RUN = f"⁻?[{_SUP_DIGITS}]+"          # deretan karakter superscript, angka pangkat tu
_FRACTION_AHEAD = r"[.,](?:\d|$)"          # tanda desimal setelah angka (pangkat pecahan, kek 0.5)


def superscript_exponents(text: str) -> str:
    """
    Ubah pangkat bilangan bulat menjadi superscript:
        "x**2 - 5x + 4" -> "x² - 5x + 4"      "x^10" -> "x¹⁰"      "x**-2" -> "x⁻²"
    Dipakai untuk kolom input (diketik langsung berubah) DAN semua tampilan fungsi.

    Aturan khusus:
      - Pangkat pecahan/desimal ("x**0.5", "x**(1/2)") dibiarkan apa adanya, sebab
        tidak ada simbol superscript untuk titik/koma/garis miring.
      - Digit yang diketik tepat setelah superscript digabung ("x¹" + "0" -> "x¹⁰").
      - Fungsi ini aman dipanggil berulang kali (hasilnya stabil).
    """
    # 1) pengguna mengetik "." / "," tepat setelah superscript -> kembalikan ke bentuk ketikan
    text = re.sub(rf"({_SUP_RUN})(?={_FRACTION_AHEAD})",
                  lambda m: "**" + m.group(1).translate(_FROM_SUP), text)
    # 2) digit baru setelah superscript digabung menjadi satu pangkat
    text = re.sub(rf"({_SUP_RUN})(\d+)(?!\d)(?!{_FRACTION_AHEAD})",
                  lambda m: (m.group(1) + m.group(2)).translate(_TO_SUP), text)
    # 3) "**n" atau "^n" (n bulat, boleh negatif) -> superscript
    text = re.sub(rf"\s*(?:\*\*|\^)\s*(-?\d+)(?!\d)(?!{_FRACTION_AHEAD})",
                  lambda m: m.group(1).translate(_TO_SUP), text)
    return text


def _superscripts_to_python(text: str) -> str:
    """Kebalikannya, untuk parser: "x²" -> "x**2", "x⁻²" -> "x**(-2)"."""
    def repl(m):
        digits = m.group(0).translate(_FROM_SUP)
        return f"**({digits})" if digits.startswith("-") else f"**{digits}"
    return re.sub(_SUP_RUN, repl, text)


def _normalize_function_text(text: str) -> str:
    """Rapikan teks fungsi: huruf kecil, simbol unicode, koma desimal."""
    s = text.strip().lower()
    s = _superscripts_to_python(s)                    
    for old, new in {"−": "-", "–": "-", "×": "*", "÷": "/"}.items():
        s = s.replace(old, new)
    # "0,5x" -> "0.5x" (koma di antara dua digit dianggap koma desimal)
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    return s


@dataclass
class ParsedFunction:
    """Hasil parsing fungsi: ekspresi SymPy + fungsi skalar + fungsi vektor."""
    expr: sp.Expr
    scalar: Callable[[float], float]                # untuk perhitungan bisangatseksi
    vector: Callable[[np.ndarray], np.ndarray]      # untuk menggambar grafik gacor
    display: str                                    # teks fungsi untuk legends never die


def parse_function(text: str) -> ParsedFunction:
    """Ubah teks seperti 'x**2 - 2x + 1' menjadi objek ParsedFunction."""
    if not text or not text.strip():
        raise InputError("Fungsi F(X) belum diisi.")

    s = _normalize_function_text(text)
    hint = "Contoh penulisan yang benar: x**2 - 5x + 4  atau  sin(x) - x/2"   
    if not _ALLOWED_CHARS.match(s):
        raise InputError("Fungsi mengandung karakter yang tidak diizinkan.\n" + hint)

    try:
        expr = parse_expr(s, local_dict=dict(_LOCALS), global_dict=dict(_GLOBALS),
                          transformations=_TRANSFORMS)
    except Exception:
        raise InputError("Penulisan fungsi tidak valid.\n" + hint)

    if not isinstance(expr, sp.Expr):
        raise InputError("Fungsi tidak valid.\n" + hint)
    if not expr.free_symbols <= {_X}:
        raise InputError("Fungsi hanya boleh memakai variabel x.\n" + hint)

    raw_scalar = sp.lambdify(_X, expr, modules="math")
    raw_vector = sp.lambdify(_X, expr, modules="numpy")

    def scalar(x: float) -> float:
        try:
            val = float(raw_scalar(x))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            raise InputError(f"f(x) tidak terdefinisi (bukan bilangan real) di x = {x:g}.")
        if not math.isfinite(val):
            raise InputError(f"f(x) bernilai tak hingga / tidak terdefinisi di x = {x:g}.")
        return val

    def vector(xs: np.ndarray) -> np.ndarray:
        # Untuk grafik: titik yang tidak terdefinisi menjadi NaN (kurva terputus).
        with np.errstate(all="ignore"):
            try:
                ys = np.asarray(raw_vector(xs), dtype=float)
                if ys.ndim == 0:
                    ys = np.full_like(xs, float(ys), dtype=float)
            except Exception:
                ys = np.full_like(xs, np.nan, dtype=float)
                for i, xv in enumerate(xs):
                    try:
                        ys[i] = scalar(float(xv))
                    except InputError:
                        pass
        ys[~np.isfinite(ys)] = np.nan
        return ys

    # teks legend = teks yang klen ketik (urutan suku tidak
    # ditukar seperti hasil SymPy), dengan pangkat ditulis superscript tanpa "^".
    display = superscript_exponents(text.strip())
    return ParsedFunction(expr, scalar, vector, display)


# ==========================================================================
# 2. VALIDASI INPUT
# ==========================================================================
def parse_number(text: str, label: str) -> float:
    """Ubah teks ('0,0001', '2', '1e-4') menjadi float; koma dianggap desimal."""
    t = (text or "").strip().replace(",", ".")
    if not t:
        raise InputError(f"{label} belum diisi.")
    try:
        val = float(t)
    except ValueError:
        raise InputError(f"{label} harus berupa angka.")
    if not math.isfinite(val):
        raise InputError(f"{label} harus berupa angka yang valid.")
    return val


def parse_integer(text: str, label: str) -> int:
    val = parse_number(text, label)
    if val != int(val):
        raise InputError(f"{label} harus berupa bilangan bulat.")
    return int(val)


@dataclass
class SolverInputs:
    """Semua input yang sudah tervalidasi."""
    function: ParsedFunction
    a: float
    b: float
    eps: float
    n_max: int
    decimals: int


def parse_inputs(function: str, a: str, b: str, eps: str, n: str, decimals: str) -> SolverInputs:
    """Validasi seluruh isian form; lempar InputError bila ada yang salah."""
    func = parse_function(function)
    a_val = parse_number(a, "Batas Bawah (a)")
    b_val = parse_number(b, "Batas Atas (b)")
    eps_val = parse_number(eps, "Nilai Galat (ε)")
    n_val = parse_integer(n, "Iterasi Maksimum (N)")
    dec_val = parse_integer(decimals, "Jumlah Angka Desimal")

    if a_val >= b_val:
        raise InputError("Batas Bawah (a) harus lebih kecil dari Batas Atas (b).")
    if eps_val <= 0:
        raise InputError("Nilai Galat (ε) harus lebih besar dari 0.")
    if not 1 <= n_val <= MAX_ITERATIONS:
        raise InputError(f"Iterasi Maksimum (N) harus antara 1 dan {MAX_ITERATIONS}.")
    if not 1 <= dec_val <= MAX_DECIMALS:
        raise InputError(f"Jumlah Angka Desimal harus antara 1 dan {MAX_DECIMALS}.")
    return SolverInputs(func, a_val, b_val, eps_val, n_val, dec_val)


# ==========================================================================
# 3. ALGORITMA BISEKSI
# ==========================================================================
@dataclass
class IterationRow:
    r: int          # nomor iterasi
    a: float
    b: float
    fa: float
    fb: float
    xr: float       # titik tengah
    fxr: float      # f(xr)  (tidak dinampakin di tabel, dipake untuk hasil & grafik)
    width: float    # |b - a|
    valid: bool     # True kalo galat terpenuhi (|b - a| < eps) atau f(xr) = 0


@dataclass
class BisectionResult:
    inputs: SolverInputs
    rows: List[IterationRow]            # SEMUA iterasi, r = 1 .. N
    first_true: Optional[int]           # indeks (0-based) baris TRUE pertama, None bila tak ada
    root: float                         # akar solusi (xr pada baris solusi)
    f_root: float
    root_iteration: int                 # nomor iterasi baris solusi
    converged: bool                     # True bila TRUE ditemukan dalam N iterasi
    exact_root: bool                    # True bila f(xr) = 0 tepat

    @property
    def solution_index(self) -> int:
        """Indeks baris solusi: baris TRUE pertama, atau baris terakhir (N)."""
        return self.first_true if self.first_true is not None else len(self.rows) - 1


def solve(inputs: SolverInputs) -> BisectionResult:
    """
    Jalankan metode biseksi.

    Algoritma (sesuai slide):
      - f(a)·f(b) > 0  -> interval belum mengandung akar (error).
      - xr = (a + b) / 2
      - f(xr)·f(a) < 0 -> akar di kiri  (b = xr), selain itu a = xr.
      - validasi = TRUE bila |b - a| < galat.

    Perhitungan SELALU dijalankan sampai iterasi maksimum (N) supaya tombol
    "Tampilkan Semua" dapat menampilkan seluruh tabel; baris solusi adalah
    baris TRUE yang pertama.
    """
    f = inputs.function.scalar
    a, b, eps, n_max = inputs.a, inputs.b, inputs.eps, inputs.n_max

    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise InputError(
            "f(a)·f(b) > 0, sehingga interval [a, b] belum menunjukkan adanya akar.\n"
            "Silakan ganti nilai Batas Bawah (a) dan Batas Atas (b)."
        )

    # Kasus khusus: salah satu batas sudah akar eksak -> interval dipersempit
    # menjadi satu titik supaya iterasi tidak "menjauh" dari akar.
    if fa == 0:
        b, fb = a, fa
    elif fb == 0:
        a, fa = b, fb

    rows: List[IterationRow] = []
    first_true: Optional[int] = None
    exact = False

    for r in range(1, n_max + 1):
        xr = (a + b) / 2
        fxr = f(xr)
        width = abs(b - a)
        valid = (width < eps) or (fxr == 0)

        rows.append(IterationRow(r, a, b, fa, fb, xr, fxr, width, valid))
        if valid and first_true is None:
            first_true = len(rows) - 1
            exact = fxr == 0

        # --- pilih sub-interval yang masih mengandung akar ---
        if fxr == 0:
            a = b = xr                      # akar eksak: bekukan interval
            fa = fb = 0.0
        elif fxr * fa < 0:
            b, fb = xr, fxr                 # akar di sebelah kiri
        else:
            a, fa = xr, fxr                 # akar di sebelah kanan

    sol = rows[first_true] if first_true is not None else rows[-1]
    return BisectionResult(
        inputs=inputs, rows=rows, first_true=first_true,
        root=sol.xr, f_root=sol.fxr, root_iteration=sol.r,
        converged=first_true is not None, exact_root=exact,
    )


# ==========================================================================
# 4. PEMFORMATAN ANGKA
# ==========================================================================
def format_number(value: Optional[float], decimals: int) -> str:
    """
    Format angka sesuai jumlah desimal, dengan koma sebagai pemisah desimal.

    Jika nilai bukan nol tetapi akan tampil sebagai 0,0000 (presisinya lebih
    panjang dari jumlah desimal), otomatis dipakai notasi ilmiah,
    contoh: 0.000045776 -> '4,5776E-05'.
    """
    if value is None:
        return ""
    if not math.isfinite(value):
        return str(value)
    if value == 0:
        value = 0.0                                   # hindari "-0,0000"
    if value != 0 and round(value, decimals) == 0:
        text = f"{value:.{decimals}E}"
    else:
        text = f"{value:.{decimals}f}"
    return text.replace(".", ",")


# ==========================================================================
# 5. RINGKASAN HASIL
# ==========================================================================
@dataclass
class ResultSummary:
    root_text: str
    f_root_text: str
    iteration_text: str
    conclusion: str


def summarize(result: BisectionResult) -> ResultSummary:
    """Susun teks untuk blok 'Hasil Perhitungan'."""
    d = result.inputs.decimals
    root = format_number(result.root, d)
    f_root = format_number(result.f_root, d)
    eps = format_number(result.inputs.eps, d)
    sol_row = result.rows[result.solution_index]

    if result.converged and result.exact_root:
        conclusion = (f"Akar fungsi berhasil ditemukan pada iterasi ke-{result.root_iteration} "
                      f"dengan nilai xr = {root} dan f(xr) = 0 (akar eksak).")
    elif result.converged:
        conclusion = (f"Akar fungsi berhasil ditemukan pada iterasi ke-{result.root_iteration} "
                      f"dengan nilai xr = {root} dan galat < {eps}.")
    else:
        width = format_number(sol_row.width, d)
        conclusion = (f"Peringatan: iterasi maksimum (N = {result.inputs.n_max}) tercapai sebelum "
                      f"galat < {eps} terpenuhi. Akar pendekatan adalah xr = {root} "
                      f"pada iterasi ke-{result.root_iteration} dengan |b - a| = {width}.")

    return ResultSummary(root, f_root, str(result.root_iteration), conclusion)
