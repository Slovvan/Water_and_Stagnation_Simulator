"""Soluciones de referencia de Green-Ampt (las mismas de validar_suelo.py, sin ejecutar su cuerpo)."""
import numpy as np
from water.soil_data import RAWLS_1983


def ga_acumulada(tex, i, t_min, S0=0.3):
    """Infiltración acumulada F(t) [mm], lluvia constante i [mm/h] (solución implícita)."""
    psi, ks, th = RAWLS_1983[tex]
    A = psi * (1 - S0) * th
    tp = A * ks / (i - ks) / i * 60
    if t_min <= tp:
        return i * t_min / 60
    Fp = i * tp / 60
    ts = (Fp - A * np.log(1 + Fp / A)) / ks
    rhs = ks * (t_min / 60 - tp / 60 + ts)
    lo, hi = Fp, 1e4
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mid - A * np.log(1 + mid / A) < rhs else (lo, mid)
    return 0.5 * (lo + hi)


def ga_t_charco(tex, i, umbral_mm=0.02):
    """Minuto en que el propio Green-Ampt acumula `umbral_mm` de exceso (lluvia − infiltración)."""
    for t in np.arange(0.05, 240, 0.05):
        if i * t / 60 - ga_acumulada(tex, i, t) > umbral_mm:
            return t
    return np.inf
