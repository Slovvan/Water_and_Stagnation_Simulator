"""Verifica la infiltración de TODOS los materiales del proyecto con el motor aéreo.
   python -m tests.verificar_materiales
Por material e intensidad: tp de la fórmula del xlsx, tiempo en que el propio Green-Ampt acumula 0.02 mm de charco,
tiempo del motor, y F(60 min) del motor frente a Green-Ampt (solución implícita). Al final: la infiltración por grupo
de suelo en el mapa por defecto."""
import numpy as np

from aereo.motor import MotorAereo, tp_formula_min
from core.escenario import Escenario
from core.mapa import generar_mapa, mapa_homogeneo
from tests.referencia import ga_acumulada, ga_t_charco
from water.soil_data import CUSTOM_MATERIALS, RAWLS_1983

CERRADO = dict(ruido_lluvia=0.0, ks_sigma_ln=0.0, bordes_abiertos=False)
MATERIALES = {**RAWLS_1983, **{k: v for k, v in CUSTOM_MATERIALS.items() if k == "Leptosol"}}


def ga_leptosol(i, t_min, A, ks):
    return ga_acumulada_param(A, ks, i, t_min)


def ga_acumulada_param(A, ks, i, t_min):
    if i <= ks:
        return i * t_min / 60
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


def filas(intensidades=(30.0, 60.0), minutos=60):
    out = []
    for nombre, (psi, ks, th) in MATERIALES.items():
        A = psi * 0.7 * th
        for i in intensidades:
            esc = Escenario(n=6, intensidad_mm_h=i, duracion_min=1e6, **CERRADO)
            m = mapa_homogeneo(nombre, n=6, profundidad_mm=1e5) if nombre in RAWLS_1983 else None
            if m is None:                                         # Leptosol: perfil somero real (no infinito)
                m = mapa_homogeneo("Leptosol", n=6, profundidad_mm=1e5)
            mo = MotorAereo(m, esc)
            mo.avanzar_hasta(minutos)
            F = float(mo.F[2, 2])
            ref = ga_acumulada_param(A, ks, i, minutos)
            tp = tp_formula_min(psi, ks, th, i)
            t02 = ga_t_charco(nombre, i) if nombre in RAWLS_1983 and i > ks else np.inf
            mo2 = MotorAereo(m, esc)
            mo2.avanzar_hasta(300 if np.isfinite(t02) and t02 < 250 else 30)
            out.append((nombre, ks, i, tp, t02, float(mo2.t_charco[2, 2]), F, ref))
    return out


if __name__ == "__main__":
    print(f"{'material':<16}{'Ks':>7}{'i':>5}{'tp xlsx':>9}{'GA 0.02':>9}{'motor':>8}   F(60) motor / Green-Ampt   desv")
    peor = 0.0
    for nombre, ks, i, tp, t02, tm, F, ref in filas():
        f = lambda v: "   -  " if not np.isfinite(v) else f"{v:7.1f}"
        d = 100 * (F - ref) / ref
        peor = max(peor, abs(d))
        print(f"{nombre:<16}{ks:7.1f}{i:5.0f}{f(tp):>9}{f(t02):>9}{f(tm):>8}   {F:8.2f} / {ref:8.2f}   {d:+5.2f}%")
    print(f"\npeor desviación de F(60 min): {peor:.2f} %")
    # --- por grupo de suelo en el mapa por defecto ---
    esc = Escenario(duracion_min=180)
    mapa = generar_mapa(esc)
    mo = MotorAereo(mapa, esc)
    mo.avanzar_hasta(60)
    print("\nMapa por defecto a los 60 min (30 mm/h; Ks con variabilidad lognormal sigma = 0.5):")
    print(f"{'grupo':<12}{'celdas':>7}{'Ks sup':>8}{'F medio':>9}{'F/lluvia':>9}{'perfil lleno':>13}{'con charco':>11}{'1er charco':>11}")
    lluvia = 30.0
    for k, cod in enumerate(mapa.codes):
        sel = mapa.soil_id == k
        lleno = ((mo.F - mo.D) >= mo.cap - 1e-6)[sel].mean() * 100
        ch = (mo.t_charco[sel] < np.inf)
        t1 = np.median(mo.t_charco[sel][ch]) if ch.any() else np.inf
        print(f"{mapa.nombres[cod]:<12}{sel.sum():7d}{np.median(mapa.ks_mm_h[sel]):8.1f}{mo.F[sel].mean():9.1f}{mo.F[sel].mean() / lluvia:9.2f}{lleno:12.0f}%{100 * ch.mean():10.0f}%"
              f"{('%.1f min' % t1) if np.isfinite(t1) else '   -':>11}")
