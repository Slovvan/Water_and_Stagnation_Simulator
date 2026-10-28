"""Validaciones del CA con suelo.  Ejecutar desde esta carpeta:  python validar_suelo.py"""
import numpy as np

from water.soil_data import RAWLS_1983
from water.soil_terrain import flat_soil, generate_soil_terrain
from water.water_class import water_cellular_automata


def ga_acumulada(tex, i, t_min, S0=0.3):
    """Infiltración acumulada F(t) [mm] de Green-Ampt con lluvia constante (solución implícita)."""
    psi, ks, th = RAWLS_1983[tex]
    A = psi * (1 - S0) * th
    tp = A * ks / (i - ks) / i * 60                               # min (misma fórmula de tu xlsx)
    if t_min <= tp:
        return i * t_min / 60
    Fp = i * tp / 60
    ts = (Fp - A * np.log(1 + Fp / A)) / ks                       # h
    rhs = ks * (t_min / 60 - tp / 60 + ts)
    lo, hi = Fp, 1e4
    for _ in range(80):                                           # bisección de F - A ln(1+F/A) = rhs
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mid - A * np.log(1 + mid / A) < rhs else (lo, mid)
    return 0.5 * (lo + hi)


def ga_t_charco(tex, i, umbral_mm=0.02):
    """Tiempo en que el propio Green-Ampt acumula `umbral_mm` de exceso (lluvia − infiltración)."""
    for t in np.arange(0.05, 240, 0.05):
        if i * t / 60 - ga_acumulada(tex, i, t) > umbral_mm:
            return t
    return np.inf


def correr_columna(tex, i, minutes, dz=5.0, dt=10.0, S0=0.3):
    soil = flat_soil(tex, grid_size=60, dz_mm=dz, surface_row=10)
    sim = water_cellular_automata(60, int(minutes * 60 / dt), rain_intensity_mm_h=i, dz_mm=dz, dt_s=dt,
                                  substeps=4, soil=soil, soil_initial_saturation=S0, bottom_drain=False)
    init = S0 * soil.cap[10:, 5].sum() * dz
    tp_ca, F = None, {}
    for k in range(1, sim.time_steps + 1):
        sim.step()
        t = k * dt / 60
        if tp_ca is None and sim.surface_water_mm > 0.02:         # primer charco medible (0.02 mm)
            tp_ca = t
        for tt in (30, 60, 90):
            if abs(t - tt) < 1e-9:
                F[tt] = sim.grid[10:, 5].sum() * dz - init
    return tp_ca, F


print("1) Columna homogénea vs Green-Ampt (S0 = 0.30). tp = tiempo a encharcamiento [min]; F = infiltración [mm]")
print("   (tp GA = fórmula de tu xlsx · «0.02» = cuándo el propio GA acumula 0.02 mm de charco, mismo criterio que el CA)")
print(f"{'textura':<11}{'i':>5}{'tp GA':>8}{'GA 0.02':>9}{'tp CA':>8}   F(30/60/90 min)  CA  vs  GA")
worst = 0.0
for tex in ["Clay Loam", "Silt Loam", "Loam", "Sandy Loam"]:
    for i in (30.0, 60.0):
        psi, ks, th = RAWLS_1983[tex]
        tp_a = psi * 0.7 * th * ks / (i - ks) / i * 60
        tp_s, F = correr_columna(tex, i, 90)
        ga = {t: ga_acumulada(tex, i, t) for t in (30, 60, 90)}
        worst = max(worst, *(abs(F[t] - ga[t]) / ga[t] for t in F))
        print(f"{tex:<11}{i:5.0f}{tp_a:8.1f}{ga_t_charco(tex, i):9.1f}{(tp_s or float('nan')):8.1f}   "
              + "  ".join(f"{F[t]:5.1f}/{ga[t]:5.1f}" for t in (30, 60, 90)))
print(f"   peor desviación de F(t): {100 * worst:.1f} %")
assert worst < 0.10

print("\n2) Sin suelo se usan tus reglas originales (sin cambios): se conserva la masa")
n = 40
a = water_cellular_automata(n, 200, rain_intensity_mm_h=30, rain_seed=1, substeps=2)
for _ in range(200):
    a.step()
print(f"   agua en rejilla = {a.stored_mm:.3f} mm · lluvia = {a.rain_mm:.3f} mm · error masa = {a.mass_balance_error():.1e}")

print("\n3) Terreno procedural completo: tormenta 3 h + 7 h de drenaje, bordes abiertos")
soil = generate_soil_terrain(grid_size=200, dz_mm=10, seed=3)
print(soil.summary())
sim = water_cellular_automata(200, 600, rain_steps=180, rain_intensity_mm_h=30, rain_noise=0.3, rain_seed=40,
                              dz_mm=10, dt_s=60, substeps=4, soil=soil, edge_runoff=True)
for _ in range(600):
    sim.step()
print(f"   lluvia {sim.rain_mm:.1f} mm · escorrentía fuera {sim.runoff_mm:.1f} mm · percolación profunda "
      f"{sim.deep_mm:.2f} mm · en suelo {sim.soil_water_mm:.1f} mm · charcos {sim.surface_water_mm:.1f} mm")
print(f"   error de masa = {sim.mass_balance_error():.2e} unidades")
assert abs(sim.mass_balance_error()) < 1e-6
assert sim.grid.min() >= -1e-9 and np.all(sim.grid <= soil.cap + 1e-9), "agua fuera de [0, capacidad]"

print("\n4) Intensidad de lluvia: 60 mm/h debe producir más agua superficial que 30 mm/h")
surface_by_intensity = {}
for intensity in (30.0, 60.0):
    sim = water_cellular_automata(200, 180, rain_intensity_mm_h=intensity, rain_noise=0.3, rain_seed=40,
                                  dz_mm=10, dt_s=60, substeps=4, soil=soil, edge_runoff=True)
    for _ in range(180):
        sim.step()
    surface_by_intensity[intensity] = sim.surface_water_mm
    print(f"   {intensity:.0f} mm/h: lluvia {sim.rain_mm:.1f} mm · charcos {sim.surface_water_mm:.1f} mm · "
          f"escorrentía {sim.runoff_mm:.1f} mm · error de masa {sim.mass_balance_error():.2e}")
    assert abs(sim.mass_balance_error()) < 1e-6
    assert sim.grid.min() >= -1e-9 and np.all(sim.grid <= soil.cap + 1e-9), "agua fuera de [0, capacidad]"
assert surface_by_intensity[60.0] > surface_by_intensity[30.0]
print("OK")
