"""Pruebas de la fase 4 (lupa).  Ejecutar:  python -m tests.test_lupa"""
import time
import numpy as np

from core.escenario import Escenario
from core.mapa import generar_mapa, mapa_homogeneo
from aereo.motor import MotorAereo
from corte.lupa import Lupa
from corte.medio import construir_medio, celdas_del_transecto, N_CORTE
from tests.referencia import ga_acumulada

CERRADO = dict(ruido_lluvia=0.0, ks_sigma_ln=0.0, bordes_abiertos=False)


def test_medio_usa_los_datos_del_mapa():
    esc = Escenario()
    m = generar_mapa(esc)
    for (r, c, d) in ((50, 50, "E"), (0, 0, "O"), (99, 99, "E"), (10, 99, "S")):
        medio, info = construir_medio(m, r, c, d, bedrock=esc.bedrock, semilla=esc.seed)
        assert medio.kind.shape == (N_CORTE, N_CORTE)
        assert medio.surface_row.min() >= 1 and medio.surface_row.max() <= N_CORTE - 8
        a, b = info["celdas"]
        assert 0 <= a[0] < 100 and 0 <= b[0] < 100 and 0 <= a[1] < 100 and 0 <= b[1] < 100 and a != b
        # el horizonte superficial de cada mitad tiene el Ks de su celda aérea (sin variabilidad local)
        for lado, (rr, cc) in enumerate((a, b)):
            col = 20 + 100 * lado
            fila = medio.surface_row[col]
            assert abs(medio.ks_mm_h[fila, col] - m.ks_mm_h[rr, cc]) / m.ks_mm_h[rr, cc] < 1e-9
    # misma celda, misma dirección: mismo medio
    a1, _ = construir_medio(m, 40, 40, "E", semilla=3)
    a2, _ = construir_medio(m, 40, 40, "E", semilla=3)
    assert np.array_equal(a1.kind, a2.kind) and np.array_equal(a1.ks_mm_h, a2.ks_mm_h)


def test_inicializacion_conserva_el_agua():
    esc = Escenario(duracion_min=30, ruido_lluvia=0.0)
    m = generar_mapa(esc)
    mo = MotorAereo(m, esc)
    mo.avanzar_hasta(20)
    r, c = 60, 40
    lu = Lupa(mo, r, c, "E")
    a, b = lu.info["celdas"]
    izq = lu.info["columna_celda"] == 0
    F = np.where(izq, mo.F[a], mo.F[b]).mean()
    H = np.where(izq, mo.H[a], mo.H[b]).mean()
    esperado_suelo = lu.suelo0_mm
    # agua del suelo = S0·capacidad + F (cuando el perfil no se desborda)
    base = esc.s0 * lu.sim.scale.depth_mm(lu.medio.cap[lu.medio.kind == 1].sum(), N_CORTE)
    assert abs(esperado_suelo - base - F) < 0.02 * max(F, 1.0), (esperado_suelo, base, F)
    assert abs(lu.sim.surface_water_mm - H) < 0.05 * max(H, 1.0) + 0.02, (lu.sim.surface_water_mm, H)
    lu.avanzar(60)
    assert abs(lu.sim.mass_balance_error()) < 1e-6
    assert lu.sim.grid.min() >= -1e-9 and np.all(lu.sim.grid <= lu.medio.cap + 1e-9)


def test_lupa_y_mapa_infiltran_igual():
    """Suelo homogéneo y plano: la infiltración que la lupa añade en 30 min debe coincidir con la del mapa (≤ 10 %)."""
    for tex in ("Loam", "Silt Loam"):
        esc = Escenario(n=8, intensidad_mm_h=30, duracion_min=500, **CERRADO)
        m = mapa_homogeneo(tex, n=8, profundidad_mm=1200.0)
        mo = MotorAereo(m, esc)
        mo.avanzar_hasta(30)
        lu = Lupa(mo, 3, 3, "E", usar_aporte=False)
        F0 = mo.F[3, 3]
        mo.avanzar_hasta(60)
        lu.avanzar(int(30 * 60 / esc.dt_s))
        dF_mapa = mo.F[3, 3] - F0
        dF_lupa = lu.infiltracion_mm()
        err = abs(dF_lupa - dF_mapa) / dF_mapa
        print(f"   {tex}: infiltración de 30 a 60 min  mapa {dF_mapa:.2f} mm | lupa {dF_lupa:.2f} mm | diferencia {100 * err:.1f} %")
        assert err < 0.10, (tex, dF_lupa, dF_mapa)


def test_lupa_mismo_reloj_y_lluvia_en_vivo():
    esc = Escenario(n=8, intensidad_mm_h=30, duracion_min=500, **CERRADO)
    mo = MotorAereo(mapa_homogeneo("Loam", n=8), esc)
    mo.avanzar(30)
    lu = Lupa(mo, 3, 3, "E", usar_aporte=False)
    assert lu.sim.current_step == mo.paso and abs(lu.tiempo_min - mo.tiempo_min) < 1e-9
    mo.set_intensidad(60); lu.set_intensidad(60)
    r0 = lu.sim.total_rain
    lu.avanzar(6)
    assert lu.sim.total_rain > r0 and abs(lu.sim.rain_intensity_mm_h - 60) < 1e-12
    lu.set_duracion(0)                                    # la lluvia se corta en vivo
    r1 = lu.sim.total_rain
    lu.avanzar(6)
    assert lu.sim.total_rain == r1


def test_aporte_y_velocidad_de_apertura():
    esc = Escenario(duracion_min=60)
    m = generar_mapa(esc)
    mo = MotorAereo(m, esc)
    mo.avanzar_hasta(30)
    assert mo.aporte.max() > 0 and mo.aporte.min() >= 0
    t = time.perf_counter()
    lu = Lupa(mo, 70, 30, "S")
    dt = time.perf_counter() - t
    print(f"   abrir la lupa: {1000 * dt:.0f} ms")
    assert dt < 1.5
    lu.avanzar(30)
    assert lu.metricas()["aporte_mm"] >= 0 and abs(lu.sim.mass_balance_error()) < 1e-6


def test_el_aporte_se_apila_sobre_un_charco_y_se_contabiliza():
    esc = Escenario(n=8, intensidad_mm_h=0.0, duracion_min=500, **CERRADO)
    mo = MotorAereo(mapa_homogeneo("Clay", n=8), esc)
    lu = Lupa(mo, 3, 3, "E", usar_aporte=True)
    lu.aporte_col[:] = 3.0                                # 3 mm por paso sobre todas las columnas (charco que se profundiza)
    lu.sim.rain_rule.set_intensity(0.0)
    r0 = lu.sim.total_rain
    lu.avanzar(40)
    esperado_mm = 3.0 * 40
    assert abs(lu.aporte_total_mm - esperado_mm) < 1e-9
    assert abs(lu.sim.scale.depth_mm(lu.sim.total_rain - r0, N_CORTE) - esperado_mm) < 1e-6      # todo el aporte quedó en el agua
    assert abs(lu.sim.mass_balance_error()) < 1e-6


def test_el_detalle_siempre_tiene_bordes_cerrados():
    """Aunque el mapa se configure con bordes abiertos, el autómata de 2 m no deja salir agua por los lados (si no, parpadea)."""
    esc = Escenario(duracion_min=120, bordes_abiertos=True)
    mo = MotorAereo(generar_mapa(esc), esc)
    mo.avanzar_hasta(60)
    lu = Lupa(mo, 50, 50, "S")
    lu.avanzar(300)
    assert lu.sim.runoff_mm == 0.0 and lu.sim.surface_rule.open_edges is False
    assert abs(lu.sim.mass_balance_error()) < 1e-6
    # el aporte es el NETO que se acumula en la celda (con bordes cerrados no puede salir), nunca negativo
    assert (lu.aporte_col >= 0).all() and (mo.aporte_neto >= 0).all()


if __name__ == "__main__":
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        t = time.perf_counter()
        f()
        print(f"OK  {f.__name__}  {time.perf_counter() - t:.1f} s")
    print("Todas las pruebas de la lupa pasaron")
