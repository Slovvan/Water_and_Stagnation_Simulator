"""Pruebas de las fases 1 y 2.  Ejecutar desde la raíz:  python -m tests.test_aereo   (o con pytest)."""
import time
import numpy as np

from core.escenario import Escenario
from core.mapa import generar_mapa, mapa_homogeneo, resumen_mapa
from aereo.motor import MotorAereo, capacidad_ga, tp_formula_min, _g
from tests.referencia import ga_acumulada, ga_t_charco
from water.soil_data import AGS, RAWLS_1983

CERRADO = dict(ruido_lluvia=0.0, ks_sigma_ln=0.0, bordes_abiertos=False)


def test_newton_ga():
    rng = np.random.default_rng(0)
    A = 10 ** rng.uniform(0, 3, 5000)
    F = np.where(rng.random(5000) < .3, 0, 10 ** rng.uniform(-3, 2.5, 5000))
    k = 10 ** rng.uniform(-5, 0, 5000)
    Fn = F + capacidad_ga(F, A, k)
    assert np.max(np.abs(_g(Fn, A) - (_g(F, A) + k)) / (_g(F, A) + k)) < 1e-9


def test_mapa_porcentajes_y_reproducibilidad():
    esc = Escenario()
    m = generar_mapa(esc)
    r = resumen_mapa(m)
    total = sum(AGS.pct.values())
    for c, p in r["pct"].items():
        assert abs(p - 100 * AGS.pct[c] / total) <= 0.011, c                 # % exactos del dataset (1 celda de 10 000)
    assert abs(r["confianza"]["N/A"] - 100 * 18.48 / total) < 0.02            # Leptosoles = sin equivalente
    assert np.array_equal(m.soil_id, generar_mapa(esc).soil_id)               # misma semilla, mismo mapa
    assert not np.array_equal(m.soil_id, generar_mapa(esc.con(seed=7)).soil_id)
    assert m.elevacion_m.min() >= 0 and m.elevacion_m.max() <= esc.amplitud_m + 1e-9
    t = time.perf_counter(); generar_mapa(esc); assert time.perf_counter() - t < 0.2


def test_green_ampt_columna():
    """F(t) del motor frente a la solución implícita, y tiempo de encharcamiento frente al criterio del propio GA."""
    peor = 0.0
    for tex in ("Clay", "Clay Loam", "Silt Loam", "Loam", "Sandy Loam"):
        for i in (30.0, 60.0):
            esc = Escenario(n=6, intensidad_mm_h=i, duracion_min=500, **CERRADO)
            mo = MotorAereo(mapa_homogeneo(tex, n=6), esc)
            for t in (30, 60, 90):
                mo.avanzar_hasta(t)
                ga = ga_acumulada(tex, i, t)
                peor = max(peor, abs(mo.F[2, 2] - ga) / ga)
            mo2 = MotorAereo(mapa_homogeneo(tex, n=6), esc)
            mo2.avanzar_hasta(100)
            ref = ga_t_charco(tex, i)
            assert abs(mo2.t_charco[2, 2] - ref) <= max(0.4, 0.05 * ref), (tex, i, mo2.t_charco[2, 2], ref)
    assert peor < 0.03, peor
    return peor


def test_tp_formula_del_documento():
    psi, ks, th = RAWLS_1983["Loam"]
    assert abs(tp_formula_min(psi, ks, th, 30.0) - 18.33) < 0.01              # tabla 1 del documento
    assert tp_formula_min(*RAWLS_1983["Sandy Loam"], 12.0) > 1500             # i = 12 > Ks = 10.9: tarda ~26 h
    assert tp_formula_min(*RAWLS_1983["Sandy Loam"], 10.0) == float("inf")    # i <= Ks: no se encharca


def test_balance_y_bordes_abiertos():
    for abiertos in (True, False):
        esc = Escenario(duracion_min=40, bordes_abiertos=abiertos)
        mo = MotorAereo(generar_mapa(esc), esc)
        mo.avanzar_hasta(90)                                                    # 40 min de lluvia y 50 de drenaje
        assert abs(mo.error_balance()) < 1e-6, mo.error_balance()
        assert mo.H.min() >= 0 and mo.F.min() >= 0
        assert (mo.salida_cel > 0) == abiertos                                  # cerrado: nada sale
        assert np.all((mo.F - mo.D) <= mo.cap + 1e-6)                           # el perfil no se sobrellena


def test_sin_envoltura_de_bordes():
    """Plano inclinado: el agua de la fila 99 baja a la 98; no aparece en la fila 0 (el motor anterior la 'teletransportaba')."""
    n = 100
    elev = np.tile(np.linspace(0, 1, n)[:, None], (1, n))                      # fila 0 baja, fila 99 alta
    esc = Escenario(n=n, intensidad_mm_h=0, **CERRADO)
    mo = MotorAereo(mapa_homogeneo("Clay", n=n, elevacion_m=elev), esc)
    mo.H[99, 50] = 10.0
    mo._rutear()
    assert mo.H[98].sum() > 0 and mo.H[0].sum() == 0 and abs(mo.H.sum() - 10.0) < 1e-12
    elev2 = elev[::-1].copy()                                                   # ahora la fila 0 es la alta
    mo = MotorAereo(mapa_homogeneo("Clay", n=n, elevacion_m=elev2), esc)
    mo.H[0, 50] = 10.0
    mo._rutear()
    assert mo.H[1].sum() > 0 and mo.H[99].sum() == 0
    # borde abierto: el agua de la celda más baja del borde sale y se cuenta
    mo = MotorAereo(mapa_homogeneo("Clay", n=n, elevacion_m=elev), esc.con(bordes_abiertos=True))
    mo.H[0, 50] = 10.0
    mo._rutear()
    assert mo.salida_cel > 0 and abs(mo.H.sum() + mo.salida_cel - 10.0) < 1e-12


def test_el_agua_encharcada_se_infiltra_despues_de_la_lluvia():
    esc = Escenario(n=6, intensidad_mm_h=60, duracion_min=20, **CERRADO)
    mo = MotorAereo(mapa_homogeneo("Loam", n=6), esc)
    mo.avanzar_hasta(20)
    h0 = mo.H.sum()
    assert h0 > 0
    mo.avanzar_hasta(120)
    assert mo.H.sum() < 0.2 * h0                                                # el motor anterior lo dejaba congelado
    assert abs(mo.error_balance()) < 1e-9
    assert not np.isnan(mo.t_mitad_min) and mo.t_mitad_min > 0


def test_el_agua_que_llega_se_infiltra():
    """Un sumidero de Loam recibe agua de una ladera de Clay y debe infiltrarla (no solo acumularla)."""
    n = 20
    elev = np.tile(np.linspace(1, 0, n)[None, :], (n, 1)) * 0.5               # baja hacia la derecha
    esc = Escenario(n=n, intensidad_mm_h=30, duracion_min=60, **CERRADO)
    mo = MotorAereo(mapa_homogeneo("Clay", n=n, elevacion_m=elev), esc)
    mo.ks[:, -1] = 7.6                                                         # la última columna es mucho más permeable
    mo.A[:, -1] = 88.9 * 0.7 * 0.434
    mo.ks_dt = mo.ks * mo.dt_h
    mo.avanzar_hasta(90)
    solo_lluvia = ga_acumulada('Loam', 30.0, 60.0)               # lo que infiltraría con la lluvia directa nada más
    assert mo.F[:, -1].mean() > solo_lluvia + 5.0 and mo.F[:, -1].mean() > 4 * mo.F[:, 0].mean()


def test_perfil_somero_se_satura():
    """Perfil pequeño: al llenarse, el exceso de saturación se encharca aunque Ks sea alta."""
    esc = Escenario(n=6, intensidad_mm_h=20, duracion_min=300, **CERRADO)
    mo = MotorAereo(mapa_homogeneo("Sandy Loam", n=6, profundidad_mm=100.0, k_fondo_mm_h=0.5), esc)
    mo.avanzar_hasta(300)
    assert np.all((mo.F - mo.D) <= mo.cap + 1e-6)
    assert mo.H.mean() > 5.0 and mo.D.mean() > 0
    assert abs(mo.error_balance()) < 1e-9


def test_lluvia_en_vivo_y_velocidad_de_visualizacion():
    esc = Escenario(n=10, ruido_lluvia=0.3, duracion_min=500, **{k: v for k, v in CERRADO.items() if k != "ruido_lluvia"})
    m = mapa_homogeneo("Loam", n=10)
    a = MotorAereo(m, esc)
    for k in range(180):
        if k == 90:
            a.set_intensidad(60.0)
        a.step()
    assert abs(a.lluvia_cel / 100 - (30 * 15 / 60 + 60 * 15 / 60)) < 0.8      # 7.5 + 15 mm, el ruido promedia
    b = MotorAereo(m, esc)
    for k in range(180):                                                       # mismo resultado avanzando por bloques
        if k == 90:
            b.set_intensidad(60.0)
    # (la velocidad de visualización solo cambia cuántos pasos se piden por fotograma, no la física)
    c, d = MotorAereo(m, esc), MotorAereo(m, esc)
    c.avanzar(120); d.avanzar(7); d.avanzar(33); d.avanzar(80)
    assert np.array_equal(c.H, d.H) and np.array_equal(c.F, d.F)


def test_velocidad():
    esc = Escenario()
    mo = MotorAereo(generar_mapa(esc), esc)
    mo.avanzar(60)
    t = time.perf_counter(); mo.avanzar(300)
    ms = 1000 * (time.perf_counter() - t) / 300
    print(f"   motor aéreo: {ms:.2f} ms por paso (100 × 100)")
    assert ms < 6.0


def test_los_charcos_quedan_horizontales():
    """Cuenca en forma de cuenco: el agua que se acumula y crece debe tener la superficie horizontal (antes llegaba a >70 mm de desnivel)."""
    n = 40
    y, x = np.mgrid[0:n, 0:n]
    elev = 0.0004 * ((x - 20) ** 2 + (y - 20) ** 2)                            # cuenco de 0.3 m de profundidad
    esc = Escenario(n=n, intensidad_mm_h=60, duracion_min=500, **CERRADO)
    mo = MotorAereo(mapa_homogeneo("Clay", n=n, elevacion_m=elev), esc)
    peor = 0.0
    for k in range(400):
        mo.step()
        if k > 60 and mo.paso % 2 == 1:                                          # justo después de nivelar
            from aereo.motor import etiquetar, POZA_MM
            lab, nc = etiquetar(mo.H > POZA_MM)
            L = mo.z + mo.H * 1e-3
            for l in range(1, nc + 1):
                c = lab == l
                if c.sum() >= 6:
                    peor = max(peor, (L[c].max() - L[c].min()) * 1000)
    assert peor < 1e-6, peor                                                    # mm de desnivel dentro de un charco
    assert mo.H.max() > 20 and abs(mo.error_balance()) < 1e-6


def test_lluvia_se_detiene_se_prolonga_y_es_continua():
    esc = Escenario(n=6, intensidad_mm_h=30, duracion_min=10, **CERRADO)
    mo = MotorAereo(mapa_homogeneo("Loam", n=6), esc)
    mo.avanzar_hasta(5)
    r5 = mo.lluvia_cel
    assert abs(r5 / 36 - 2.5) < 1e-9 and mo.llueve
    mo.detener_lluvia(); mo.avanzar_hasta(8)
    assert mo.lluvia_cel == r5 and not mo.llueve
    mo.reanudar_lluvia(); mo.avanzar_hasta(12)                                   # t = 8 < 10: llueve hasta el minuto 10
    assert abs(mo.lluvia_cel / 36 - (2.5 + 2 * 0.5 * 1.0)) < 1e-9 and not mo.llueve
    mo.prolongar(30)                                                             # +30 min: llueve hasta 42
    mo.avanzar_hasta(20)
    assert mo.llueve and mo.duracion_min == 42.0
    mo.set_duracion(float("inf")); mo.avanzar_hasta(100)
    assert mo.llueve                                                             # lluvia continua: no se corta nunca
    mo.detener_lluvia(); mo.avanzar_hasta(101)
    assert not mo.llueve


def test_contadores_por_celda_cierran_el_balance_de_cada_celda():
    esc = Escenario(duracion_min=60)
    mo = MotorAereo(generar_mapa(esc), esc)
    mo.avanzar_hasta(90)
    resid = mo.P + mo.recibido - mo.enviado - mo.F - mo.H                         # lluvia + recibido − enviado − infiltrado − agua = 0
    assert np.abs(resid).max() < 1e-8, np.abs(resid).max()
    assert abs(mo.salida_lado.sum() - mo.salida_cel) < 1e-9


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in pruebas:
        t = time.perf_counter()
        r = f()
        extra = f" (desviación máx. de F: {100 * r:.2f} %)" if f.__name__ == "test_green_ampt_columna" else ""
        print(f"OK  {f.__name__}  {time.perf_counter() - t:.1f} s{extra}")
    print("Todas las pruebas pasaron")
