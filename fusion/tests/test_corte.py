"""Pruebas del corte lateral sincronizado.  Ejecutar:  python -m tests.test_corte"""
import time
import numpy as np

from app.corte_sync import CorteSincronizado, FILAS
from app.estado import Controlador
from app.render import Gotas
from aereo.motor import MotorAereo
from core.escenario import Escenario
from core.mapa import generar_mapa, mapa_homogeneo

CERRADO = dict(ruido_lluvia=0.0, ks_sigma_ln=0.0, bordes_abiertos=False)


def test_lado_derecho_por_defecto_y_misma_topografia():
    c = Controlador(Escenario())
    assert c.eje == "col" and c.indice == c.mapa.n - 1
    cs = c.corte
    assert np.array_equal(cs.z, c.mapa.elevacion_m[:, -1])                 # el relieve del corte ES el del mapa
    assert np.array_equal(cs.depth, c.mapa.profundidad_mm[:, -1])
    c.alternar_eje(); c.mover_linea(-5)
    assert c.eje == "fila" and c.corte.eje == "fila"
    assert np.array_equal(c.corte.z, c.mapa.elevacion_m[c.indice, :])


def test_el_frente_de_humedad_sale_de_F():
    esc = Escenario(n=10, intensidad_mm_h=30, duracion_min=500, **CERRADO)
    m = mapa_homogeneo("Loam", n=10, profundidad_mm=1200.0)
    mo = MotorAereo(m, esc)
    mo.avanzar_hasta(60)
    cs = CorteSincronizado(m, esc)
    frente = cs.frente_mm(*cs.datos(mo)[:2])
    esperado = mo.F[0, -1] / ((1 - esc.s0) * 0.434)                          # Green-Ampt: profundidad = F / Δθ
    assert abs(frente.mean() - esperado) / esperado < 0.01, (frente.mean(), esperado)
    img, _ = cs.imagen(mo)
    # filas mojadas por columna (azul fuerte) ≈ frente / (mm por fila)
    mm_fila = cs.dzv * 1000.0
    mojadas = (img == cs.img_mojada).all(axis=2) & (cs.d_suelo < np.inf)
    assert abs(mojadas[:, 3].sum() * mm_fila - esperado) < 2 * mm_fila + 1e-6


def test_sin_segundo_motor_y_cambia_con_el_tiempo():
    c = Controlador(Escenario(duracion_min=120))
    c.abrir_corte()
    assert c.lupa is None
    c.pasos_por_frame = 30
    c.avanzar(1e9)
    a, _ = c.corte.imagen(c.motor)
    for _ in range(20):
        c.avanzar(1e9)
    b, _ = c.corte.imagen(c.motor)
    assert (a != b).any() and c.motor.paso == 21 * 30


def test_agua_dibujada_coincide_con_H():
    esc = Escenario(n=10, intensidad_mm_h=30, duracion_min=500, **CERRADO)
    m = mapa_homogeneo("Clay", n=10, profundidad_mm=1000.0)               # la arcilla encharca casi toda la lluvia
    mo = MotorAereo(m, esc)
    mo.avanzar_hasta(30)
    cs = CorteSincronizado(m, esc)
    H = mo.H[cs.rr, cs.cc]
    assert H.min() > 5.0
    img, sup_y = cs.imagen(mo)
    filas_agua = cs.fila_sup - sup_y
    assert np.allclose(filas_agua, H / 1000.0 / cs.dzv)                      # espesor del agua dibujado = H, a escala
    for col in range(cs.n):
        r0, r1 = int(np.ceil(sup_y[col] - 0.5)), int(np.floor(cs.fila_sup[col] - 0.5))
        pix = img[r0:r1 + 1, col]
        assert len(pix) >= 1 and (pix[:, 2] > pix[:, 0]).all()               # azul sobre la superficie
    assert (img[0, 0] == (cs.cielo).astype(np.uint8)).all()                  # y el cielo queda arriba
    # la superficie del agua de un charco es horizontal: misma fila superior en todas sus columnas
    assert np.ptp(np.floor(sup_y)) == 0 or np.ptp(sup_y) < 1e-9


def test_gotas_visibles_y_se_detienen_en_la_superficie():
    c = Controlador(Escenario())
    cs = c.corte
    g = Gotas(cs.n, FILAS, n_max=260, caida=7.0, estela=6.0)
    sup_y = cs.fila_sup.copy()
    for _ in range(30):
        seg = g.segmentos(30.0, sup_y)
    assert len(seg) > 50
    col = np.clip(np.rint(seg[:, 0]).astype(int), 0, cs.n - 1)
    assert (seg[:, 2] <= sup_y[col] + 1e-9).all()                              # nunca atraviesan el suelo ni el agua
    assert len(g.segmentos(0.0, sup_y)) == 0                                   # sin lluvia, ninguna
    # y están en el aire (arriba del terreno), así que se ven
    assert (seg[:, 2] - seg[:, 1] > 0).mean() > 0.5


def test_la_leyenda_muestra_varios_materiales_y_el_corte_es_rapido():
    c = Controlador(Escenario())
    cs = c.corte
    assert len(cs.leyenda) >= 4, cs.leyenda
    c.motor.avanzar_hasta(30)
    t = time.perf_counter()
    for _ in range(50):
        cs.imagen(c.motor)
    ms = 20 * (time.perf_counter() - t)
    print(f"   corte sincronizado: {ms:.2f} ms por fotograma, {len(cs.leyenda)} materiales en la leyenda")
    assert ms < 6.0


def test_vertical_y_horizontal_cubren_todo_el_mapa():
    c = Controlador(Escenario())
    cs = c.corte
    assert cs.n == c.mapa.n == 100
    assert cs.top > cs.z.max() and cs.bottom < (cs.z - cs.depth / 1000.0).min()   # todo el relieve y los perfiles caben


if __name__ == "__main__":
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        t = time.perf_counter()
        f()
        print(f"OK  {f.__name__}  {time.perf_counter() - t:.1f} s")
    print("Todas las pruebas del corte sincronizado pasaron")
