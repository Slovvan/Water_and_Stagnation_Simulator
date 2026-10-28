"""Pruebas del controlador y de los dibujos (fase 5, sin pygame).  Ejecutar:  python -m tests.test_app"""
import time
import numpy as np

from app.estado import Controlador
from app.render import VistaAerea, imagen_corte, Gotas
from core.escenario import Escenario


def test_vistas_y_reloj():
    c = Controlador(Escenario(duracion_min=60))
    assert c.vista == "aerea" and c.lupa is None
    c.pasos_por_frame = 6
    assert c.avanzar(1000) == 6 and c.motor.paso == 6
    c.alternar_vista()                                                    # mapa -> corte lateral
    assert c.vista == "corte" and c.lupa is None                          # sin segundo motor
    assert c.avanzar(1000) == 6 and c.motor.paso == 12                    # el corte avanza el MISMO motor
    c.abrir_detalle()
    assert c.vista == "detalle" and c.lupa is not None and c.lupa.sim.current_step == c.motor.paso
    p = c.motor.paso
    c.avanzar(1000)
    assert c.motor.paso == p and c.lupa.sim.current_step == p + 6         # en el detalle solo avanza la lupa
    c.volver()
    assert c.vista == "corte" and c.lupa is None and c.motor.paso == p
    c.alternar_vista()
    assert c.vista == "aerea"
    c.pausa = True
    assert c.avanzar(1000) == 0


def test_cambiar_de_vista_no_toca_la_simulacion():
    c = Controlador(Escenario(duracion_min=60))
    c.pasos_por_frame = 50
    c.avanzar(1e9)
    H, F, paso = c.motor.H.copy(), c.motor.F.copy(), c.motor.paso
    for _ in range(3):
        c.alternar_vista()
        c.corte.imagen(c.motor)
        c.resumen()
    assert c.motor.paso == paso and np.array_equal(H, c.motor.H) and np.array_equal(F, c.motor.F)


def test_controles_en_vivo():
    c = Controlador(Escenario(duracion_min=500))
    c.set_intensidad(60.0)
    assert c.motor.intensidad_mm_h == 60.0 and c.esc.intensidad_mm_h == 60.0
    c.seleccionar(30, 30); c.abrir_detalle()
    c.set_intensidad(10.0); c.set_duracion(0.0)
    assert abs(c.lupa.sim.rain_intensity_mm_h - 10.0) < 1e-12 and c.motor.duracion_min == 0.0 and not c.lupa.llueve
    c.regenerar(seed=5, intensidad_mm_h=45.0)                             # regenerar conserva lo que está en pantalla
    assert c.esc.seed == 5 and c.motor.intensidad_mm_h == 45.0 and c.vista == "aerea" and c.motor.paso == 0
    assert c.indice == c.mapa.n - 1                                       # la línea vuelve al lado derecho
    v = c.version
    c.reiniciar()
    assert c.version == v + 1 and c.motor.paso == 0


def test_velocidad_de_visualizacion_no_cambia_la_fisica():
    a, b = Controlador(Escenario(duracion_min=500)), Controlador(Escenario(duracion_min=500))
    a.pasos_por_frame, b.pasos_por_frame = 1, 7
    a.alternar_vista()                                                    # una en el mapa y otra en el corte: igual
    for _ in range(42):
        a.avanzar(1e9)
    for _ in range(6):
        b.avanzar(1e9)
    assert a.motor.paso == b.motor.paso == 42
    assert np.array_equal(a.motor.H, b.motor.H) and np.array_equal(a.motor.F, b.motor.F)


def test_presupuesto_de_tiempo():
    c = Controlador()
    c.pasos_por_frame = 1000
    t = time.perf_counter()
    hechos = c.avanzar(presupuesto_ms=15.0)
    ms = 1000 * (time.perf_counter() - t)
    assert 1 <= hechos < 1000 and ms < 60.0, (hechos, ms)


def test_dibujos():
    c = Controlador(Escenario(duracion_min=120))
    c.pasos_por_frame = 200
    for _ in range(6):
        c.avanzar(1e9)
    va = c.vista_aerea
    img = va.imagen(c.motor.H)
    assert img.shape == (100, 100, 3) and img.dtype == np.uint8
    assert (img != va.imagen(np.zeros_like(c.motor.H))).any()             # el agua cambia la imagen
    t = time.perf_counter()
    for _ in range(50):
        va.imagen(c.motor.H); va.sumideros(c.motor.H)
    ms = 20 * (time.perf_counter() - t)
    print(f"   dibujo aéreo: {ms:.2f} ms por fotograma")
    assert ms < 5.0
    c.abrir_detalle(); c.avanzar(1e9)
    ic = imagen_corte(c.lupa.sim, c.lupa.medio)
    assert ic.shape == (200, 200, 3)
    g = Gotas(200, 200)
    seg = g.segmentos(30.0, c.lupa.sim.surface_y())
    assert seg.shape[1] == 3 and (seg[:, 2] >= seg[:, 1]).all()


def test_resumen_de_las_tres_vistas():
    c = Controlador(Escenario(duracion_min=60))
    c.pasos_por_frame = 100
    for _ in range(5):
        c.avanzar(1e9)
    for vista in ("aerea", "corte", "detalle"):
        if vista == "detalle":
            c.abrir_detalle()
        elif vista == "corte":
            c.abrir_corte()
        lineas = c.resumen()
        assert len(lineas) >= 11
        for linea in lineas:
            assert isinstance(linea, str) and len(linea) < 70, linea


if __name__ == "__main__":
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        t = time.perf_counter()
        f()
        print(f"OK  {f.__name__}  {time.perf_counter() - t:.1f} s")
    print("Todas las pruebas de la aplicación pasaron")
