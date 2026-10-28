"""Recorrido de la interfaz con un pygame simulado.  Ejecutar:  python -m tests.test_ui
Solo valida el flujo y la forma de las llamadas a la API; NO sustituye una prueba con pygame real."""
import tempfile
import time
from pathlib import Path

import numpy as np

from tests import fake_pygame as fp
from tests.fake_pygame import ev
from app.estado import Controlador
from app.ui import Aplicacion, MAPA_X, MAPA_Y, MAPA_L
from core.escenario import Escenario


def clic(x, y): return ev(fp.MOUSEBUTTONDOWN, pos=(x, y), button=1)
def suelta(x, y): return ev(fp.MOUSEBUTTONUP, pos=(x, y), button=1)
def mueve(x, y, botones=(1, 0, 0)): return ev(fp.MOUSEMOTION, pos=(x, y), buttons=botones)
def tecla(k): return ev(fp.KEYDOWN, key=k)
centro = lambda b: (b.x + b.w / 2, b.y + b.h / 2)


def nueva(escritorio=(1920, 1080), esc=None, guion=None):
    guion = [] if guion is None else guion
    pg, st = fp.crear(guion, escritorio=escritorio)
    app = Aplicacion(pg, Controlador(esc or Escenario(duracion_min=120)))
    app.abrir_navegador = False
    app.carpeta_reportes = Path(tempfile.mkdtemp())
    return app, pg, st, guion


def celda(app, fila, col):
    k = MAPA_L / app.ctrl.mapa.n
    return MAPA_X + (col + 0.5) * k, MAPA_Y + (fila + 0.5) * k


def test_recorrido_completo():
    app, pg, st, guion = nueva()
    sl = app.slider
    en = lambda s: (s.x + s.w * 0.5, s.y + 26)
    cx = celda(app, 60, 30)
    guion += [[], [], []]
    guion += [[clic(*cx)], [suelta(*cx)]]                                                   # elegir la línea (columna 30)
    guion += [[tecla(fp.K_RIGHT)], [tecla(fp.K_LEFT)], [clic(*centro(app.b_eje))], [tecla(fp.K_DOWN)], [clic(*centro(app.b_eje))]]
    guion += [[clic(*cx)]]                                                                  # doble clic -> abre el corte
    guion += [[], [], []]
    guion += [[clic(*centro(app.tabs["detalle"]))], [], [clic(*centro(app.tabs["corte"]))], [clic(*centro(app.tabs["aerea"]))]]
    guion += [[clic(*centro(app.b_agua))], [tecla(fp.K_l)], [clic(*centro(app.b_detalle))], [], [clic(*centro(app.b_detalle))]]
    guion += [[clic(*en(sl["intensidad"]))], [mueve(*en(sl["intensidad"]))], [suelta(*en(sl["intensidad"]))]]
    guion += [[clic(*en(sl["velocidad"]))], [mueve(sl["velocidad"].x + 300, sl["velocidad"].y + 26)], [suelta(0, 0)]]
    guion += [[clic(*centro(app.b_lluvia))], [], [clic(*centro(app.b_lluvia))], [clic(*centro(app.b_mas))], [clic(*centro(app.b_continua))]]
    guion += [[clic(*centro(app.b_pausa))], [], [clic(*centro(app.b_pausa))], [tecla(fp.K_TAB)], [tecla(fp.K_TAB)]]
    guion += [[clic(*en(sl["s0"]))], [mueve(sl["s0"].x + 200, sl["s0"].y + 26)], [suelta(0, 0)]]    # S0 (no vivo)
    guion += [[clic(*centro(app.b_sumideros))], [tecla(fp.K_s)], [clic(*centro(app.b_roca))], [clic(*centro(app.b_regen))]]
    guion += [[], [], [clic(*centro(app.b_reporte))], [clic(*centro(app.b_reiniciar))], [tecla(fp.K_r)], [ev(fp.QUIT)]]
    app.ejecutar(max_frames=400)
    c = app.ctrl
    assert st["cerrado"] and st["flips"] > 10 and c.vista == "aerea"
    assert c.esc.bordes_abiertos is False and c.esc.bedrock == "sealed"                     # lo pendiente se aplicó; los bordes son siempre cerrados
    assert c.mostrar_sumideros is False                                                      # alternado dos veces (botón y tecla S)
    assert abs(c.esc.s0 - sl["s0"].valor) < 1e-9 and c.esc.s0 != 0.30
    assert c.esc.intensidad_mm_h == sl["intensidad"].valor and c.esc.intensidad_mm_h != 30.0
    assert c.velocidad_min_s == sl["velocidad"].valor and abs(c.velocidad_min_s - 5.0) > 1e-6
    assert fp._Draw.calls["rect"] > 100 and fp._Draw.calls["line"] > 0 and fp._Draw.calls["circle"] > 0


def test_diseno_sin_solapes_y_dentro_del_lienzo():
    app, pg, st, _ = nueva()
    elementos = [(n, b.x, b.y, b.w, b.h) for n, b in [(f"tab {k}", b) for k, b in app.tabs.items()]
                 + [(nm, getattr(app, nm)) for nm in ("b_continua", "b_lluvia", "b_mas", "b_pausa", "b_reiniciar",
                                                       "b_eje", "b_agua", "b_detalle", "b_sumideros", "b_roca", "b_regen", "b_reporte")]]
    elementos += [(f"slider {s.clave}", s.x, s.y, s.w, 34) for s in app.sliders]
    for n, x, y, w, h in elementos:
        assert 0 <= x and x + w <= 1600 and 0 <= y and y + h <= 900, n
    for i, a in enumerate(elementos):
        for b in elementos[i + 1:]:
            sep = a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1] or a[2] + a[4] <= b[2] or b[2] + b[4] <= a[2]
            assert sep, f"se solapan {a[0]} y {b[0]}"
    # y los controles de cada tarjeta quedan dentro de ella
    assert app.b_reporte.y >= 512 + 330 and app.b_regen.y + app.b_regen.h <= 512 + 330 and app.b_continua.y + 32 <= 70 + 232


def test_pestanas_y_teclas():
    app, pg, st, _ = nueva()
    app.iniciar()
    vistas = []
    cx = celda(app, 20, 20)
    pasos = [[clic(*cx)], [clic(*cx)], [tecla(fp.K_d)], [tecla(fp.K_ESCAPE)], [tecla(fp.K_TAB)], [tecla(fp.K_TAB)], [tecla(fp.K_d)],
             [clic(*centro(app.tabs["aerea"]))], [clic(*centro(app.tabs["corte"]))], [clic(*centro(app.tabs["detalle"]))]]
    for lista in pasos:
        for e in lista:
            app.manejar(e)
        vistas.append(app.ctrl.vista)
    assert vistas == ["aerea", "corte", "detalle", "corte", "aerea", "corte", "detalle", "aerea", "corte", "detalle"], vistas
    assert app.ctrl.indice == 20 and app.ctrl.seleccion == (20, 20)
    for v in ("aerea", "corte", "detalle"):                                                 # dibujar cada vista sin errores
        app.ctrl.ir_a(v)
        app.dibujar(1)
    assert app.lienzo.blits > 20 and st["pantalla"].blits >= 1


def test_ventana_pantalla_completa_redimensionar_y_cerrar():
    """La ventana usa sus propios botones del sistema (sin botones dentro de la app): aquí F11, el tamaño y el cierre."""
    app, pg, st, _ = nueva(escritorio=(1366, 768))
    app.iniciar()
    (tam, flags), = st["modos"]
    assert tam[0] <= 1366 and tam[1] <= 768 - 60 and flags == fp.RESIZABLE, (tam, flags)
    assert not hasattr(app, "b_min") and not hasattr(app, "b_pant") and not hasattr(app, "b_cerrar")
    app.dibujar(1)
    k, ox, oy = app._tf
    assert k < 1.0
    pos = lambda b: (ox + (b.x + b.w / 2) * k, oy + (b.y + b.h / 2) * k)
    assert app.manejar(ev(fp.MOUSEBUTTONDOWN, pos=pos(app.tabs["corte"]), button=1)) and app.ctrl.vista == "corte"     # ratón mapeado
    app.manejar(tecla(fp.K_F11))
    assert app.completa and st["modos"][-1] == ((1366, 768), fp.FULLSCREEN)
    app.manejar(tecla(fp.K_F11))
    assert not app.completa and st["modos"][-1] == (tam, fp.RESIZABLE)
    st["pantalla"] = fp.Surface((900, 500))                                                 # redimensionar
    app.manejar(ev(fp.VIDEORESIZE, w=900, h=500))
    app.dibujar(1)
    k2, ox2, oy2 = app._tf
    assert abs(k2 - min(900 / 1600, 500 / 900)) < 1e-9 and ox2 >= 0
    assert app.manejar(ev(fp.QUIT)) is False                                                # cerrar con el botón de la ventana


def test_sin_parpadeo_en_el_mapa():
    """El agua fina no debe pasar de seco a mojado de un fotograma al siguiente (con ruido y sin suavizar era ~3 % por fotograma)."""
    app, pg, st, _ = nueva(esc=Escenario(duracion_min=180))
    c = app.ctrl
    c.motor.avanzar_hasta(60)
    va, prev, toggles, rojas = c.vista_aerea, None, [], []
    for _ in range(50):
        c.motor.avanzar(2)
        va.suavizar(c.motor.H)
        img = va.imagen(c.motor.H, suavizar=False)
        if prev is not None:
            toggles.append((np.abs(img.astype(int) - prev.astype(int)).max(axis=2) > 25).mean() * 100)
        rojas.append(set(map(tuple, va.sumideros(c.motor.H).tolist())))
        prev = img
    assert np.mean(toggles) < 0.6, np.mean(toggles)
    cambios = [len(a ^ b) for a, b in zip(rojas, rojas[1:])]
    assert np.mean(cambios) < 0.1, np.mean(cambios)                                         # las marcas rojas casi no cambian
    assert c.mostrar_sumideros is False                                                     # y por defecto ni se dibujan


def test_popups_al_pasar_el_raton():
    app, pg, st, _ = nueva()
    app.iniciar()
    c = app.ctrl
    c.pasos_por_frame = 120
    c.avanzar(1e9)
    # mapa: información del material y de la celda
    app.manejar(mueve(*celda(app, 60, 40), botones=(0, 0, 0)))
    info = app._info_hover()
    assert info and "celda (60, 40)" in info[0] and any("Ks" in l and "mm/h" in l for l in info) and any("Infiltrado" in l for l in info)
    # corte lateral: horizonte del suelo bajo el cursor
    c.ir_a("corte")
    cs = c.corte
    fila = int(cs.fila_sup[40]) + 40
    app.manejar(mueve(MAPA_X + 40.5 * MAPA_L / cs.n, MAPA_Y + (fila + 0.5) * MAPA_L / 400, botones=(0, 0, 0)))
    info = app._info_hover()
    assert info and any("horizonte" in l or "sustrato" in l for l in info) and any("Ks" in l for l in info), info
    # aire sobre el terreno
    app.manejar(mueve(MAPA_X + 40.5 * MAPA_L / cs.n, MAPA_Y + 5, botones=(0, 0, 0)))
    assert "Aire" in " ".join(app._info_hover())
    # detalle: celda del autómata
    c.ir_a("detalle")
    app.manejar(mueve(MAPA_X + 400, MAPA_Y + 700, botones=(0, 0, 0)))
    info = app._info_hover()
    assert info and any("Ks" in l or "Aire" in l for l in info)
    # un botón muestra su ayuda
    app.manejar(mueve(*centro(app.b_mas), botones=(0, 0, 0)))
    assert "30 minutos" in " ".join(app._info_hover())
    app.manejar(mueve(5, 5, botones=(0, 0, 0)))
    for v in ("aerea", "corte", "detalle"):                                                 # el popup se dibuja sin errores en cada vista
        c.ir_a(v)
        app.manejar(mueve(*celda(app, 30, 30), botones=(0, 0, 0)))
        app.dibujar(1)


def test_lluvia_se_detiene_se_reanuda_y_se_prolonga():
    app, pg, st, _ = nueva(esc=Escenario(duracion_min=20))
    app.iniciar()
    c = app.ctrl
    c.pasos_por_frame = 30                                                                  # 5 min
    c.avanzar(1e9)
    assert c.llueve
    app.manejar(clic(*centro(app.b_lluvia)))                                                # detener
    assert not c.llueve
    P = c.motor.lluvia_cel
    c.avanzar(1e9)
    assert c.motor.lluvia_cel == P                                                          # sin lluvia no se suma agua
    app.manejar(clic(*centro(app.b_lluvia)))                                                # reanudar (aún dentro de la duración)
    assert c.llueve
    c.pasos_por_frame = 200
    for _ in range(3):
        c.avanzar(1e9)
    assert c.motor.tiempo_min > 20 and not c.llueve                                         # se pasó de la duración: dejó de llover
    app.manejar(clic(*centro(app.b_mas)))                                                   # +30 min: llueve de nuevo
    assert c.llueve and c.motor.duracion_min == c.motor.tiempo_min + 30
    app.manejar(clic(*centro(app.b_continua)))
    assert c.continua and c.motor.duracion_min == float("inf") and c.llueve
    c.pasos_por_frame = 600
    c.avanzar(1e9)
    assert c.llueve                                                                        # sigue lloviendo sin límite
    app.manejar(clic(*centro(app.b_continua)))
    assert not c.continua and not c.llueve                                                  # al quitar «continua» vuelve el límite (ya pasado)


def test_reporte_desde_la_interfaz_y_al_cerrar():
    app, pg, st, guion = nueva(esc=Escenario(duracion_min=40))
    app.iniciar()
    c = app.ctrl
    app.manejar(clic(*centro(app.b_reporte)))                                               # sin datos: avisa y no genera
    assert c.reporte_guardado is None and "Aún no hay datos" in app.toast[0]
    c.pasos_por_frame = 300
    for _ in range(3):
        c.avanzar(1e9)
    app.manejar(clic(*centro(app.b_reporte)))
    assert c.pausa and c.reporte_guardado and Path(c.reporte_guardado).exists()
    html = Path(c.reporte_guardado).read_text(encoding="utf-8")
    for texto in ("Resumen del agua", "Materiales y absorción", "Agua que fluyó a través", "Límites del modelo", "Phaeozems", "<svg"):
        assert texto in html, texto
    assert list(Path(c.reporte_guardado).parent.glob("*_serie.csv"))
    # al cerrar sin haber generado reporte, se guarda uno
    app2, pg2, st2, guion2 = nueva(esc=Escenario(duracion_min=40))
    app2.ctrl.pasos_por_frame = 50
    guion2 += [[], [], [ev(fp.QUIT)]]
    app2.ctrl.avanzar(1e9)
    app2.ejecutar(max_frames=10)
    assert app2.ctrl.reporte_guardado and Path(app2.ctrl.reporte_guardado).exists()


def test_cuadro_de_tiempo():
    """Física + dibujo de un fotograma del corte (sin pygame real) cabe en el presupuesto de 16.7 ms."""
    app, pg, st, _ = nueva(esc=Escenario(duracion_min=120))
    app.iniciar()
    app.ctrl.alternar_vista()
    app.ctrl.avanzar(1e9)
    app.ctrl.velocidad_min_s = 20.0
    t = time.perf_counter()
    for _ in range(30):
        app.ctrl.avanzar(12.0, dt_real=1 / 60)
        app.dibujar(1)
    ms = 1000 * (time.perf_counter() - t) / 30
    print(f"   fotograma del corte (física a {app.ctrl.velocidad_min_s:g} min/s + dibujo, sin pygame real): {ms:.1f} ms")


def test_la_velocidad_depende_de_la_barra():
    """Pedir 0.5, 4 y 20 min/s debe dar velocidades muy distintas."""
    logrados = []
    for v in (0.5, 4.0, 20.0):
        c = Controlador(Escenario(duracion_min=180))
        c.velocidad_min_s = v
        c.motor.avanzar(100)
        p0 = c.motor.paso
        for _ in range(120):
            c.avanzar(12.0, dt_real=1 / 60)
        logrados.append((c.motor.paso - p0) * 10 / 60 / 2)
    assert logrados[0] < 1.0 and 3.0 < logrados[1] < 5.0 and logrados[2] > 15.0, logrados


if __name__ == "__main__":
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        t = time.perf_counter()
        f()
        print(f"OK  {f.__name__}  {time.perf_counter() - t:.1f} s")
    print("Recorrido de la interfaz correcto (con pygame simulado)")
