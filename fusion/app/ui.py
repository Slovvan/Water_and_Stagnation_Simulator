"""Interfaz pygame-ce. Lienzo lógico de 1600 x 900 (se escala a la ventana):
  barra superior (pestañas Mapa / Corte lateral / Detalle 2 m, estado y botones de ventana),
  mapa de 820 x 820, columna de resumen y gráficas en vivo, y columna de controles en tarjetas.
El módulo no importa pygame: recibe el módulo `pg` desde app/ejecutar.py (así se prueba con un pygame simulado)."""
from __future__ import annotations

import math
import time
import webbrowser
from pathlib import Path

import numpy as np

from app.corte_sync import FILAS as FILAS_CORTE
from app.estado import Controlador
from app.render import Gotas, MARCAS_LEYENDA_MM, VistaAerea, imagen_corte
from app.reporte import generar_reporte
from core.escenario import Escenario
from corte.medio import celdas_del_transecto

ANCHO_V, ALTO_V, FPS = 1600, 900, 60
MAPA_X, MAPA_Y, MAPA_L = 14, 70, 820
X2, W2 = 850, 366
X3, W3 = 1230, 356

C_FONDO, C_BARRA, C_TARJETA = (16, 22, 28), (24, 33, 41), (31, 42, 51)
C_TEXTO, C_SEC, C_TENUE = (230, 236, 240), (165, 180, 190), (110, 126, 138)
C_ACENTO, C_BORDE, C_OK, C_ALERTA = (72, 170, 224), (66, 84, 96), (255, 205, 70), (214, 82, 70)
C_SUMIDERO, C_GOTA = (232, 70, 60), (96, 160, 226)
SERIES_BALANCE = [("Lluvia", "lluvia", (96, 160, 226)), ("Infiltrado", "infiltrado", (80, 190, 130)),
                  ("Salió", "salida", (232, 150, 70)), ("Superficie", "superficie", (170, 120, 220))]
SERIES_CHARCOS = [("Área encharcada %", "area_pct", (96, 160, 226)), ("Perfiles llenos %", "lleno_pct", (232, 100, 90))]


def ajustar(fuente, texto, ancho):
    """Parte `texto` en líneas que quepan en `ancho` px."""
    out, linea = [], ""
    for palabra in texto.split(" "):
        prueba = (linea + " " + palabra).strip()
        if fuente.size(prueba)[0] <= ancho or not linea:
            linea = prueba
        else:
            out.append(linea)
            linea = palabra
    return out + [linea]


class Slider:
    def __init__(self, clave, nombre, minimo, maximo, valor, x, y, w, entero=False, vivo=True, fmt="{:.3g}", log=False, ayuda=""):
        self.clave, self.nombre, self.minimo, self.maximo, self.valor = clave, nombre, minimo, maximo, valor
        self.entero, self.vivo, self.fmt, self.log, self.ayuda = entero, vivo, fmt, log, ayuda
        self.x, self.y, self.w = x, y, w
        self.arrastrando = False

    def contiene(self, pos):
        return self.x - 4 <= pos[0] <= self.x + self.w + 4 and self.y <= pos[1] <= self.y + 36

    def mover(self, x):
        f = min(1.0, max(0.0, (x - self.x) / self.w))
        v = self.minimo * (self.maximo / self.minimo) ** f if self.log else self.minimo + f * (self.maximo - self.minimo)
        self.valor = int(round(v)) if self.entero else v

    def texto_valor(self):
        return f"{self.valor:.0f}" if self.entero else self.fmt.format(self.valor)

    def dibujar(self, pg, sup, fuente):
        sup.blit(fuente.render(self.nombre, True, C_SEC), (self.x, self.y))
        v = fuente.render(self.texto_valor(), True, C_TEXTO)
        sup.blit(v, (self.x + self.w - v.get_width(), self.y))
        pg.draw.rect(sup, C_BORDE, (self.x, self.y + 23, self.w, 4), border_radius=2)
        f = (math.log(self.valor / self.minimo) / math.log(self.maximo / self.minimo) if self.log
             else (self.valor - self.minimo) / (self.maximo - self.minimo))
        pg.draw.circle(sup, C_ACENTO, (int(self.x + f * self.w), self.y + 25), 8)


class Boton:
    def __init__(self, texto, x, y, w, h, ayuda=""):
        self.texto, self.x, self.y, self.w, self.h, self.ayuda = texto, x, y, w, h, ayuda

    def contiene(self, pos):
        return self.x <= pos[0] <= self.x + self.w and self.y <= pos[1] <= self.y + self.h

    def dibujar(self, pg, sup, fuente, activo=False, color=C_ACENTO, relleno=C_BORDE):
        pg.draw.rect(sup, color if activo else relleno, (self.x, self.y, self.w, self.h), border_radius=7)
        img = fuente.render(self.texto, True, (10, 30, 40) if activo else C_TEXTO)
        sup.blit(img, (self.x + (self.w - img.get_width()) // 2, self.y + (self.h - img.get_height()) // 2))


class Aplicacion:
    def __init__(self, pg, ctrl: Controlador | None = None):
        self.pg = pg
        self.ctrl = ctrl or Controlador()
        self.abrir_navegador = True
        self.carpeta_reportes = Path("reportes")
        e = self.ctrl.esc
        self.pend = dict(bedrock=e.bedrock)                                           # se aplica al regenerar
        # --- barra superior ---
        self.tabs = {"aerea": Boton("Mapa", 330, 11, 150, 34, "Vista aérea del terreno: charcos, suelos y la línea del corte."),
                     "corte": Boton("Corte lateral", 488, 11, 170, 34, "Sección de la línea amarilla del mapa, dibujada con la misma simulación."),
                     "detalle": Boton("Detalle 2 m", 666, 11, 150, 34, "Autómata celular de 2 m sobre la línea (motor aparte). Tecla D.")}
        # --- tarjeta Lluvia ---
        x, w = X3 + 14, W3 - 28
        self.sliders = [
            Slider("intensidad", "Intensidad (mm/h)", 0, 100, e.intensidad_mm_h, x, 102, w, ayuda="Intensidad de la lluvia. Cambia en vivo. 5 moderada · 30 fuerte · 60 violenta."),
            Slider("duracion", "Duración de la lluvia (min)", 0, 360, e.duracion_min, x, 146, w, ayuda="Minutos de lluvia desde el inicio. Se puede prolongar o dejar continua.")]
        self.b_continua = Boton("Continua", x, 196, 100, 32, "Lluvia sin límite de duración (se puede detener).")
        self.b_lluvia = Boton("Detener", x + 108, 196, 124, 32, "Detiene o reanuda la lluvia ahora mismo.")
        self.b_mas = Boton("+30 min", x + 240, 196, 88, 32, "Prolonga la lluvia 30 minutos más (y la reanuda).")
        # --- tarjeta Simulación ---
        self.sliders.append(Slider("velocidad", "Velocidad (min simulados por s)", 0.25, 60, self.ctrl.velocidad_min_s, x, 346, w, log=True,
                                   ayuda="Minutos de simulación por segundo real (0.25 a 60). Si el equipo no alcanza, se limita."))
        self.b_pausa = Boton("Pausa", x, 392, 104, 32, "Pausa o reanuda (Espacio). La simulación no se detiene sola.")
        self.b_reiniciar = Boton("Reiniciar", x + 112, 392, 104, 32, "Vuelve a t = 0 con el mismo mapa (R).")
        self.b_eje = Boton("", x, 432, 104, 30, "Línea del corte: una columna (norte-sur) o una fila (oeste-este).")
        self.b_agua = Boton("Más agua", x + 112, 432, 104, 30, "Lleva la línea a donde hay más agua ahora (L).")
        self.b_detalle = Boton("Detalle 2 m", x + 224, 432, 104, 30, "Abre o cierra el detalle de 2 m (D).")
        self.b_sumideros = Boton("", x, 470, 328, 28, "Marca con rojo las fosas del terreno que tienen más de 5 mm de agua (tecla S).")
        # --- tarjeta Terreno y suelo ---
        y = 548
        for clave, nombre, lo, hi, val, ent, ay in [
            ("seed", "Semilla del terreno", 0, 9999, e.seed, True, "Otra semilla = otro terreno."),
            ("escala", "Escala del relieve", 0.005, 0.1, e.escala, False, "Más alto = más ondulaciones pequeñas."),
            ("octavas", "Octavas", 1, 6, e.octavas, True, "Detalle fino del relieve."),
            ("persistencia", "Persistencia", 0.1, 1.0, e.persistencia, False, "Cuánto pesan las octavas finas."),
            ("amplitud", "Amplitud del relieve (m)", 0, 5, e.amplitud_m, False, "Desnivel máximo del terreno."),
            ("s0", "Saturación inicial S0", 0.0, 0.9, e.s0, False, "Humedad inicial del suelo (supuesto del dataset: 0.30).")]:
            self.sliders.append(Slider(clave, nombre, lo, hi, val, x, y, w, ent, vivo=False, ayuda=ay + " Se aplica con Regenerar."))
            y += 36
        self.b_roca = Boton("", x, y + 4, 328, 30, "Fracturada: fuga de 0.5 mm/h al llenarse el perfil. Sellada: sin fuga.")
        self.b_regen = Boton("Regenerar terreno", x, y + 40, 328, 32, "Genera un terreno nuevo con estos valores y reinicia la simulación.")
        self.b_reporte = Boton("Terminar y generar reporte", X3, 850, W3, 40, "Pausa y guarda un reporte con tablas, gráficas y el estado final.")
        # --- estado de la interfaz ---
        self.completa = False
        self._tam_ventana = None
        self._tf = (1.0, 0, 0)
        self._tam_prev = None
        self.mouse = None
        self.toast = ("", 0.0)
        self.sucio = True
        self.ultimo_clic = (0.0, None)
        self.gotas_detalle = Gotas(200, 200)
        self.gotas_corte = Gotas(100, FILAS_CORTE, n_max=260, caida=7.0, estela=6.0, seed=2)
        self._seg = np.zeros((0, 3))
        self._dt = 1.0 / FPS

    @property
    def slider(self):
        return {s.clave: s for s in self.sliders}

    # ------------------------------------------------------------------ ventana
    def _fuente(self, tam, negrita=False):
        pg = self.pg
        try:
            return pg.font.SysFont("segoeui,arial,helvetica,dejavusans,liberationsans", tam, negrita)
        except Exception:
            return pg.font.Font(None, tam + 5)

    def _tam_inicial(self):
        pg = self.pg
        try:
            w, h = pg.display.get_desktop_sizes()[0]
        except Exception:
            info = pg.display.Info()
            w, h = info.current_w, info.current_h
        k = min(1.0, 0.95 * w / ANCHO_V, 0.88 * h / ALTO_V)
        return max(int(ANCHO_V * k), 800), max(int(ALTO_V * k), 450)

    def iniciar(self):
        pg = self.pg
        pg.init()
        self.lienzo = pg.Surface((ANCHO_V, ALTO_V))
        self.pantalla = self.lienzo
        self.mapa_surf = pg.Surface((MAPA_L, MAPA_L))
        self.ventana = pg.display.set_mode(self._tam_inicial(), pg.RESIZABLE)
        pg.display.set_caption("Simulador de encharcamiento 2D: mapa, corte lateral y detalle")
        self.f_txt, self.f_peq, self.f_tit = self._fuente(17), self._fuente(15), self._fuente(24, True)
        self.f_card = self._fuente(16, True)
        self.reloj = pg.time.Clock()

    def alternar_pantalla_completa(self):
        pg = self.pg
        if not self.completa:
            self._tam_ventana = pg.display.get_surface().get_size()
            try:
                tam = pg.display.get_desktop_sizes()[0]
            except Exception:
                info = pg.display.Info()
                tam = (info.current_w, info.current_h)
            pg.display.set_mode(tam, pg.FULLSCREEN)
        else:
            pg.display.set_mode(self._tam_ventana or self._tam_inicial(), pg.RESIZABLE)
        self.completa = not self.completa
        self.sucio = True

    def _mapear(self, pos):
        k, ox, oy = self._tf
        return ((pos[0] - ox) / k, (pos[1] - oy) / k)

    def _presentar(self):
        pg = self.pg
        ventana = pg.display.get_surface()
        ww, wh = ventana.get_size()
        k = min(ww / ANCHO_V, wh / ALTO_V)
        if (ww, wh) == (ANCHO_V, ALTO_V):
            ventana.blit(self.lienzo, (0, 0))
            self._tf = (1.0, 0, 0)
        else:
            tam = (max(int(ANCHO_V * k), 1), max(int(ALTO_V * k), 1))
            img = pg.transform.smoothscale(self.lienzo, tam)
            ox, oy = (ww - tam[0]) // 2, (wh - tam[1]) // 2
            ventana.fill((0, 0, 0))
            ventana.blit(img, (ox, oy))
            self._tf = (k, ox, oy)
        self._tam_prev = (ww, wh)

    # ------------------------------------------------------------------ acciones
    def _avisar(self, texto, seg=7.0):
        self.toast = (texto, time.perf_counter() + seg)

    def generar_reporte(self, abrir=True):
        c = self.ctrl
        if c.motor.paso == 0:
            self._avisar("Aún no hay datos: deja correr la simulación antes de generar el reporte.")
            return None
        c.pausa = True
        rutas = generar_reporte(c, self.carpeta_reportes)
        c.reporte_guardado = rutas["html"]
        self._avisar(f"Reporte guardado en {rutas['html']}  (simulación en pausa)", 10.0)
        if abrir and self.abrir_navegador:
            try:
                webbrowser.open(Path(rutas["html"]).resolve().as_uri())
            except Exception:
                pass
        return rutas

    def _aplicar_slider(self, s):
        c = self.ctrl
        if s.clave == "intensidad":
            c.set_intensidad(s.valor)
        elif s.clave == "duracion":
            c.set_duracion(s.valor)
        elif s.clave == "velocidad":
            c.velocidad_min_s = float(s.valor)

    def regenerar(self):
        v = {s.clave: s.valor for s in self.sliders}
        self.ctrl.regenerar(seed=int(v["seed"]), escala=v["escala"], octavas=int(v["octavas"]),
                            persistencia=v["persistencia"], amplitud_m=v["amplitud"], s0=v["s0"],
                            intensidad_mm_h=v["intensidad"], duracion_min=v["duracion"], bordes_abiertos=False, **self.pend)

    def _clic_mapa(self, pos):
        c = self.ctrl
        if c.vista != "aerea":
            return
        k = MAPA_L / c.mapa.n
        fila, col = int((pos[1] - MAPA_Y) // k), int((pos[0] - MAPA_X) // k)
        ahora = time.perf_counter()
        t_ant, cel_ant = self.ultimo_clic
        c.seleccionar(fila, col)
        if cel_ant == c.seleccion and ahora - t_ant < 0.45:
            c.abrir_corte()
        self.ultimo_clic = (ahora, c.seleccion)

    def _en_mapa(self, pos):
        return MAPA_X <= pos[0] < MAPA_X + MAPA_L and MAPA_Y <= pos[1] < MAPA_Y + MAPA_L

    def _botones_clic(self):
        c = self.ctrl
        acciones = [(self.b_continua, lambda: c.set_continua(not c.continua)), (self.b_lluvia, c.alternar_lluvia),
                    (self.b_mas, lambda: c.prolongar_lluvia(30.0)), (self.b_pausa, self._pausar),
                    (self.b_reiniciar, c.reiniciar), (self.b_eje, c.alternar_eje), (self.b_agua, c.linea_mas_agua),
                    (self.b_detalle, c.alternar_detalle), (self.b_regen, self.regenerar), (self.b_reporte, self.generar_reporte),
                    (self.b_sumideros, self._alternar_sumideros),
                    (self.b_roca, lambda: self.pend.update(bedrock="sealed" if self.pend["bedrock"] == "fractured" else "fractured"))]
        for nombre, b in self.tabs.items():
            acciones.append((b, (lambda n=nombre: c.ir_a(n))))
        return acciones

    def _alternar_sumideros(self):
        self.ctrl.mostrar_sumideros = not self.ctrl.mostrar_sumideros

    def _pausar(self):
        self.ctrl.pausa = not self.ctrl.pausa

    def manejar(self, ev) -> bool:
        """Procesa un evento. Devuelve False si hay que cerrar la aplicación."""
        pg, c = self.pg, self.ctrl
        if ev.type == pg.QUIT:
            return False
        if ev.type == pg.KEYDOWN:
            if ev.key == pg.K_TAB:
                c.alternar_vista()
            elif ev.key == pg.K_d:
                c.alternar_detalle()
            elif ev.key == pg.K_l:
                c.linea_mas_agua()
            elif ev.key == pg.K_f:
                self.generar_reporte()
            elif ev.key == pg.K_s:
                self._alternar_sumideros()
            elif ev.key == pg.K_F11:
                self.alternar_pantalla_completa()
            elif ev.key == pg.K_SPACE:
                c.pausa = not c.pausa
            elif ev.key == pg.K_r:
                c.reiniciar()
            elif ev.key == pg.K_ESCAPE:
                c.volver()
            elif c.vista != "detalle":
                mas, menos = (pg.K_RIGHT, pg.K_LEFT) if c.eje == "col" else (pg.K_DOWN, pg.K_UP)
                if ev.key == mas:
                    c.mover_linea(1)
                elif ev.key == menos:
                    c.mover_linea(-1)
        elif ev.type == pg.MOUSEBUTTONDOWN and getattr(ev, "button", 1) == 1:
            pos = self._mapear(ev.pos)
            self.mouse = pos
            for b, accion in self._botones_clic():
                if b.contiene(pos):
                    accion()
                    break
            else:
                if self._en_mapa(pos):
                    self._clic_mapa(pos)
                else:
                    for s in self.sliders:
                        if s.contiene(pos):
                            s.arrastrando = True
                            s.mover(pos[0])
                            if s.vivo:
                                self._aplicar_slider(s)
        elif ev.type == pg.MOUSEBUTTONUP:
            for s in self.sliders:
                s.arrastrando = False
        elif ev.type == pg.MOUSEMOTION:
            self.mouse = self._mapear(ev.pos)
            for s in self.sliders:
                if s.arrastrando:
                    s.mover(self.mouse[0])
                    if s.vivo:
                        self._aplicar_slider(s)
        self.sucio = True
        return True

    # ------------------------------------------------------------------ dibujo: auxiliares
    def _superficie(self, img, tam):
        pg = self.pg
        chica = pg.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        return pg.transform.scale(chica, tam)

    def _tarjeta(self, x, y, w, h, titulo):
        pg, p = self.pg, self.pantalla
        pg.draw.rect(p, C_TARJETA, (x, y, w, h), border_radius=10)
        p.blit(self.f_card.render(titulo, True, C_ACENTO), (x + 14, y + 8))

    def _texto(self, sup, txt, pos, color=C_TEXTO, fuente=None):
        sup.blit((fuente or self.f_peq).render(txt, True, color), pos)

    def _grafica(self, x, y, w, h, titulo, claves, ylab):
        """Gráfica de líneas en vivo con la serie temporal del controlador."""
        pg, p, S = self.pg, self.pantalla, self.ctrl.serie
        self._tarjeta(x, y, w, h, titulo)
        gx, gy, gw, gh = x + 44, y + 52, w - 60, h - 86
        pg.draw.rect(p, C_FONDO, (gx, gy, gw, gh), border_radius=4)
        lx = x + 14
        for nombre, _, col in claves:
            pg.draw.rect(p, col, (lx, y + 34, 10, 4))
            img = self.f_peq.render(nombre, True, C_SEC)
            p.blit(img, (lx + 14, y + 29))
            lx += 26 + img.get_width()
        if len(S) < 2:
            self._texto(p, "La gráfica aparece cuando corre la simulación", (gx + 10, gy + gh // 2 - 8), C_TENUE)
            return
        paso = max(len(S) // 240, 1)
        datos = S[::paso] + ([S[-1]] if (len(S) - 1) % paso else [])
        t0, t1 = datos[0]["t"], datos[-1]["t"]
        ymax = max(max(d[k] for d in datos) for _, k, _ in claves)
        ymax = ymax * 1.1 if ymax > 0 else 1.0
        for _, k, col in claves:
            pts = [(gx + (d["t"] - t0) / max(t1 - t0, 1e-9) * gw, gy + gh - d[k] / ymax * gh) for d in datos]
            pg.draw.lines(p, col, False, pts, 2)
        self._texto(p, f"{ymax:.3g}", (x + 8, gy - 4), C_TENUE)
        self._texto(p, "0", (x + 30, gy + gh - 12), C_TENUE)
        self._texto(p, f"{t0:.0f}", (gx, gy + gh + 6), C_TENUE)
        self._texto(p, f"{t1:.0f} min", (gx + gw - 46, gy + gh + 6), C_TENUE)
        self._texto(p, ylab, (x + 8, y + h - 20), C_TENUE)

    # ------------------------------------------------------------------ dibujo: mapa y vistas
    def _linea_en_mapa(self, sup, esc, color=C_OK, grosor=3):
        c = self.ctrl
        m = c.indice * esc + esc / 2
        a, b = ((m, 0), (m, MAPA_L)) if c.eje == "col" else ((0, m), (MAPA_L, m))
        self.pg.draw.line(sup, color, a, b, grosor)

    def _mapa_aereo(self, sup):
        pg, c = self.pg, self.ctrl
        va = c.vista_aerea
        sup.blit(self._superficie(va.imagen(c.motor.H, suavizar=False), (MAPA_L, MAPA_L)), (0, 0))
        esc = MAPA_L / c.mapa.n
        if c.mostrar_sumideros:
            for f, col in va.sumideros(c.motor.H):
                pg.draw.rect(sup, C_SUMIDERO, (int(col * esc), int(f * esc), int(esc), int(esc)), 1)
        self._linea_en_mapa(sup, esc)
        f, col = c.seleccion
        pg.draw.circle(sup, C_OK, (int((col + .5) * esc), int((f + .5) * esc)), 5)
        velo = pg.Surface((92, 176), pg.SRCALPHA)
        velo.fill((10, 16, 22, 150))
        sup.blit(velo, (MAPA_L - 98, MAPA_L - 200))
        barra = VistaAerea.barra_leyenda(150)
        sup.blit(self._superficie(np.repeat(barra, 18, axis=1), (18, 150)), (MAPA_L - 34, MAPA_L - 190))
        for v in MARCAS_LEYENDA_MM:
            self._texto(sup, f"{v} mm", (MAPA_L - 84, MAPA_L - 190 + 150 - int(min(v, 100) / 100 * 150) - 6), C_TEXTO)
        self._rotulo(sup, f"Mapa aéreo · {c.mapa.n * c.mapa.dx_m:.0f} m × {c.mapa.n * c.mapa.dx_m:.0f} m",
                     f"Línea del corte: {c.nombre_linea}. Clic para elegirla · flechas para moverla · doble clic o Tab para ver el corte")

    def _mapa_corte(self, sup, avance):
        pg, c = self.pg, self.ctrl
        cs = c.corte
        img, sup_y = cs.imagen(c.motor, suavizar=True)
        sup.blit(self._superficie(img, (MAPA_L, MAPA_L)), (0, 0))
        kx, ky = MAPA_L / cs.n, MAPA_L / FILAS_CORTE
        if avance:
            self._seg = self.gotas_corte.segmentos(c.motor.intensidad_mm_h if c.motor.llueve else 0.0, sup_y)
        for x, y0, y1 in self._seg:
            pg.draw.line(sup, C_GOTA, ((x + 0.5) * kx, (y0 + 0.5) * ky), ((x + 0.5) * kx, (y1 + 0.5) * ky), 2)
        px_m = MAPA_L / (cs.top - cs.bottom)
        pg.draw.line(sup, (240, 244, 248), (MAPA_L - 20, MAPA_L - 36 - px_m), (MAPA_L - 20, MAPA_L - 36), 3)
        self._texto(sup, "1 m", (MAPA_L - 54, MAPA_L - 36 - px_m / 2 - 6), (240, 244, 248))
        izq, der = ("N", "S") if c.eje == "col" else ("O", "E")
        self._texto(sup, f"{izq}  0 m", (46, MAPA_L - 26), (20, 30, 40))
        self._texto(sup, f"{c.mapa.n * c.mapa.dx_m:.0f} m  {der}", (MAPA_L - 100, MAPA_L - 26), (20, 30, 40))
        self._minimapa(sup)
        self._leyenda_materiales(sup, cs.leyenda)
        self._rotulo(sup, f"Corte lateral · {c.nombre_linea}", f"Dibujado con el mismo motor del mapa · altura ×{cs.exageracion:.0f} · pasa el ratón sobre un material",
                     oscuro=True)

    def _mapa_detalle(self, sup, avance):
        pg, c = self.pg, self.ctrl
        lu = c.lupa
        sup.blit(self._superficie(imagen_corte(lu.sim, lu.medio), (MAPA_L, MAPA_L)), (0, 0))
        k = MAPA_L / lu.sim.grid_size
        if avance:
            self._seg = self.gotas_detalle.segmentos(lu.sim.rain_intensity_mm_h if lu.llueve else 0.0, lu.sim.surface_y())
        for x, y0, y1 in self._seg:
            pg.draw.line(sup, C_GOTA, ((x + 0.5) * k, (y0 + 0.5) * k), ((x + 0.5) * k, (y1 + 0.5) * k), 2)
        self._minimapa(sup, lu.info["celdas"])
        self._leyenda_materiales(sup, [(n, rgb) for n, rgb in lu.medio.legend()][:9])
        self._rotulo(sup, "Detalle de 2 m · autómata celular (motor aparte)", "Parte del estado del mapa al abrirse · Esc o D para volver", oscuro=True)

    def _rotulo(self, sup, titulo, ayuda, oscuro=False):
        col = (20, 30, 40) if oscuro else C_TEXTO
        sup.blit(self.f_card.render(titulo, True, col), (12, 8))
        sup.blit(self.f_peq.render(ayuda, True, col), (12, 30))

    def _minimapa(self, sup, ventana=None):
        pg, c = self.pg, self.ctrl
        x0, y0, L = MAPA_L - 134, 12, 122
        sup.blit(self._superficie(c.vista_aerea.imagen(c.motor.H, suavizar=False), (L, L)), (x0, y0))
        pg.draw.rect(sup, C_TEXTO, (x0, y0, L, L), 1)
        k = L / c.mapa.n
        if ventana is None:
            m = c.indice * k + k / 2
            a, b = ((x0 + m, y0), (x0 + m, y0 + L)) if c.eje == "col" else ((x0, y0 + m), (x0 + L, y0 + m))
            pg.draw.line(sup, C_OK, a, b, 2)
        else:
            a, b = ventana
            f0, f1, c0, c1 = min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])
            pg.draw.rect(sup, C_OK, (x0 + c0 * k - 1, y0 + f0 * k - 1, (c1 - c0 + 1) * k + 2, (f1 - f0 + 1) * k + 2), 2)

    def _leyenda_materiales(self, sup, items):
        pg = self.pg
        alto = 17 * len(items) + 10
        velo = pg.Surface((300, alto), pg.SRCALPHA)
        velo.fill((10, 16, 22, 170))
        sup.blit(velo, (10, MAPA_L - alto - 46))
        for i, (nombre, rgb) in enumerate(items):
            y = MAPA_L - alto - 41 + i * 17
            pg.draw.rect(sup, tuple(int(255 * v) for v in rgb), (18, y + 1, 12, 12))
            sup.blit(self.f_peq.render(nombre, True, (238, 242, 246)), (38, y))

    # ------------------------------------------------------------------ dibujo: barra, columnas, popups
    def _dibujar_barra(self):
        pg, c, p = self.pg, self.ctrl, self.pantalla
        pg.draw.rect(p, C_BARRA, (0, 0, ANCHO_V, 56))
        p.blit(self.f_tit.render("Encharcamiento 2D", True, C_TEXTO), (16, 14))
        for nombre, b in self.tabs.items():
            b.dibujar(pg, p, self.f_txt, c.vista == nombre)
        eng = c.activo
        estado = "llueve" if c.llueve else "sin lluvia"
        i = (eng.sim.rain_intensity_mm_h if c.vista == "detalle" else eng.intensidad_mm_h)
        pausa = "  ·  EN PAUSA" if c.pausa else ""
        p.blit(self.f_txt.render(f"t = {eng.tiempo_min:.0f} min  ·  {estado} ({i:g} mm/h)  ·  {c.vel_real:.1f} min/s{pausa}", True,
                                 C_OK if c.pausa else C_SEC), (846, 18))
        p.blit(self.f_peq.render("F11: pantalla completa  ·  Espacio: pausa  ·  F: reporte", True, C_TENUE), (ANCHO_V - 380, 20))

    def _dibujar_columna_resumen(self):
        c, p = self.ctrl, self.pantalla
        self._tarjeta(X2, 70, W2, 330, "Resumen")
        y = 100
        for linea in c.resumen():
            for parte in ajustar(self.f_peq, linea, W2 - 28):
                if y > 70 + 330 - 20:
                    break
                p.blit(self.f_peq.render(parte, True, C_TEXTO if parte is not None else C_SEC), (X2 + 14, y))
                y += 17
        self._grafica(X2, 410, W2, 240, "Balance de agua (mm sobre el mapa)", SERIES_BALANCE, "mm")
        self._grafica(X2, 660, W2, 230, "Charcos y perfiles llenos", SERIES_CHARCOS, "%")

    def _dibujar_columna_controles(self):
        pg, c, p = self.pg, self.ctrl, self.pantalla
        sl = self.slider
        self._tarjeta(X3, 70, W3, 232, "Lluvia")
        for k in ("intensidad", "duracion"):
            sl[k].dibujar(pg, p, self.f_peq)
        self.b_continua.dibujar(pg, p, self.f_peq, c.continua)
        self.b_lluvia.texto = "Detener" if c.llueve else "Reanudar"
        self.b_lluvia.dibujar(pg, p, self.f_peq, not c.llueve, C_OK)
        self.b_mas.dibujar(pg, p, self.f_peq)
        self._texto(p, "Dura hasta el minuto " + ("∞ (continua)" if c.continua or not np.isfinite(c.motor.duracion_min) else f"{c.motor.duracion_min:.0f}")
                    + ("" if c.motor.lluvia_activa else "  ·  detenida"), (X3 + 14, 240), C_TENUE)
        self._tarjeta(X3, 312, W3, 190, "Simulación")
        sl["velocidad"].dibujar(pg, p, self.f_peq)
        self.b_pausa.texto = "Reanudar" if c.pausa else "Pausa"
        self.b_pausa.dibujar(pg, p, self.f_peq, c.pausa, C_OK)
        self.b_reiniciar.dibujar(pg, p, self.f_peq)
        self.b_eje.texto = f"Línea: {'columna' if c.eje == 'col' else 'fila'}"
        self.b_eje.dibujar(pg, p, self.f_peq)
        self.b_agua.dibujar(pg, p, self.f_peq)
        self.b_detalle.texto = "Volver" if c.vista == "detalle" else "Detalle 2 m"
        self.b_detalle.dibujar(pg, p, self.f_peq, c.vista == "detalle")
        self.b_sumideros.texto = f"Marcar sumideros (S): {'sí' if c.mostrar_sumideros else 'no'}"
        self.b_sumideros.dibujar(pg, p, self.f_peq, c.mostrar_sumideros)
        self._tarjeta(X3, 512, W3, 330, "Terreno y suelo  (con Regenerar)")
        for k in ("seed", "escala", "octavas", "persistencia", "amplitud", "s0"):
            sl[k].dibujar(pg, p, self.f_peq)
        self.b_roca.texto = f"Roca del fondo: {'fracturada (con fuga)' if self.pend['bedrock'] == 'fractured' else 'sellada'}"
        self.b_roca.dibujar(pg, p, self.f_peq)
        self.b_regen.dibujar(pg, p, self.f_txt, True)
        self.b_reporte.dibujar(pg, p, self.f_txt, True, (96, 190, 140))

    def _info_hover(self):
        """Líneas del popup según lo que haya bajo el ratón (material en el mapa o ayuda de un control)."""
        c, m = self.ctrl, self.mouse
        if m is None:
            return None
        if self._en_mapa(m):
            rx, ry = m[0] - MAPA_X, m[1] - MAPA_Y
            if c.vista == "aerea":
                k = MAPA_L / c.mapa.n
                return c.info_aereo(int(ry // k), int(rx // k))
            if c.vista == "corte":
                cs = c.corte
                return cs.info(c.motor, int(rx // (MAPA_L / cs.n)), int(ry // (MAPA_L / FILAS_CORTE)))
            k = MAPA_L / 200
            return c.info_detalle(int(ry // k), int(rx // k))
        for b in [*self.tabs.values(), self.b_sumideros, self.b_continua, self.b_lluvia, self.b_mas, self.b_pausa,
                  self.b_reiniciar, self.b_eje, self.b_agua, self.b_detalle, self.b_roca, self.b_regen, self.b_reporte]:
            if b.ayuda and b.contiene(m):
                return ajustar(self.f_peq, b.ayuda, 300)
        for s in self.sliders:
            if s.ayuda and s.contiene(m):
                return ajustar(self.f_peq, s.ayuda, 300)
        return None

    def _dibujar_popup(self, lineas):
        pg, p, m = self.pg, self.pantalla, self.mouse
        imgs = [self.f_peq.render(lin, True, C_TEXTO if i else C_OK) for i, lin in enumerate(lineas)]
        w = max(i.get_width() for i in imgs) + 20
        h = 17 * len(imgs) + 14
        x = m[0] + 18 if m[0] + 18 + w < ANCHO_V else m[0] - w - 12
        y = m[1] + 18 if m[1] + 18 + h < ALTO_V else m[1] - h - 12
        velo = pg.Surface((w, h), pg.SRCALPHA)
        velo.fill((12, 18, 24, 238))
        p.blit(velo, (x, y))
        pg.draw.rect(p, C_ACENTO, (x, y, w, h), 1, border_radius=4)
        for i, img in enumerate(imgs):
            p.blit(img, (x + 10, y + 7 + 17 * i))

    def _dibujar_toast(self):
        texto, hasta = self.toast
        if texto and time.perf_counter() < hasta:
            img = self.f_txt.render(texto, True, (10, 30, 40))
            w, h = img.get_width() + 28, img.get_height() + 16
            x, y = MAPA_X + (MAPA_L - w) // 2, MAPA_Y + MAPA_L - h - 12
            self.pg.draw.rect(self.pantalla, (96, 190, 140), (x, y, w, h), border_radius=8)
            self.pantalla.blit(img, (x + 14, y + 8))

    def dibujar(self, pasos: int = 0):
        p = self.pantalla
        p.fill(C_FONDO)
        c = self.ctrl
        c.vista_aerea.suavizar(c.motor.H)                                   # una sola vez por fotograma
        sup = self.mapa_surf
        if c.vista == "aerea":
            self._mapa_aereo(sup)
        elif c.vista == "corte":
            self._mapa_corte(sup, pasos)
        else:
            self._mapa_detalle(sup, pasos)
        p.blit(sup, (MAPA_X, MAPA_Y))
        self.pg.draw.rect(p, C_BORDE, (MAPA_X - 1, MAPA_Y - 1, MAPA_L + 2, MAPA_L + 2), 1)
        self._dibujar_barra()
        self._dibujar_columna_resumen()
        self._dibujar_columna_controles()
        self._dibujar_toast()
        info = self._info_hover()
        if info:
            self._dibujar_popup(info)
        self._presentar()
        self.pg.display.flip()
        self.sucio = False

    # ------------------------------------------------------------------ bucle
    def _hay_lluvia_animada(self) -> bool:
        c = self.ctrl
        if c.pausa or c.vista == "aerea":
            return False
        return c.llueve and (c.motor.intensidad_mm_h > 0 if c.vista == "corte" else c.lupa.sim.rain_intensity_mm_h > 0)

    def ejecutar(self, max_frames: int | None = None):
        self.iniciar()
        frames, activo = 0, True
        while activo:
            for ev in self.pg.event.get():
                if not self.manejar(ev):
                    activo = False
            if not activo:
                break
            pasos = self.ctrl.avanzar(presupuesto_ms=12.0, dt_real=self._dt)
            if self.pg.display.get_surface().get_size() != self._tam_prev:
                self.sucio = True
            anim = self._hay_lluvia_animada()
            if pasos or anim or self.sucio or (self.toast[0] and time.perf_counter() < self.toast[1] + 0.2):
                self.dibujar(pasos or int(anim))
            self._dt = self.reloj.tick(FPS) / 1000.0
            frames += 1
            if max_frames is not None and frames >= max_frames:
                break
        self._al_salir()
        self.pg.quit()

    def _al_salir(self):
        """Si hubo simulación y no se generó un reporte, se guarda uno (sin abrir el navegador)."""
        c = self.ctrl
        if c.motor.paso > 0 and c.reporte_guardado is None:
            try:
                rutas = generar_reporte(c, self.carpeta_reportes)
                c.reporte_guardado = rutas["html"]
                print(f"Reporte guardado en {rutas['html']}")
            except Exception as exc:                                  # nunca impide cerrar
                print(f"No se pudo guardar el reporte: {exc}")
