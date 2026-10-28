"""Controlador de la aplicación (sin pygame). Una sola simulación (el motor aéreo) y tres vistas:
  aerea   el mapa;
  corte   la vista lateral SINCRONIZADA de una línea del mapa (por defecto el lado derecho): dibuja el mismo motor;
  detalle la lupa de 2 m con el autómata del corte (motor aparte, se inicializa desde el mapa al abrirla).
"""
from __future__ import annotations

import time

import numpy as np

from aereo.motor import MotorAereo
from app.corte_sync import CorteSincronizado
from app.render import VistaAerea
from core.escenario import Escenario
from core.mapa import generar_mapa, resumen_mapa
from corte.lupa import Lupa
from corte.medio import _perfil
from water.soil_data import AGS, CONFIANZA_ASIGNACION
from water.soil_terrain import get_material
from water.units import classify_rain

MAX_PASOS_FRAME = 60
REGISTRAR_CADA = 6            # pasos entre muestras de la serie temporal (1 min simulado con dt = 10 s)


def _fmt_min(m):
    return "no se produce" if m is None or not np.isfinite(m) else f"{m:.1f} min"


class Controlador:
    def __init__(self, esc: Escenario | None = None):
        self.esc = esc or Escenario()
        self.pasos_por_frame = 2          # modo por pasos (pruebas): pasos pedidos por fotograma
        self.velocidad_min_s = 5.0        # modo de la interfaz: minutos simulados por segundo real que se piden
        self._acum_s = 0.0
        self.vel_real = 0.0               # minutos simulados por segundo real que se logran (suavizado)
        self.usar_aporte = True
        self.pausa = False
        self.eje = "col"                  # la línea del corte: una columna (N->S) o una fila (O->E)
        self.continua = False             # lluvia continua (sin límite de duración)
        self.mostrar_sumideros = False    # marcas rojas de las fosas con agua (apagado por defecto)
        self.reconstruir()

    # ---- construcción ------------------------------------------------------
    def reconstruir(self):
        """Mapa nuevo + simulación desde cero (cambió el terreno, el suelo, S0 o las fronteras)."""
        self.mapa = generar_mapa(self.esc)
        self.resumen_suelos = resumen_mapa(self.mapa)
        self.vista_aerea = VistaAerea(self.mapa)
        self.indice = self.mapa.n - 1     # por defecto, el lado derecho del mapa
        self.seleccion = (self.mapa.n // 2, self.mapa.n - 1)
        self.reiniciar()

    def reiniciar(self):
        """Misma topografía, simulación desde t = 0."""
        self.motor = MotorAereo(self.mapa, self.esc)
        if self.continua:
            self.motor.set_duracion(float("inf"))
        self.lupa = None
        self.vista = "aerea"
        self._corte = None
        self.serie = []                   # muestras en el tiempo (para las gráficas y el reporte)
        if hasattr(self, "vista_aerea"):
            self.vista_aerea.reiniciar()
        self.reporte_guardado = None
        self.version = getattr(self, "version", 0) + 1

    def regenerar(self, **cambios):
        self.esc = self.esc.con(**cambios)
        self.reconstruir()

    # ---- controles en vivo -------------------------------------------------
    def set_intensidad(self, mm_h: float):
        self.esc = self.esc.con(intensidad_mm_h=float(mm_h))
        self.motor.set_intensidad(mm_h)
        if self.lupa:
            self.lupa.set_intensidad(mm_h)

    def set_duracion(self, minutos: float):
        self.esc = self.esc.con(duracion_min=float(minutos))
        if not self.continua:
            self.motor.set_duracion(minutos)
            if self.lupa:
                self.lupa.set_duracion(minutos)

    def set_continua(self, continua: bool):
        """Lluvia continua: no se corta por duración (se puede detener con alternar_lluvia)."""
        self.continua = bool(continua)
        d = float("inf") if self.continua else self.esc.duracion_min
        self.motor.set_duracion(d)
        if self.lupa:
            self.lupa.set_duracion(d)

    @property
    def llueve(self) -> bool:
        return self.lupa.llueve if self.vista == "detalle" else self.motor.llueve

    def alternar_lluvia(self):
        """Detiene la lluvia ahora, o la reanuda (sin límite si ya había pasado la duración)."""
        if self.llueve:
            self.motor.detener_lluvia()
            if self.lupa:
                self.lupa.set_lluvia_activa(False)
        else:
            self.motor.reanudar_lluvia()
            if self.lupa:
                self.lupa.set_lluvia_activa(True)

    def prolongar_lluvia(self, minutos: float = 30.0):
        """Alarga la lluvia `minutos` más (y la reanuda si estaba detenida)."""
        self.motor.prolongar(minutos)
        if self.lupa:
            fin = self.lupa.tiempo_min + minutos
            self.lupa.set_duracion(float("inf") if not np.isfinite(self.motor.duracion_min) else fin)

    # ---- línea del corte ---------------------------------------------------
    @property
    def corte(self) -> CorteSincronizado:
        clave = (id(self.mapa), self.eje, self.indice, self.esc.s0)
        if self._corte is None or self._corte[0] != clave:
            self._corte = (clave, CorteSincronizado(self.mapa, self.esc, self.eje, self.indice))
        return self._corte[1]

    def seleccionar(self, fila: int, col: int):
        n = self.mapa.n
        self.seleccion = (int(np.clip(fila, 0, n - 1)), int(np.clip(col, 0, n - 1)))
        self.indice = self.seleccion[1] if self.eje == "col" else self.seleccion[0]

    def mover_linea(self, delta: int):
        self.indice = int(np.clip(self.indice + delta, 0, self.mapa.n - 1))
        f, c = self.seleccion
        self.seleccion = (f, self.indice) if self.eje == "col" else (self.indice, c)

    def alternar_eje(self):
        self.eje = "fila" if self.eje == "col" else "col"
        f, c = self.seleccion
        self.indice = c if self.eje == "col" else f

    @property
    def nombre_linea(self) -> str:
        n = self.mapa.n
        if self.eje == "col":
            lado = " (lado derecho)" if self.indice == n - 1 else " (lado izquierdo)" if self.indice == 0 else ""
            return f"columna {self.indice}{lado}, norte -> sur"
        lado = " (borde superior)" if self.indice == 0 else " (borde inferior)" if self.indice == n - 1 else ""
        return f"fila {self.indice}{lado}, oeste -> este"

    # ---- vistas ------------------------------------------------------------
    def abrir_corte(self):
        self.lupa, self.vista = None, "corte"

    def abrir_detalle(self):
        d = "S" if self.eje == "col" else "E"
        self.lupa = Lupa(self.motor, *self.seleccion, d, usar_aporte=self.usar_aporte)
        self.vista = "detalle"

    def volver(self):
        if self.vista == "detalle":
            self.lupa, self.vista = None, "corte"
        else:
            self.vista = "aerea"

    def ir_a(self, vista: str):
        """Pestañas: 'aerea' (mapa), 'corte' (lateral sincronizado) o 'detalle' (autómata de 2 m)."""
        if vista == "detalle":
            if self.vista != "detalle":
                self.abrir_detalle()
        elif vista == "corte":
            self.abrir_corte()
        else:
            self.lupa, self.vista = None, "aerea"

    def alternar_vista(self):
        """El botón principal: mapa <-> corte lateral (desde el detalle regresa al corte)."""
        if self.vista == "aerea":
            self.abrir_corte()
        elif self.vista == "corte":
            self.vista = "aerea"
        else:
            self.volver()

    def alternar_detalle(self):
        self.volver() if self.vista == "detalle" else self.abrir_detalle()

    @property
    def activo(self):
        return self.lupa if self.vista == "detalle" else self.motor

    @property
    def tiempo_min(self) -> float:
        return self.activo.tiempo_min

    # ---- tiempo ------------------------------------------------------------
    def avanzar(self, presupuesto_ms: float = 12.0, dt_real: float | None = None) -> int:
        """Avanza el motor de la vista activa (en 'aerea' y 'corte' es el mismo).
        Con dt_real (segundos reales desde el fotograma anterior) el avance es por TIEMPO: velocidad_min_s minutos
        simulados por segundo real, con un acumulador (a velocidades bajas algunos fotogramas no avanzan y otros avanzan un
        paso). Sin dt_real avanza `pasos_por_frame` pasos (modo de pruebas). En ambos casos se detiene al agotar el
        presupuesto de tiempo del fotograma: si no alcanza, se descarta lo pendiente y vel_real muestra lo logrado."""
        eng = self.activo
        dt_sim = self.esc.dt_s
        if self.pausa:
            self.vel_real = 0.9 * self.vel_real
            return 0
        if dt_real is None:
            pedidos = max(int(self.pasos_por_frame), 0)
        else:
            self._acum_s += self.velocidad_min_s * 60.0 * min(max(dt_real, 0.0), 0.1)
            pedidos = int(self._acum_s // dt_sim)
            self._acum_s -= pedidos * dt_sim
        t0, hechos = time.perf_counter(), 0
        while hechos < pedidos:
            eng.step()
            hechos += 1
            if eng is self.motor and self.motor.paso % REGISTRAR_CADA == 0:
                self._registrar()
            if (time.perf_counter() - t0) * 1000.0 >= presupuesto_ms:
                break
        if dt_real is not None:
            if hechos < pedidos:
                self._acum_s = 0.0                                     # el equipo no alcanza: no se acumula deuda
            ref = max(dt_real, 1e-3)
            self.vel_real = 0.9 * self.vel_real + 0.1 * (hechos * dt_sim / 60.0 / ref)
        return hechos

    def _registrar(self):
        mo, n2 = self.motor, self.mapa.n ** 2
        k = len(self.mapa.codes)
        cnt = np.bincount(self.mapa.soil_id.ravel(), minlength=k).clip(1)
        F_g = np.bincount(self.mapa.soil_id.ravel(), weights=mo.F.ravel(), minlength=k) / cnt
        self.serie.append({
            "t": mo.tiempo_min, "lluvia": mo.lluvia_cel / n2, "infiltrado": float(mo.F.mean()), "fuga": float(mo.D.mean()),
            "superficie": float(mo.H.mean()), "salida": mo.salida_cel / n2,
            "intensidad": mo.intensidad_mm_h if mo.llueve else 0.0, "prof_max": float(mo.H.max()),
            "area_pct": 100.0 * float((mo.H > 1.0).mean()), "lleno_pct": 100.0 * float(((mo.F - mo.D) >= mo.cap - 1e-6).mean()),
            "F_grupo": F_g.tolist()})

    # ---- atajo: línea con más agua ------------------------------------------
    def linea_mas_agua(self):
        """Mueve la línea del corte a la columna (o fila) que más agua superficial tiene ahora."""
        H = self.motor.H
        tot = H.sum(axis=0) if self.eje == "col" else H.sum(axis=1)
        self.indice = int(np.argmax(tot))
        f, c = self.seleccion
        self.seleccion = (f, self.indice) if self.eje == "col" else (self.indice, c)

    # ---- información al pasar el ratón ------------------------------------------
    def info_aereo(self, r: int, c: int) -> list[str]:
        m, mo = self.mapa, self.motor
        r, c = int(np.clip(r, 0, m.n - 1)), int(np.clip(c, 0, m.n - 1))
        cod = m.codes[m.soil_id[r, c]]
        mats, fr = _perfil(cod, m, AGS)
        conf = CONFIANZA_ASIGNACION.get(cod, "N/A")
        cap = mo.cap[r, c]
        lleno = 100.0 * (mo.F[r, c] - mo.D[r, c]) / cap if cap > 0 else 0.0
        tch = mo.t_charco[r, c]
        return [
            f"{m.nombres.get(cod, cod)} ({cod}) · celda ({r}, {c})",
            f"Horizonte superficial: {mats[0]}  ·  confianza de la equivalencia: {conf}",
            f"Ks {m.ks_mm_h[r, c]:.2f} mm/h · ψ {m.psi_mm[r, c]:.0f} mm · θe {m.theta_e[r, c]:.3f}",
            f"Perfil: {m.profundidad_mm[r, c]:.0f} mm sobre roca · cabe {cap:.0f} mm de agua",
            f"Elevación {m.elevacion_m[r, c]:.2f} m · pendiente {100 * m.pendiente[r, c]:.1f}%",
            f"Infiltrado: {mo.F[r, c]:.1f} mm · perfil lleno al {lleno:.0f}%",
            f"Agua en superficie: {mo.H[r, c]:.2f} mm",
            f"Lluvia {mo.P[r, c]:.1f} mm · recibió {mo.recibido[r, c]:.1f} · envió {mo.enviado[r, c]:.1f} mm",
            "Primer charco: " + ("aún no" if not np.isfinite(tch) else f"{tch:.1f} min"),
        ]

    def info_detalle(self, r: int, c: int) -> list[str]:
        lu = self.lupa
        med, sim = lu.medio, lu.sim
        r, c = int(np.clip(r, 0, 199)), int(np.clip(c, 0, 199))
        w = sim.grid[r, c]
        celdas = lu.info["celdas"]
        cel = celdas[int(lu.info["columna_celda"][c])]
        if med.kind[r, c] == 0:
            return [f"Aire / agua libre · fila {r}, columna {c}", f"Agua en la celda: {100 * w:.0f}% ({w * 10:.1f} mm)",
                    f"Terreno de esta columna: fila {med.surface_row[c]}"]
        if med.kind[r, c] == 2:
            return [f"Roca impermeable · fila {r}, columna {c}"]
        mat = med.materials[int(med.soil_id[c])][int(med.layer[r, c])]
        capa = int(med.layer[r, c])
        nombre = "sustrato" if capa == 4 else f"horizonte {capa + 1} de 4"
        cap = med.cap[r, c]
        return [f"{mat} ({nombre}) · celda aérea {cel}",
                f"Ks {med.ks_mm_h[r, c]:.2f} mm/h · ψ {med.psi_mm[r, c]:.0f} mm · θe {cap:.3f}",
                f"Profundidad bajo el terreno: {med.depth_mm[r, c]:.0f} mm",
                f"Saturación: {100 * w / cap:.0f}% · capacidad de campo {100 * med.sfc[r, c]:.0f}%",
                "Agua que se mueve por gravedad" if w / cap > med.sfc[r, c] else "Agua retenida (no drena por gravedad)"]

    # ---- texto del panel ---------------------------------------------------
    def resumen(self) -> list[str]:
        return {"aerea": self._resumen_aereo, "corte": self._resumen_corte, "detalle": self._resumen_detalle}[self.vista]()

    def _cabecera_tiempo(self, eng, i):
        return f"t = {eng.tiempo_min:.1f} min | {'llueve' if eng.llueve else 'sin lluvia'} | {i:g} mm/h ({classify_rain(i)})"

    def _resumen_aereo(self) -> list[str]:
        m, r = self.motor.metricas(), self.resumen_suelos
        n = self.mapa.n
        area = (n * self.mapa.dx_m) ** 2
        i = self.motor.intensidad_mm_h
        inf_pct = 100.0 * m["vol_infiltrado_m3"] / m["vol_lluvia_m3"] if m["vol_lluvia_m3"] > 0 else 0.0
        c = r["confianza"]
        return [
            f"Mapa {n * self.mapa.dx_m:.0f} m x {n * self.mapa.dx_m:.0f} m = {area:,.0f} m2".replace(",", " "),
            f"Con parametros: {r['con_parametros_pct']:.1f}% | dominante: {r['dominante']} {r['dominante_pct']:.1f}%",
            f"Confianza Media {c['Media']:.1f}% Baja {c['Baja']:.1f}% N/A {c['N/A']:.1f}%",
            self._cabecera_tiempo(self.motor, i),
            f"Lluvia: {m['lluvia_mm']:.1f} mm = {m['vol_lluvia_m3']:.0f} m3",
            f"Infiltrado: {m['vol_infiltrado_m3']:.0f} m3 ({inf_pct:.0f}%)",
            f"En superficie: {m['vol_superficial_m3']:.1f} m3 | salio: {m['vol_salida_m3']:.0f} m3",
            f"Encharcamiento max {m['prof_max_mm']:.1f} mm | medio {m['prof_media_mm']:.2f} mm",
            f"Area encharcada (>1 mm): {m['area_encharcada_pct']:.2f}%",
            f"Sumideros activos: {len(self.vista_aerea.sumideros(self.motor.H))}",
            f"Primer charco: {_fmt_min(m['primer_charco_min'])}",
            f"Vaciado a la mitad del pico: {_fmt_min(m['t_mitad_min'])}",
            f"Balance de masa: {m['error_balance_mm_celda']:.1e} mm*celda",
            f"Velocidad: {self.vel_real:.1f} min/s logrados (pedido {self.velocidad_min_s:.2g})",
        ]

    def _resumen_corte(self) -> list[str]:
        cs = self.corte
        r = cs.resumen(self.motor)
        i = self.motor.intensidad_mm_h
        suelos = " ".join(f"{k} {p:.0f}%" for p, k in r["suelos"][:3])
        return [
            f"Corte: {self.nombre_linea}",
            f"Largo {self.mapa.n * self.mapa.dx_m:.0f} m | desnivel {r['desnivel_m']:.2f} m | vertical x{cs.exageracion:.0f}",
            f"Suelos: {suelos}",
            "Mismo motor que el mapa (sin simulacion aparte)",
            self._cabecera_tiempo(self.motor, i),
            f"Infiltrado medio en la linea: {r['infiltrado_medio_mm']:.1f} mm",
            f"Frente de humedad: medio {r['frente_medio_mm']:.0f} mm | max {r['frente_max_mm']:.0f} mm",
            f"Perfiles llenos: {r['perfiles_llenos']} de {self.mapa.n}",
            f"Agua en superficie: media {r['agua_media_mm']:.2f} mm | max {r['agua_max_mm']:.1f} mm",
            f"Encharcado (>1 mm): {r['encharcado_pct']:.0f}% de la linea",
            f"Balance de masa del mapa: {self.motor.error_balance():.1e}",
            f"Velocidad: {self.vel_real:.1f} min/s logrados (pedido {self.velocidad_min_s:.2g})",
        ]

    def _resumen_detalle(self) -> list[str]:
        lu = self.lupa
        m = lu.metricas()
        a, b = lu.info["celdas"]
        nombres = [self.mapa.nombres.get(self.mapa.codes[self.mapa.soil_id[r, c]], "?") for r, c in (a, b)]
        i = lu.sim.rain_intensity_mm_h
        return [
            f"Detalle de 2 m (automata): celdas {a} y {b}",
            f"Suelos: {nombres[0]} | {nombres[1]}",
            f"Pendiente local: {100 * lu.info['pendiente']:.1f}%",
            self._cabecera_tiempo(lu, i),
            "(mm = lamina sobre el ancho de 2 m)",
            f"Lluvia: {m['lluvia_mm']:.1f} mm (incluye aporte {m['aporte_mm']:.1f})",
            *([f"Aviso: {m['aporte_no_cupo_mm']:.1f} mm de aporte no cupieron"] if m["aporte_no_cupo_mm"] > 0.05 else []),
            f"En el suelo: {m['suelo_mm']:.1f} mm | charcos: {m['charcos_mm']:.1f} mm",
            f"Escorrentia fuera: {m['escorrentia_fuera_mm']:.1f} mm",
            f"Percolacion profunda: {m['percolacion_mm']:.2f} mm",
            f"Infiltrado desde la apertura: {lu.infiltracion_mm():.2f} mm",
            f"Primer charco en el detalle: {_fmt_min(m['t_charco_min'])}",
            f"Balance de masa: {m['error_masa']:.1e}",
            "Motor aparte: Esc vuelve al corte del mapa",
            f"Velocidad: {self.vel_real:.1f} min/s logrados (pedido {self.velocidad_min_s:.2g})",
        ]
