"""Lupa: el corte vertical (autómata de water/) de una ventana de 2 m del mapa aéreo.

Se inicializa desde el estado del mapa en ese instante, sin repetir la historia:
  - cada columna recibe en el suelo la infiltración acumulada F de su celda aérea (se llenan los horizontes de arriba
    hacia abajo hasta repartir exactamente F), y
  - el agua superficial H de la celda se apila sobre el suelo.
Después corre con el mismo reloj (dt) y la misma lluvia que el mapa. El escurrimiento que la celda recibía de sus
vecinas (`aporte`, mm por paso) entra como agua extra repartida sobre sus columnas mientras dure la lluvia. La lupa
no modifica el mapa aéreo (acoplamiento de un solo sentido).
"""
from __future__ import annotations

import numpy as np

from aereo.motor import MotorAereo
from core.escenario import Escenario
from core.mapa import Mapa
from corte.medio import DZ_MM, N_CORTE, construir_medio

SIN_LIMITE = 10 ** 12
from water.water_class import water_cellular_automata


def _llenar_suelo(sim, medio, F_col_mm):
    """Añade F_col_mm[c] (mm) de agua a cada columna, de arriba hacia abajo, hasta su capacidad."""
    soil = medio.kind == 1
    espacio = np.where(soil, medio.cap - sim.grid, 0.0) * DZ_MM                    # mm que caben en cada celda
    antes = np.cumsum(espacio, axis=0) - espacio
    poner = np.clip(F_col_mm[None, :] - antes, 0.0, espacio)
    sim.grid = sim.grid + poner / DZ_MM
    return float(poner.sum() / DZ_MM)


def _apilar_agua(sim, medio, mm_col):
    """Pone mm_col[c] (mm) de agua libre sobre el agua o el suelo de cada columna, de abajo hacia arriba (como la lluvia
    del corte). Cambia sim.grid en sitio y devuelve las unidades añadidas."""
    resto = np.asarray(mm_col, float) / DZ_MM
    aire = medio.kind == 0
    total = 0.0
    for r in range(int(medio.surface_row.max()) - 1, -1, -1):
        poner = np.minimum(1.0 - sim.grid[r], resto) * aire[r]
        sim.grid[r] += poner
        resto = resto - poner
        total += float(poner.sum())
        if resto.max() <= 1e-15:
            break
    return total


class Lupa:
    def __init__(self, motor: MotorAereo, fila: int, col: int, direccion: str = "E", usar_aporte: bool = True,
                 micro_mm: float = 60.0):
        esc: Escenario = motor.esc
        mapa: Mapa = motor.mapa
        self.fila, self.col, self.direccion, self.usar_aporte = fila, col, direccion, usar_aporte
        self.medio, self.info = construir_medio(mapa, fila, col, direccion, ks_sigma_ln=0.0, bedrock=esc.bedrock,
                                                micro_mm=micro_mm, semilla=esc.seed)
        pasos_lluvia = SIN_LIMITE if not np.isfinite(motor.duracion_min) else int(round(motor.duracion_min * 60.0 / esc.dt_s))
        if not motor.lluvia_activa:
            pasos_lluvia = motor.paso
        self.sim = water_cellular_automata(
            N_CORTE, max(pasos_lluvia, motor.paso) + 10_000_000, rain_steps=0, rain_intensity_mm_h=motor.intensidad_mm_h,
            rain_noise=esc.ruido_lluvia, rain_seed=esc.semilla_lluvia, dz_mm=DZ_MM, dt_s=esc.dt_s, substeps=4,
            soil=self.medio, soil_initial_saturation=esc.s0, edge_runoff=False,           # el 2D siempre con bordes cerrados
            
            bottom_drain=(esc.bedrock == "fractured"))
        self.sim.rain_steps = pasos_lluvia                    # la lluvia dura lo mismo que en el mapa
        self.sim.current_step = motor.paso                    # mismo reloj que el mapa
        self.motor = motor
        c = self.info["celdas"]
        izq = self.info["columna_celda"] == 0
        F = np.where(izq, motor.F[c[0]], motor.F[c[1]])
        H = np.where(izq, motor.H[c[0]], motor.H[c[1]])
        neto = motor.aporte_neto                                # con bordes cerrados solo importa lo que se acumula
        self.aporte_col = np.where(izq, neto[c[0]], neto[c[1]]).astype(float)   # mm por paso
        self.F0_mm = F.copy()
        _llenar_suelo(self.sim, self.medio, F)
        _apilar_agua(self.sim, self.medio, H)
        if self.sim.surface_rule is not None:                  # el agua copiada del mapa queda con la superficie horizontal
            self.sim.grid, _ = self.sim.surface_rule.apply(self.sim.grid)
        self.paso0 = motor.paso
        self.suelo0_mm = self.sim.soil_water_mm
        self.sim.initial_water = float(self.sim.grid.sum())   # desde aquí se lleva el balance de masa de la lupa
        self.aporte_total_mm = 0.0                            # mm sobre el ancho de la ventana
        self.aporte_no_cupo_mm = 0.0
        self.t_charco = None

    # ---- control en vivo ---------------------------------------------------
    def set_intensidad(self, mm_h: float):
        self.sim.set_rain_intensity(mm_h)

    def set_duracion(self, minutos: float):
        self.sim.rain_steps = SIN_LIMITE if not np.isfinite(minutos) else int(round(minutos * 60.0 / self.sim.scale.dt_s))

    def set_lluvia_activa(self, activa: bool):
        """Detiene la lluvia ahora mismo o la reanuda sin límite."""
        self.sim.rain_steps = SIN_LIMITE if activa else self.sim.current_step

    @property
    def tiempo_min(self) -> float:
        return self.sim.current_step * self.sim.scale.dt_s / 60.0

    @property
    def llueve(self) -> bool:
        return self.sim.current_step < self.sim.rain_steps

    def step(self):
        sim = self.sim
        if self.usar_aporte and self.llueve and self.aporte_col.any():
            puesto = _apilar_agua(sim, self.medio, self.aporte_col)
            sim.total_rain += puesto
            self.aporte_total_mm += float(self.aporte_col.mean())
            # si el charco llega al borde superior de la ventana, el agua que no cabe se descarta y se reporta
            self.aporte_no_cupo_mm += max(float(self.aporte_col.sum()) / DZ_MM - puesto, 0.0) * DZ_MM / N_CORTE
        sim.step()
        if self.t_charco is None and sim.surface_water_mm > 0.02:
            self.t_charco = self.tiempo_min

    def avanzar(self, pasos: int):
        for _ in range(int(pasos)):
            self.step()

    # ---- medidas (mm sobre el ancho de 2 m, como en el corte) ---------------
    def infiltracion_mm(self) -> float:
        """Agua que entró al suelo desde que se abrió la lupa (mm), contando lo que se fue por el fondo."""
        return self.sim.soil_water_mm - self.suelo0_mm + self.sim.deep_mm

    def metricas(self) -> dict:
        s = self.sim
        return {"tiempo_min": self.tiempo_min, "lluvia_mm": s.rain_mm, "suelo_mm": s.soil_water_mm,
                "charcos_mm": s.surface_water_mm, "escorrentia_fuera_mm": s.runoff_mm, "percolacion_mm": s.deep_mm,
                "aporte_mm": self.aporte_total_mm, "aporte_no_cupo_mm": self.aporte_no_cupo_mm, "t_charco_min": self.t_charco,
                "error_masa": s.mass_balance_error()}
