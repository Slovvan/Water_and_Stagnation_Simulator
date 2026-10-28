"""Escenario compartido por las dos vistas (aérea y corte): terreno, lluvia y condiciones de frontera."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Escenario:
    # --- terreno (vista aérea) ---
    seed: int = 42
    n: int = 100                  # celdas por lado
    dx_m: float = 1.0             # metros por celda aérea
    escala: float = 0.02          # frecuencia del relieve (1/celdas); el tamaño de las formas es ~1/escala celdas
    octavas: int = 3
    persistencia: float = 0.5
    amplitud_m: float = 1.5       # desnivel máximo del terreno
    # --- lluvia (compartida; se puede cambiar en vivo desde la barra) ---
    intensidad_mm_h: float = 30.0
    duracion_min: float = 180.0   # minutos con lluvia (tormenta de 3 h como en el corte)
    ruido_lluvia: float = 0.0     # ruido relativo entre celdas (0 = lluvia uniforme; con ruido la película fina parpadea)
    semilla_lluvia: int = 40
    # --- suelo ---
    s0: float = 0.30              # saturación inicial (supuesto del dataset)
    ks_sigma_ln: float = 0.5      # variabilidad lognormal de Ks (0 = suelo uniforme)
    bedrock: str = "fractured"    # 'fractured' (poroso, Ks 0.5 mm/h) o 'sealed' (impermeable)
    # --- fronteras y tiempo ---
    bordes_abiertos: bool = False # cerrados por defecto (la interfaz solo usa bordes cerrados: con bordes abiertos el agua parpadea)
    dt_s: float = 10.0            # paso de tiempo (el mismo reloj para ambas vistas)
    pasos_ruteo: int = 4          # pasadas del reparto superficial por paso

    def __post_init__(self):
        if self.n < 4:
            raise ValueError("n debe ser al menos 4")
        if not self.dx_m > 0 or not self.dt_s > 0:
            raise ValueError("dx_m y dt_s deben ser positivos")
        if not 0 <= self.s0 < 1:
            raise ValueError("s0 debe estar en [0, 1)")
        if self.intensidad_mm_h < 0 or self.duracion_min < 0 or self.ruido_lluvia < 0:
            raise ValueError("intensidad, duración y ruido no pueden ser negativos")
        if self.bedrock not in ("fractured", "sealed"):
            raise ValueError("bedrock debe ser 'fractured' o 'sealed'")
        if self.escala <= 0 or int(self.octavas) < 1 or int(self.pasos_ruteo) < 1:
            raise ValueError("escala > 0, octavas >= 1 y pasos_ruteo >= 1")

    def con(self, **cambios) -> "Escenario":
        return replace(self, **cambios)
