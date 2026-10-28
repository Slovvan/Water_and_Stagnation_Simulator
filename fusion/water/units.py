"""Escala física de la simulación: unidades de agua <-> milímetros y pasos <-> tiempo.

Una «unidad de agua» es una celda llena (valor 1.0). Dos parámetros fijan la escala:
  dz_mm : milímetros de agua que representa una unidad (una celda llena).
  dt_s  : segundos que dura un paso de la simulación.
"""
from dataclasses import dataclass

# Niveles de lluvia predefinidos (mm/h). Etiquetas según la clasificación por intensidad:
# ligera < 2.5 | moderada 2.5-10 | fuerte 10-50 | violenta > 50 mm/h.
RAIN_PRESETS = {"Moderada": 5.0, "Fuerte": 30.0, "Violenta": 60.0}
MAX_RAIN_MM_H = 100.0


def classify_rain(mm_per_hour):
    if mm_per_hour <= 0:
        return "Sin lluvia"
    if mm_per_hour < 2.5:
        return "Ligera"
    if mm_per_hour < 10:
        return "Moderada"
    if mm_per_hour < 50:
        return "Fuerte"
    return "Violenta"


@dataclass(frozen=True)
class Scale:
    dz_mm: float = 5.0
    dt_s: float = 60.0

    def __post_init__(self):
        if not self.dz_mm > 0:
            raise ValueError("dz_mm debe ser mayor que cero")
        if not self.dt_s > 0:
            raise ValueError("dt_s debe ser mayor que cero")

    @property
    def hours_per_step(self):
        return self.dt_s / 3600.0

    def rain_units_per_step(self, mm_per_hour):
        """Agua que recibe cada columna en un paso, en unidades de agua."""
        return mm_per_hour * self.hours_per_step / self.dz_mm

    def depth_mm(self, units, columns):
        """Lámina equivalente (mm) de una cantidad de agua repartida sobre `columns` columnas."""
        return units * self.dz_mm / columns

    def steps_to_hours(self, steps):
        return steps * self.hours_per_step

    def hours_to_steps(self, hours):
        return int(round(hours / self.hours_per_step))
