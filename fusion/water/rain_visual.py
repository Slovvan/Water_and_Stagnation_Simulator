import numpy as np
from matplotlib.collections import LineCollection

from water.units import MAX_RAIN_MM_H


class RainOverlay:
    """Gotas dibujadas sobre la rejilla. Es solo un efecto visual.

    La física deposita la lluvia directo en la superficie del agua (sin retraso de caída);
    estas rayas caen unas filas por fotograma y se detienen al tocar el agua de su columna.
    Cuántas gotas se ven depende de la intensidad de lluvia (escala con la raíz cuadrada,
    para que la lluvia débil también se note).
    """

    def __init__(self, ax, columns, rows, drops_per_column=4, fall_rows_per_frame=2.5,
                 streak_rows=1.6, seed=0):
        self.columns, self.rows = columns, rows
        self.rng = np.random.default_rng(seed)
        self.n_max = max(1, int(drops_per_column * columns))
        self.x = self.rng.uniform(-0.5, columns - 0.5, self.n_max)
        self.y = self.rng.uniform(-0.5, rows - 0.5, self.n_max)  # repartidas desde el inicio
        self.speed = fall_rows_per_frame * self.rng.uniform(0.8, 1.2, self.n_max)
        self.streak = streak_rows
        self.collection = LineCollection([], colors="#2f6db5", linewidths=1.1, alpha=0.9)
        ax.add_collection(self.collection)

    def active_drops(self, intensity_mm_h):
        if intensity_mm_h <= 0:
            return 0
        share = min(intensity_mm_h / MAX_RAIN_MM_H, 1.0)
        return max(1, int(round(self.n_max * np.sqrt(share))))

    def clear(self):
        self.collection.set_segments([])

    def update(self, intensity_mm_h, surface_y):
        """Avanza un fotograma. surface_y[c] = borde superior del agua en la columna c (coordenada de fila)."""
        n = self.active_drops(intensity_mm_h)
        if n == 0:
            self.clear()
            return self.collection
        x, y = self.x[:n], self.y[:n]  # vistas: modifican self.x y self.y
        y += self.speed[:n]
        column = np.clip(np.rint(x).astype(int), 0, self.columns - 1)
        landed = y >= surface_y[column]
        count = int(landed.sum())
        if count:  # las gotas que tocan el agua reaparecen sobre la rejilla
            y[landed] = self.rng.uniform(-4.0, -0.5, count)
            x[landed] = self.rng.uniform(-0.5, self.columns - 0.5, count)
            column = np.clip(np.rint(x).astype(int), 0, self.columns - 1)
        floor = surface_y[column]
        top = np.maximum(y - self.streak, -0.5)
        bottom = np.maximum(np.minimum(np.maximum(y, -0.5), floor), top)
        segments = np.empty((n, 2, 2))
        segments[:, 0, 0] = x
        segments[:, 1, 0] = x
        segments[:, 0, 1] = top
        segments[:, 1, 1] = bottom
        self.collection.set_segments(segments)
        return self.collection
