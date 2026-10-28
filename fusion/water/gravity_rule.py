import numpy as np


class gravity_Rule:
    """Gravedad: cada celda pasa agua a la de abajo (vectorizada con numpy, mismo resultado que la versión por bucles)."""

    def __init__(self, flow_rate=1.0):
        self.flow_rate = flow_rate

    def apply(self, current, blocked_targets=()):
        flow = 1.0 - current[1:]
        np.minimum(flow, current[:-1], out=flow)
        np.minimum(flow, self.flow_rate, out=flow)
        if isinstance(blocked_targets, np.ndarray):
            if blocked_targets.shape != current.shape:
                raise ValueError("blocked_targets debe tener la misma forma que current")
            flow[blocked_targets[1:]] = 0.0
        else:
            for row, column in blocked_targets:  # celdas destino bloqueadas
                if 1 <= row < current.shape[0]:
                    flow[row - 1, column] = 0.0
        updated = current.copy()
        updated[:-1] -= flow
        updated[1:] += flow
        return updated


class lateral_Flow_Rule:
    """Flujo lateral: pares de columnas vecinas se igualan si alguna está apoyada en una celda llena."""

    def __init__(self, flow_rate=1.0):
        self.flow_rate = flow_rate

    def apply(self, current, step, supported_cells=(), solid_cells=()):
        rows, columns = current.shape
        updated = current.copy()
        if columns < 2:
            return updated
        first = step % 2
        left = slice(first, columns - 1, 2)   # vistas (sin copiar): columnas izquierdas de cada par
        right = slice(first + 1, columns, 2)  # columnas derechas de cada par

        # una celda está «apoyada» si la de abajo está llena o es una celda de soporte (drenaje)
        supported = np.empty((rows, columns), dtype=bool)
        np.greater_equal(current[1:], 1.0, out=supported[:-1])
        if isinstance(solid_cells, np.ndarray):
            if solid_cells.shape != current.shape:
                raise ValueError("solid_cells debe tener la misma forma que current")
            supported[:-1] |= solid_cells[1:]
        supported[-1] = True  # el piso siempre sostiene al agua
        for row, column in supported_cells:
            if row >= 1:
                supported[row - 1, column] = True
        active = supported[:, left] | supported[:, right]

        difference = current[:, left] - current[:, right]
        flow = np.abs(difference)
        flow *= 0.5
        np.minimum(flow, self.flow_rate, out=flow)
        flow *= np.sign(difference)
        flow *= active
        if isinstance(solid_cells, np.ndarray):
            flow *= ~(solid_cells[:, left] | solid_cells[:, right])
        updated[:, left] -= flow
        updated[:, right] += flow
        return updated
