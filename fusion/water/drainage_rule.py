class DrainageRule:
    def __init__(self, column=None, flow_rate=0.0, lateral_flow_rate=0.25):
        if flow_rate < 0:
            raise ValueError("La tasa de drenaje no puede ser negativa")
        if lateral_flow_rate < 0:
            raise ValueError("La tasa lateral del drenaje no puede ser negativa")

        self.column = column
        self.flow_rate = flow_rate
        self.lateral_flow_rate = lateral_flow_rate

    def apply(self, current):
        updated = current.copy()
        if self.column is None:
            return updated, 0.0
        if not 0 <= self.column < updated.shape[1]:
            raise IndexError("La columna de drenaje está fuera de la cuadrícula")

        bottom_row = updated.shape[0] - 1
        drained = min(updated[bottom_row, self.column], self.flow_rate)
        updated[bottom_row, self.column] -= drained

        neighbors = [
            neighbor
            for neighbor in (self.column - 1, self.column + 1)
            if 0 <= neighbor < updated.shape[1]
        ]
        available_capacity = self.flow_rate - drained
        requested = [
            min(updated[bottom_row, neighbor], self.lateral_flow_rate)
            for neighbor in neighbors
        ]
        total_requested = sum(requested)
        scale = min(1.0, available_capacity / total_requested) if total_requested else 0.0

        for neighbor, amount in zip(neighbors, requested):
            transferred = amount * scale
            updated[bottom_row, neighbor] -= transferred
            drained += transferred

        return updated, drained
