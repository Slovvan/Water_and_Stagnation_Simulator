import numpy as np
from water.drainage_rule import DrainageRule
from water.gravity_rule import gravity_Rule, lateral_Flow_Rule
from water.rain_class import RainRule, UniformRainRule
from water.soil_rules import SoilGravityRule, SoilLateralRule
from water.surface_leveling import SurfaceLeveling
from water.units import Scale


class water_cellular_automata:
    def __init__(
        self,
        grid_size,
        time_steps,
        flow_rate=1.0,
        rain_steps=None,
        rain_drops_per_step=0,
        rain_amount_per_drop=0.0,
        rain_seed=None,
        drain_column=None,
        drain_rate=0.0,
        drain_lateral_rate=0.25,
        rain_intensity_mm_h=None,
        rain_noise=0.0,
        dz_mm=5.0,
        dt_s=60.0,
        substeps=1,
        terrain=None,
        # --- suelo procedural (opcional) ---
        soil=None,                    # SoilMedium de water.soil_terrain.generate_soil_terrain
        soil_initial_saturation=0.30, # saturación inicial del suelo (el supuesto de tu dataset)
        soil_lateral_anisotropy=1.0,  # Kh / Kv del suelo
        edge_runoff=False,            # True: el agua libre sale por los bordes izquierdo/derecho
        bottom_drain=True,            # True: la última fila drena (percolación profunda) a su Ks
        surface_leveling=True,        # True: el agua libre se nivela (charcos horizontales, desborde entre cuencas)
    ):
        if grid_size <= 0:
            raise ValueError("grid_size debe ser mayor que cero")
        if time_steps < 0:
            raise ValueError("time_steps no puede ser negativo")
        if rain_steps is None:
            rain_steps = time_steps
        if rain_steps < 0 or rain_steps > time_steps or int(rain_steps) != rain_steps:
            raise ValueError("rain_steps debe estar entre 0 y time_steps")
        if not 0 < flow_rate <= 1:
            raise ValueError("flow_rate debe estar entre 0 y 1")
        if not np.isfinite(drain_rate) or drain_rate < 0:
            raise ValueError("drain_rate debe ser un número no negativo")
        if not np.isfinite(drain_lateral_rate) or drain_lateral_rate < 0:
            raise ValueError("drain_lateral_rate debe ser un número no negativo")
        if drain_column is not None and not 0 <= drain_column < grid_size:
            raise IndexError("La columna de drenaje está fuera de la cuadrícula")
        if int(substeps) != substeps or substeps < 1:
            raise ValueError("substeps debe ser un entero mayor o igual que 1")
        if rain_intensity_mm_h is not None and rain_drops_per_step > 0:
            raise ValueError(
                "Usa lluvia por gotas (rain_drops_per_step) o lluvia calibrada "
                "(rain_intensity_mm_h), no ambas"
            )
        if soil is not None:
            if soil.grid_size != grid_size:
                raise ValueError("soil.grid_size debe ser igual a grid_size")
            if abs(soil.dz_mm - dz_mm) > 1e-9:
                raise ValueError("soil.dz_mm debe ser igual a dz_mm (genera el suelo con el mismo dz_mm)")
            if terrain is not None:
                raise ValueError("Con soil no se usa terrain: la roca sólida viene de soil (bedrock='sealed')")
            if not 0 <= soil_initial_saturation <= 1:
                raise ValueError("soil_initial_saturation debe estar entre 0 y 1")

        self.grid_size = grid_size
        self.time_steps = time_steps
        self.rain_steps = int(rain_steps)
        self.flow_rate = flow_rate
        self.substeps = int(substeps)
        self.scale = Scale(dz_mm=dz_mm, dt_s=dt_s)
        self.grid = np.zeros((grid_size, grid_size), dtype=float)
        self.soil = soil
        if soil is not None:
            self.terrain = soil.solid.copy()          # roca impermeable (si la hay)
            self.ground = soil.ground.copy()          # suelo + roca: aquí no cae lluvia
            self.grid = soil_initial_saturation * np.where(soil.kind == 1, soil.cap, 0.0)
        elif terrain is None:
            self.terrain = np.zeros_like(self.grid, dtype=bool)
            self.ground = self.terrain
        else:
            self.terrain = np.asarray(terrain, dtype=bool)
            if self.terrain.shape != self.grid.shape:
                raise ValueError("terrain debe tener la misma forma que la cuadrícula")
            self.terrain = self.terrain.copy()
            self.ground = self.terrain
        self.initial_water = float(self.grid.sum())
        self.current_step = 0
        self.total_rain = 0.0
        self.total_drained = 0.0
        self.total_runoff = 0.0      # agua libre que salió por los bordes
        self.total_deep = 0.0        # percolación profunda por el fondo del suelo
        self.calibrated_rain = rain_intensity_mm_h is not None
        if self.calibrated_rain:
            self.rain_rule = UniformRainRule(
                self.scale,
                intensity_mm_h=rain_intensity_mm_h,
                noise=rain_noise,
                seed=rain_seed,
            )
        else:
            self.rain_rule = RainRule(
                drops_per_step=rain_drops_per_step,
                amount_per_drop=rain_amount_per_drop,
                seed=rain_seed,
            )
        if soil is not None:
            self.gravity_rule = SoilGravityRule(soil, self.scale, self.substeps, flow_rate, bottom_drain)
            self.lateral_flow_rule = SoilLateralRule(
                soil, self.scale, self.substeps, flow_rate, soil_lateral_anisotropy, edge_runoff
            )
            self.surface_rule = (
                SurfaceLeveling(soil, open_edges=edge_runoff, flow_rate=flow_rate)
                if surface_leveling else None
            )
        else:
            self.gravity_rule = gravity_Rule(flow_rate=self.flow_rate)
            self.lateral_flow_rule = lateral_Flow_Rule(flow_rate=self.flow_rate)
        self.drainage_rule = DrainageRule(
            column=drain_column,
            flow_rate=drain_rate,
            lateral_flow_rate=drain_lateral_rate,
        )

    # ---- lluvia calibrada -------------------------------------------------
    @property
    def rain_intensity_mm_h(self):
        return self.rain_rule.intensity_mm_h if self.calibrated_rain else None

    def set_rain_intensity(self, mm_per_hour):
        """Cambia la intensidad de lluvia (mm/h) en cualquier momento de la simulación."""
        if not self.calibrated_rain:
            raise RuntimeError(
                "La intensidad solo se puede cambiar con lluvia calibrada "
                "(crea la simulación con rain_intensity_mm_h)"
            )
        self.rain_rule.set_intensity(mm_per_hour)

    # ---- medidas en unidades físicas ---------------------------------------
    @property
    def rain_mm(self):
        """Lámina de lluvia acumulada (mm) sobre el ancho de la rejilla."""
        return self.scale.depth_mm(self.total_rain, self.grid_size)

    @property
    def stored_mm(self):
        """Agua que hay ahora en la rejilla, como lámina equivalente (mm)."""
        return self.scale.depth_mm(self.grid.sum(), self.grid_size)

    @property
    def drained_mm(self):
        """Agua drenada acumulada, como lámina equivalente (mm)."""
        return self.scale.depth_mm(self.total_drained, self.grid_size)

    @property
    def runoff_mm(self):
        """Escorrentía que salió por los bordes (mm sobre el ancho)."""
        return self.scale.depth_mm(self.total_runoff, self.grid_size)

    @property
    def deep_mm(self):
        """Percolación profunda acumulada (mm sobre el ancho)."""
        return self.scale.depth_mm(self.total_deep, self.grid_size)

    @property
    def surface_water_mm(self):
        """Agua libre sobre el terreno (charcos y escorrentía), mm sobre el ancho."""
        if self.soil is None:
            return self.stored_mm
        return self.scale.depth_mm(self.grid[self.soil.kind == 0].sum(), self.grid_size)

    @property
    def soil_water_mm(self):
        """Agua dentro del suelo, mm sobre el ancho."""
        if self.soil is None:
            return 0.0
        return self.scale.depth_mm(self.grid[self.soil.kind != 0].sum(), self.grid_size)

    @property
    def elapsed_hours(self):
        return self.scale.steps_to_hours(self.current_step)

    def mass_balance_error(self):
        """Debe ser ~0: agua actual − (inicial + lluvia − drenada − escorrentía − percolación profunda)."""
        expected = self.initial_water + self.total_rain - self.total_drained - self.total_runoff - self.total_deep
        return float(self.grid.sum() - expected)

    def surface_y(self):
        """Fila (coordenada de imagen) del borde superior del agua libre de cada columna.
        Sirve para que las gotas dibujadas se detengan al tocar el agua o el suelo."""
        n = self.grid_size
        if self.soil is None:
            return n - 0.5 - np.minimum(self.grid.sum(axis=0), n)
        air_height = np.where(self.soil.kind == 0, self.grid, 0.0).sum(axis=0)
        return self.soil.surface_row - 0.5 - np.minimum(air_height, self.soil.surface_row)

    # ---- dinámica ----------------------------------------------------------
    def add_water(self, row, column, amount):
        if not (0 <= row < self.grid_size and 0 <= column < self.grid_size):
            raise IndexError("La posición está fuera de la cuadrícula")
        cap = 1.0 if self.soil is None else self.soil.cap[row, column]
        if not np.isfinite(amount) or amount < 0 or self.grid[row][column] + amount > cap:
            raise ValueError("El agua de una celda debe quedar entre 0 y su capacidad")

        self.grid[row][column] += amount

    def _apply_rules(self, step):
        if self.soil is not None:
            self._apply_soil_rules(step)
            return
        drain_support = ()
        if self.drainage_rule.column is not None:
            drain_column = self.drainage_rule.column
            drain_cell = (self.grid_size - 1, drain_column)
            drain_support = (drain_cell,)

        updated = self.grid
        for k in range(self.substeps):
            updated = self.gravity_rule.apply(updated, blocked_targets=self.terrain)
            updated = self.lateral_flow_rule.apply(
                updated,
                step * self.substeps + k,
                supported_cells=drain_support,
                solid_cells=self.terrain,
            )
        self.grid = updated

    def _apply_soil_rules(self, step):
        updated = self.grid
        for k in range(self.substeps):
            updated = self.gravity_rule.apply(updated)
            updated, deep = self.gravity_rule.drain_bottom(updated)
            self.total_deep += deep
            updated = self.lateral_flow_rule.apply(updated, step * self.substeps + k)
            if self.surface_rule is not None:
                updated, out = self.surface_rule.apply(updated)
                self.total_runoff += out
            else:
                updated, out = self.lateral_flow_rule.drain_edges(updated)
                self.total_runoff += out
        self.grid = updated

    def cellular_automata_2d(self):
        for step in range(self.time_steps):
            self.step()
        return self.grid

    def step(self):
        if self.current_step < self.rain_steps:
            self.grid, rain_added = self.rain_rule.apply(
                self.grid, blocked_cells=self.ground
            )
        else:
            rain_added = 0.0
        self.total_rain += rain_added
        self._apply_rules(self.current_step)
        self.grid, drained = self.drainage_rule.apply(self.grid)
        self.total_drained += drained
        self.current_step += 1
        return self.grid
