import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.widgets import Button, Slider

from water.rain_visual import RainOverlay
from water.units import MAX_RAIN_MM_H, RAIN_PRESETS, classify_rain

MAX_STEPS_PER_FRAME = 30
WET_BLUE = np.array([0.10, 0.30, 0.70])  # tinte del suelo mojado
TEXT_ROWS = 6.0  # filas de margen sobre la rejilla, dentro del eje, para el texto de estado


class WaterViewer:
    """Animación optimizada (blit): solo se redibuja la imagen, el texto y las gotas.

    Controles: barra de lluvia (mm/h) con tres niveles predefinidos y barra de velocidad
    (pasos de simulación por fotograma; 0 = pausa).
    """

    def __init__(self, simulation, steps_per_frame=2, interval_ms=16):
        if int(steps_per_frame) != steps_per_frame or steps_per_frame < 0:
            raise ValueError("steps_per_frame debe ser un entero mayor o igual que 0")
        self.sim = simulation
        self.steps_per_frame = int(steps_per_frame)
        self.interval_ms = interval_ms
        self.slider = None
        self.speed_slider = None
        self.buttons = []
        self.overlay = None
        self.animation = None
        self._built = False
        self._finished = False

    # ---- construcción de la figura -----------------------------------------
    def build(self):
        sim = self.sim
        n = sim.grid_size
        self.fig, self.ax = plt.subplots(figsize=(7, 8.4))
        self.fig.subplots_adjust(
            top=0.97, left=0.1, right=0.86, bottom=0.31 if sim.calibrated_rain else 0.14
        )
        self.soil = getattr(sim, "soil", None)
        if self.soil is not None:
            self._blues = plt.get_cmap("Blues")
            self.image = self.ax.imshow(self._render_soil(), interpolation="nearest")
        else:
            self.image = self.ax.imshow(
                sim.grid, cmap="Blues", vmin=0, vmax=1, interpolation="nearest"
            )
        terrain_rgba = np.zeros((n, n, 4))
        if self.soil is None:
            terrain_rgba[sim.terrain] = (0.48, 0.31, 0.16, 1.0)
        self.terrain_image = self.ax.imshow(
            terrain_rgba, interpolation="nearest", zorder=2
        )
        self.info = self.ax.text(
            0.5, 0.995, "", transform=self.ax.transAxes, ha="center", va="top",
            fontsize=9, linespacing=1.4,
        )
        self.title = self.info  # nombre anterior, por compatibilidad
        self.ax.set_xlabel("Columna")
        self.ax.set_ylabel("Fila")
        bottom_limit = n - 0.5
        if sim.drainage_rule.column is not None:
            drain_column = sim.drainage_rule.column
            self.ax.annotate(
                "",
                xy=(drain_column, n + 0.35),
                xytext=(drain_column, n - 0.45),
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
                annotation_clip=False,
            )
            bottom_limit = n + 0.7
        self.ax.set_ylim(bottom_limit, -0.5 - TEXT_ROWS)
        self.ax.set_yticks(np.arange(0, n, 5))
        if self.soil is not None:
            mappable = ScalarMappable(norm=Normalize(0, 1), cmap="Blues")
            self.fig.colorbar(mappable, ax=self.ax, label="Agua libre (celda llena = 1)")
            handles = [Patch(facecolor=rgb, edgecolor="none", label=name) for name, rgb in self.soil.legend()]
            self.ax.legend(handles=handles, loc="lower left", fontsize=6,
                           ncol=2, framealpha=0.85, title="Materiales (Ks en mm/h)", title_fontsize=6)
        else:
            self.fig.colorbar(self.image, ax=self.ax, label="Cantidad de agua")
        if sim.calibrated_rain:
            self.overlay = RainOverlay(self.ax, n, n, seed=1)
        self._build_controls()
        self._update_info()
        self._built = True

    def _build_controls(self):
        sim = self.sim
        speed_y = 0.215 if sim.calibrated_rain else 0.06
        ax_speed = self.fig.add_axes([0.30, speed_y, 0.50, 0.03])
        self.speed_slider = Slider(
            ax_speed, "Velocidad (0 = pausa)", 0, MAX_STEPS_PER_FRAME,
            valinit=min(self.steps_per_frame, MAX_STEPS_PER_FRAME), valstep=1, valfmt="%d pasos",
        )
        self.speed_slider.vline.set_visible(False)  # sin marca roja del valor inicial
        self.speed_slider.on_changed(self._on_speed)
        if not sim.calibrated_rain:
            return

        slider_max = max(MAX_RAIN_MM_H, sim.rain_intensity_mm_h)
        ax_slider = self.fig.add_axes([0.30, 0.15, 0.50, 0.03])
        self.slider = Slider(
            ax_slider, "Lluvia (mm/h)", 0.0, slider_max,
            valinit=sim.rain_intensity_mm_h, valstep=0.5,
        )
        for value in RAIN_PRESETS.values():  # marcas de los niveles predefinidos
            ax_slider.axvline(value, color="black", lw=0.8, alpha=0.6)
        self.slider.vline.set_visible(False)
        self.slider.on_changed(self._on_slider)

        count = len(RAIN_PRESETS)
        width, gap = 0.2, 0.04
        left = (1.0 - (count * width + (count - 1) * gap)) / 2.0
        for k, (name, value) in enumerate(RAIN_PRESETS.items()):
            ax_button = self.fig.add_axes([left + k * (width + gap), 0.04, width, 0.07])
            button = Button(ax_button, f"{name}\n{value:g} mm/h")
            button.on_clicked(self._preset_callback(value))
            self.buttons.append(button)

    def _render_soil(self):
        """RGB: aire = azul según agua libre; suelo = color del material teñido de azul según saturación."""
        sim, soil = self.sim, self.soil
        w = sim.grid
        cap = np.where(soil.cap > 0, soil.cap, 1.0)
        S = np.clip(w / cap, 0.0, 1.0)
        wet = (0.80 * S ** 1.5)[..., None]    # suelo seco conserva su color; saturado, azul intenso
        soil_rgb = soil.base_rgb * (1.0 - wet) + WET_BLUE * wet
        air_rgb = self._blues(np.clip(w, 0.0, 1.0))[..., :3]
        return np.where((soil.kind == 0)[..., None], air_rgb, soil_rgb)

    # ---- interacción --------------------------------------------------------
    def _on_slider(self, value):
        self.sim.set_rain_intensity(value)
        self._update_info()
        self.fig.canvas.draw_idle()

    def _on_speed(self, value):
        self.steps_per_frame = int(value)

    def _preset_callback(self, value):
        def callback(event):
            self.slider.set_val(value)

        return callback

    def set_rain(self, mm_per_hour):
        """Equivale a mover la barra de lluvia (útil para probar sin ventana)."""
        self.slider.set_val(mm_per_hour)

    def set_speed(self, steps_per_frame):
        """Equivale a mover la barra de velocidad."""
        self.speed_slider.set_val(steps_per_frame)

    # ---- animación ----------------------------------------------------------
    def _update_info(self):
        sim = self.sim
        if sim.calibrated_rain:
            intensity = sim.rain_intensity_mm_h
            first = (
                f"Paso {sim.current_step} | t = {sim.elapsed_hours:.1f} h | "
                f"Lluvia: {intensity:g} mm/h ({classify_rain(intensity)})"
            )
            second = [
                f"Acumulada: {sim.rain_mm:.1f} mm",
                f"En la rejilla: {sim.stored_mm:.1f} mm",
            ]
            if sim.drainage_rule.column is not None:
                second.append(f"Drenada: {sim.drained_mm:.1f} mm")
            if sim.soil is not None:
                second = [
                    f"Lluvia: {sim.rain_mm:.1f} mm",
                    f"En el suelo: {sim.soil_water_mm:.1f} mm",
                    f"Charcos: {sim.surface_water_mm:.1f} mm",
                    f"Escorrentía fuera: {sim.runoff_mm:.1f} mm",
                    f"Percolación: {sim.deep_mm:.2f} mm",
                ]
            if sim.soil is not None:
                lines = [first, " | ".join(second[:3]), " | ".join(second[3:])]
            else:
                lines = [first, " | ".join(second)]
            if sim.soil is None and sim.grid[0].min() >= 1.0:
                lines.append("REJILLA LLENA")
            text = "\n".join(lines)
        else:
            text = (
                f"Paso {sim.current_step} | "
                f"Agua: {sim.grid.sum():.2f} | "
                f"Drenada: {sim.total_drained:.2f}"
            )
        self.info.set_text(text)

    def _artists(self):
        if self.overlay is None:
            return self.image, self.terrain_image, self.info
        return self.image, self.terrain_image, self.info, self.overlay.collection

    def _refresh(self):
        sim = self.sim
        self.image.set_data(self._render_soil() if self.soil is not None else sim.grid)
        self._update_info()
        if self.overlay is not None:
            raining = sim.current_step < sim.rain_steps          # las gotas se dibujan solo mientras llueve
            self.overlay.update(sim.rain_intensity_mm_h if raining else 0.0, sim.surface_y())

    def _initialize(self):
        self._refresh()
        return self._artists()

    def _frames(self):
        frame = 0
        while self.sim.current_step < self.sim.time_steps:
            yield frame
            frame += 1

    def advance(self, frame):
        sim = self.sim
        if self.steps_per_frame > 0:
            target = min(sim.current_step + self.steps_per_frame, sim.time_steps)
            while sim.current_step < target:
                sim.step()
            self._refresh()
        if sim.current_step >= sim.time_steps:
            self._finish()
        return self._artists()

    def _finish(self):
        """Al terminar, los artistas dejan de ser «animados» para que sigan visibles al redibujar."""
        if self._finished:
            return
        self._finished = True
        if self.overlay is not None:
            self.overlay.clear()
        for artist in self._artists():
            artist.set_animated(False)
        self.fig.canvas.draw_idle()

    def snapshot(self, path, dpi=90):
        """Guarda el estado actual en una imagen (incluye los artistas animados)."""
        artists = self._artists()
        previous = [a.get_animated() for a in artists]
        for artist in artists:
            artist.set_animated(False)
        self.fig.savefig(path, dpi=dpi)
        for artist, flag in zip(artists, previous):
            artist.set_animated(flag)

    def run(self, show=True):
        if not self._built:
            self.build()
        self.animation = FuncAnimation(
            self.fig,
            self.advance,
            init_func=self._initialize,
            frames=self._frames,
            interval=self.interval_ms,
            blit=True,
            repeat=False,
            cache_frame_data=False,
        )
        if show:
            plt.show()
        return self.animation


def animate(self, steps_per_frame=2, interval_ms=16, show=True):
    viewer = WaterViewer(self, steps_per_frame=steps_per_frame, interval_ms=interval_ms)
    animation = viewer.run(show=show)
    animation.viewer = viewer  # mantiene vivos los widgets (barras y botones)
    return animation
