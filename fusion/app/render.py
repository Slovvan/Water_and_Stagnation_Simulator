"""Dibujo de las dos vistas como arreglos NumPy (sin pygame): se pueden probar y guardar como imagen."""
from __future__ import annotations

import numpy as np

from core.mapa import Mapa
from corte.medio import _perfil
from water.soil_data import AGS
from water.soil_terrain import MATERIAL_RGB

# --- agua: profundidad [mm] -> color. De azul claro (película fina) a azul marino (charco hondo); sin cian ---
PARADAS_MM = np.array([0.05, 1.0, 5.0, 20.0, 50.0, 100.0])
PARADAS_RGB = np.array([(124, 194, 242), (92, 168, 234), (60, 140, 222), (38, 108, 200), (27, 82, 168), (17, 56, 138)], float)
MARCAS_LEYENDA_MM = (0, 20, 50, 100)
CIELO = np.array([221, 235, 246], float)                          # aire del corte
AGUA_CA = np.array([58, 128, 208], float)                         # agua libre en el autómata (celda llena)
SOMBRA_HUMEDA = np.array([6.0, 12.0, 26.0])                       # el suelo mojado se oscurece (a la mitad) y se azula un poco
HUMEDO = SOMBRA_HUMEDA / 255.0
WET_BLUE = HUMEDO                                                 # nombre anterior
BLUES_X = np.array([0.0, 0.25, 0.5, 0.75, 1.0])                   # (en desuso)


def color_agua_mm(h):
    """Color RGB (0-255) del agua con profundidad h (mm, arreglo)."""
    h = np.clip(np.asarray(h, float), PARADAS_MM[0], PARADAS_MM[-1])
    return np.stack([np.interp(h, PARADAS_MM, PARADAS_RGB[:, k]) for k in range(3)], axis=-1)


def mojar(base_rgb255, humedad):
    """Color del suelo con humedad 0..1 (0 = seco: su color; 1 = saturado: la mitad de claro con un matiz azulado)."""
    m = np.asarray(humedad, float)
    m = m[..., None] if m.ndim == np.ndim(base_rgb255) - 1 else m
    return base_rgb255 * (1.0 - 0.5 * m) + SOMBRA_HUMEDA * m


def _blues(v):
    """(en desuso) mezcla cielo -> agua según la fracción v de celda llena."""
    v = np.clip(np.asarray(v, float), 0.0, 1.0)[..., None]
    return (CIELO * (1 - v) + AGUA_CA * v) / 255.0


SUAVIZADO = 0.30                 # fracción de la diferencia que se aplica en cada fotograma (evita el parpadeo)
ENCENDER_MM, APAGAR_MM = 0.35, 0.12   # histéresis: el agua se pinta con >0.35 mm y se deja de pintar con <0.12 mm


class VistaAerea:
    """Base de color por suelo + relieve sombreado (calculados una vez por mapa) y agua encima. El agua que se dibuja se
    suaviza en el tiempo y usa histéresis, para que la película fina no parpadee aunque su espesor varíe cada paso."""

    def __init__(self, mapa: Mapa):
        self.mapa = mapa
        rgb = []
        for c in mapa.codes:
            top = _perfil(c, mapa, AGS)[0][0]
            rgb.append(np.array(MATERIAL_RGB.get(top, (0.5, 0.5, 0.5))) * 255.0)
        self.base = np.array(rgb)[mapa.soil_id]
        gy, gx = np.gradient(mapa.elevacion_m)
        s = gx + gy
        s = s / (np.abs(s).max() + 1e-12)
        self.sombra = np.clip(1.0 + 0.35 * s, 0.7, 1.3)[..., None]
        # fosas: mínimos locales del TERRENO (no cambian con el agua, así que las marcas no parpadean)
        z = mapa.elevacion_m
        n = z.shape[0]
        zp = np.pad(z, 1, constant_values=np.inf)
        minimo = np.full(z.shape, np.inf)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    minimo = np.minimum(minimo, zp[1 + dr:1 + dr + n, 1 + dc:1 + dc + n])
        self.fosa = z <= minimo
        self.reiniciar()

    def reiniciar(self):
        self._hv = np.zeros(self.mapa.elevacion_m.shape)
        self._mojado = np.zeros(self.mapa.elevacion_m.shape, bool)

    def suavizar(self, H: np.ndarray) -> np.ndarray:
        self._hv += SUAVIZADO * (H - self._hv)
        self._mojado = (self._mojado | (self._hv > ENCENDER_MM)) & ~(self._hv < APAGAR_MM)
        return self._hv

    def imagen(self, H: np.ndarray, suavizar: bool = True) -> np.ndarray:
        """suavizar=False reutiliza el estado ya suavizado (para dibujar el minimapa en el mismo fotograma)."""
        hv = self.suavizar(H) if suavizar else self._hv
        img = self.base.copy()
        mojado = self._mojado
        if mojado.any():
            h = hv[mojado]
            alfa = (0.82 + 0.15 * np.minimum(h / 10.0, 1.0))[:, None]          # el agua se ve como agua, no como un velo
            img[mojado] = img[mojado] * (1 - alfa) + color_agua_mm(h) * alfa
        return np.clip(img * self.sombra, 0, 255).astype(np.uint8)

    def sumideros(self, H: np.ndarray, umbral_mm: float = 5.0) -> np.ndarray:
        """(fila, col) de las fosas del terreno que tienen más de `umbral_mm` de agua (se usa el agua suavizada si existe)."""
        agua = self._hv if self._hv.max() > 0 else H
        return np.argwhere(self.fosa & (agua > umbral_mm))

    @staticmethod
    def barra_leyenda(alto: int = 150) -> np.ndarray:
        """(alto, 1, 3) uint8: 0 mm abajo, 100 mm arriba."""
        mm = np.linspace(100.0, 0.0, alto)
        return np.clip(color_agua_mm(mm), 0, 255).astype(np.uint8)[:, None, :]


def imagen_corte(sim, medio) -> np.ndarray:
    """Detalle de 2 m: aire = cielo; agua libre = azul (más intenso cuanto más llena la celda); suelo = color del material,
    más oscuro y azulado cuanto más saturado."""
    w = sim.grid
    cap = np.where(medio.cap > 0, medio.cap, 1.0)
    S = np.clip(w / cap, 0.0, 1.0)
    suelo = mojar(medio.base_rgb * 255.0, S ** 1.5)
    t = np.clip(w, 0.0, 1.0)[..., None]
    aire = CIELO * (1 - t) + AGUA_CA * t
    img = np.where((medio.kind == 0)[..., None], aire, suelo)
    return np.clip(img, 0, 255).astype(np.uint8)


class Gotas:
    """Gotas dibujadas sobre el corte: solo un efecto visual (la física deposita la lluvia directo en la superficie)."""

    def __init__(self, columnas: int, filas: int, n_max: int = 300, caida: float = 2.5, estela: float = 1.6, seed: int = 1):
        self.columnas, self.filas, self.estela = columnas, filas, estela
        self.rng = np.random.default_rng(seed)
        self.n_max = n_max
        self.x = self.rng.uniform(-0.5, columnas - 0.5, n_max)
        self.y = self.rng.uniform(-0.5, filas - 0.5, n_max)
        self.v = caida * self.rng.uniform(0.8, 1.2, n_max)

    def activas(self, intensidad_mm_h: float, max_mm_h: float = 100.0) -> int:
        if intensidad_mm_h <= 0:
            return 0
        return max(1, int(round(self.n_max * np.sqrt(min(intensidad_mm_h / max_mm_h, 1.0)))))

    def segmentos(self, intensidad_mm_h: float, superficie_y: np.ndarray) -> np.ndarray:
        """(m, 3): x, y_arriba, y_abajo de cada raya, en coordenadas de celda del corte."""
        n = self.activas(intensidad_mm_h)
        if n == 0:
            return np.zeros((0, 3))
        x, y = self.x[:n], self.y[:n]
        y += self.v[:n]
        col = np.clip(np.rint(x).astype(int), 0, self.columnas - 1)
        cayo = y >= superficie_y[col]
        if cayo.any():
            y[cayo] = self.rng.uniform(-4.0, -0.5, int(cayo.sum()))
            x[cayo] = self.rng.uniform(-0.5, self.columnas - 0.5, int(cayo.sum()))
            col = np.clip(np.rint(x).astype(int), 0, self.columnas - 1)
        piso = superficie_y[col]
        arriba = np.maximum(y - self.estela, -0.5)
        abajo = np.maximum(np.minimum(np.maximum(y, -0.5), piso), arriba)
        return np.stack([x, arriba, abajo], axis=1)
