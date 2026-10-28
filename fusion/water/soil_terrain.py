"""
soil_terrain.py
===============
Terreno procedural en CORTE TRANSVERSAL (vista lateral), compatible con la rejilla del CA.

La rejilla es de grid_size x grid_size celdas cuadradas de dz_mm. Cada celda es:
    AIRE   capacidad 1.0 (agua libre: lluvia, charcos, escorrentía)
    SUELO  capacidad = porosidad efectiva theta_e (Rawls) -> el agua cabe solo en los poros
    SOLIDO roca impermeable (solo si bedrock="sealed")

El valor de agua de una celda sigue siendo "fracción de celda llena de agua líquida" (= contenido
volumétrico theta), así que 1 unidad = dz_mm de agua, igual que en tu units.Scale.

Qué genera:
  1. Perfil de superficie (lomas y valles) con ruido fractal 1D.
  2. Grupos de suelo por columna con los % EXACTOS del dataset (Aguascalientes), ordenados por
     posición en el paisaje: Fluvisoles en el valle, Leptosoles en las lomas.
  3. Horizontes paralelos a la superficie: 4 capas (soil_data.PROFILES) + sustrato rocoso hasta
     el fondo de la rejilla. Espesor variable por columna (ruido y pendiente).
  4. psi, Ks y theta_e de Rawls por celda, con variabilidad lognormal de Ks (mediana = Rawls).

Solo requiere numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from water.soil_data import AGS, CATENA_ORDER, CUSTOM_MATERIALS, PROFILES, SoilDataset

AIR, SOIL, SOLID = 0, 1, 2
BASEMENT = 4            # índice de la capa de sustrato (debajo de las 4 capas del perfil)
MIN_DEPTH_MM = 80.0

# Colores por material (RGB 0-1) para la animación
MATERIAL_RGB = {
    "Clay": (0.50, 0.29, 0.22), "Silty Clay": (0.54, 0.34, 0.26), "Sandy Clay": (0.62, 0.40, 0.28),
    "Clay Loam": (0.58, 0.38, 0.26), "Silty Clay Loam": (0.56, 0.38, 0.28),
    "Sandy Clay Loam": (0.66, 0.46, 0.30), "Silt Loam": (0.64, 0.50, 0.34),
    "Loam": (0.52, 0.37, 0.22), "Sandy Loam": (0.76, 0.60, 0.38), "Loamy Sand": (0.84, 0.72, 0.48),
    "Sand": (0.92, 0.82, 0.58), "Leptosol": (0.62, 0.58, 0.50), "Roca fracturada": (0.42, 0.42, 0.46),
}
LAYER_SHADE = (0.80, 0.92, 1.0, 1.0, 1.0)   # el horizonte superficial es más oscuro


# --------------------------------------------------------------------------- #
# Ruido (mismo esquema que el generador anterior)
# --------------------------------------------------------------------------- #
def norm01(a):
    a = a - a.min()
    m = a.max()
    return a / m if m > 0 else a


def zscore(a):
    s = a.std()
    return (a - a.mean()) / s if s > 0 else np.zeros_like(a)


def value_noise(shape, scale, rng):
    H, W = shape
    gh, gw = int(np.ceil(H / scale)) + 2, int(np.ceil(W / scale)) + 2
    g = rng.random((gh, gw))
    ys, xs = np.arange(H) / scale, np.arange(W) / scale
    y0, x0 = ys.astype(int), xs.astype(int)
    ty, tx = ys - y0, xs - x0
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
    a, b = g[np.ix_(y0, x0)], g[np.ix_(y0, x0 + 1)]
    c, d = g[np.ix_(y0 + 1, x0)], g[np.ix_(y0 + 1, x0 + 1)]
    top = a + (b - a) * tx[None, :]
    bot = c + (d - c) * tx[None, :]
    return top + (bot - top) * ty[:, None]


def fractal_noise(shape, base_scale, octaves, rng, persistence=0.5):
    total, amp, norm = np.zeros(shape), 1.0, 0.0
    for o in range(octaves):
        total += amp * value_noise(shape, max(base_scale / 2 ** o, 2.0), rng)
        norm += amp
        amp *= persistence
    return norm01(total / norm)


def field_capacity_fraction(psi):
    """[SUPUESTO] Saturación efectiva a capacidad de campo (fracción de theta_e), lineal con psi de
    Rawls: arena ≈ 0.20, arcilla ≈ 0.70. Por debajo de este valor el agua ya no drena por gravedad."""
    return np.clip(0.20 + 0.50 * (np.asarray(psi, float) - 49.5) / (316.3 - 49.5), 0.15, 0.75)


def get_material(name, ds: SoilDataset = AGS):
    if name in ds.rawls:
        return ds.rawls[name]
    if name in CUSTOM_MATERIALS:
        return CUSTOM_MATERIALS[name]
    raise KeyError(f"Material desconocido: {name!r}")


def resolve_profile_materials(code, ds: SoilDataset):
    out = []
    for tok in PROFILES[code].textures:
        if tok == "@":
            tex = ds.texture.get(code)
            if tex is None:
                raise ValueError(f"{code} no tiene textura Rawls; edita PROFILES[{code!r}]")
            out.append(tex)
        else:
            out.append(tok)
    return out


# --------------------------------------------------------------------------- #
# Contenedor
# --------------------------------------------------------------------------- #
@dataclass
class SoilMedium:
    grid_size: int
    dz_mm: float
    kind: np.ndarray            # AIR / SOIL / SOLID                         (n, n)
    cap: np.ndarray             # capacidad de agua por celda (1, theta_e, 0) (n, n)
    ks_mm_h: np.ndarray         # Ks saturada [mm/h] (0 fuera del suelo)      (n, n)
    psi_mm: np.ndarray          # succión del frente húmedo [mm]              (n, n)
    sfc: np.ndarray             # saturación efectiva a cap. de campo         (n, n)
    layer: np.ndarray           # -1 aire · 0..3 perfil · 4 sustrato          (n, n)
    depth_mm: np.ndarray        # profundidad bajo la superficie de su columna (n, n)
    base_rgb: np.ndarray        # color del material                          (n, n, 3)
    surface_row: np.ndarray     # primera fila de suelo de cada columna       (n,)
    soil_id: np.ndarray         # índice en `codes` por columna               (n,)
    codes: list
    materials: list            # materiales [grupo][capa 0..4]

    @property
    def solid(self):
        return self.kind == SOLID

    @property
    def ground(self):
        return self.kind != AIR

    def legend(self, ds: SoilDataset = AGS):
        """[(nombre, rgb)] de los materiales presentes (para la leyenda)."""
        seen = {}
        for mats in self.materials:
            for m in mats:
                seen[m] = MATERIAL_RGB.get(m, (0.5, 0.5, 0.5))
        present = {m for m in seen if np.any(self._mat_mask(m))}
        return [(f"{m} (Ks {get_material(m, ds)[1]:g})", seen[m]) for m in seen if m in present]

    def _mat_mask(self, m):
        mask = np.zeros(self.kind.shape, bool)
        for k, mats in enumerate(self.materials):
            for l, name in enumerate(mats):
                if name == m:
                    mask |= (self.soil_id[None, :] == k) & (self.layer == l)
        return mask

    def summary(self, ds: SoilDataset = AGS):
        n = self.grid_size
        lines = [f"Corte {n} x {n} celdas · dz = {self.dz_mm:g} mm -> {n * self.dz_mm / 1000:.1f} m x "
                 f"{n * self.dz_mm / 1000:.1f} m",
                 f"{'grupo':<12}{'% columnas':>11}{'% dataset':>11}{'prof. suelo (mm)':>18}"]
        tot = sum(ds.pct[c] for c in self.codes)
        for k, c in enumerate(self.codes):
            m = self.soil_id == k
            depth = ((self.layer[:, m] >= 0) & (self.layer[:, m] < BASEMENT)).sum(0).mean() * self.dz_mm
            lines.append(f"{ds.names.get(c, c):<12}{100 * m.mean():11.2f}{100 * ds.pct[c] / tot:11.2f}{depth:18.0f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Generación
# --------------------------------------------------------------------------- #
def generate_soil_terrain(grid_size=200, dz_mm=10.0, seed=0, top_margin=0.15, relief=0.14,
                          tilt=0.35, ks_sigma_ln=0.5, bedrock="fractured",
                          dataset: SoilDataset = AGS) -> SoilMedium:
    """
    grid_size   celdas por lado (debe ser igual al grid_size de la simulación)
    dz_mm       tamaño de celda; define la escala física: grid_size*dz_mm = ancho = alto del corte
    top_margin  fracción de la rejilla que queda de aire sobre la loma más alta (espacio para lluvia)
    relief      desnivel loma-valle como fracción de la rejilla
    tilt        0..1, peso de la pendiente regional (el agua tiende a correr hacia la izquierda)
    ks_sigma_ln desviación de ln(Ks); 0.5 ≈ CV 0.5 (el dataset da CV(Ks) = 0.65 entre grupos)
    bedrock     'fractured' sustrato rocoso poroso (Ks 0.5 mm/h) · 'sealed' roca impermeable
    """
    if bedrock not in ("fractured", "sealed"):
        raise ValueError("bedrock debe ser 'fractured' o 'sealed'")
    rng = np.random.default_rng(seed)
    n = grid_size

    # 1) Superficie ----------------------------------------------------------
    base = fractal_noise((1, n), 70, 4, rng)[0]
    ramp = np.linspace(0, 1, n)
    elev01 = norm01((1 - tilt) * base + tilt * ramp)               # alto = loma
    top = int(top_margin * n)
    relief_cells = int(relief * n)
    surface_row = np.rint(top + relief_cells * (1 - elev01)).astype(int)
    surface_row = np.clip(surface_row, 1, n - 8)

    # 2) Grupos de suelo con % exactos del dataset (catena) -----------------
    codes = [c for c in CATENA_ORDER if c in dataset.pct]
    codes += [c for c in dataset.pct if c not in codes]
    w = np.array([dataset.pct[c] for c in codes], float)
    w /= w.sum()
    edges = np.round(np.cumsum(w) * n).astype(int)
    edges[-1] = n
    pos = norm01(0.55 * elev01 + 0.45 * fractal_noise((1, n), 40, 3, rng)[0])
    order = np.argsort(pos)
    soil_id = np.empty(n, int)
    start = 0
    for k, end in enumerate(edges):
        soil_id[order[start:end]] = k
        start = end

    # 3) Profundidad y límites entre horizontes ------------------------------
    nC = len(codes)
    psi_t, ks_t, th_t = (np.zeros((5, nC)) for _ in range(3))
    frac = np.zeros((4, nC))
    materials = []
    for k, c in enumerate(codes):
        mats = resolve_profile_materials(c, dataset) + ["Roca fracturada"]
        materials.append(mats)
        for l, m in enumerate(mats):
            psi_t[l, k], ks_t[l, k], th_t[l, k] = get_material(m, dataset)
        frac[:, k] = PROFILES[c].fractions
    mean_d = np.array([PROFILES[c].depth_mm[0] for c in codes])[soil_id]
    sd_d = np.array([PROFILES[c].depth_mm[1] for c in codes])[soil_id]
    depth_mm = mean_d + sd_d * zscore(fractal_noise((1, n), 30, 3, rng)[0])
    slope = np.abs(np.gradient(surface_row.astype(float)))          # celdas cuadradas: m/m
    depth_mm = np.maximum(depth_mm * np.clip(1.0 - 0.8 * slope, 0.35, 1.0), MIN_DEPTH_MM)
    depth_cells = depth_mm / dz_mm
    bounds = np.cumsum(frac[:, soil_id], axis=0) * depth_cells[None, :]   # (4, n)

    # 4) Celdas ---------------------------------------------------------------
    rows = np.arange(n)[:, None]
    d = rows - surface_row[None, :]                                  # índice de profundidad
    in_ground = d >= 0
    dd = d + 0.5
    layer = sum((dd >= bounds[k][None, :]).astype(int) for k in range(4))     # 0..4
    layer = np.where(in_ground, layer, -1)
    lay_c = np.clip(layer, 0, 4)
    gid = np.broadcast_to(soil_id[None, :], (n, n))

    kind = np.where(in_ground, SOIL, AIR)
    if bedrock == "sealed":
        kind = np.where(in_ground & (layer == BASEMENT), SOLID, kind)

    psi = psi_t[lay_c, gid]
    ks = ks_t[lay_c, gid] * np.exp(ks_sigma_ln * zscore(fractal_noise((n, n), 14, 3, rng)))
    th = np.clip(th_t[lay_c, gid] * (1 + 0.04 * zscore(fractal_noise((n, n), 16, 2, rng))), 0.03, 0.6)
    soil_mask = kind == SOIL
    cap = np.where(soil_mask, th, np.where(kind == AIR, 1.0, 0.0))
    ks = np.where(soil_mask, ks, 0.0)
    psi = np.where(soil_mask, psi, 0.0)
    sfc = np.where(soil_mask, field_capacity_fraction(psi), 0.0)

    rgb_t = np.zeros((5, nC, 3))
    for k in range(nC):
        for l in range(5):
            rgb_t[l, k] = np.array(MATERIAL_RGB.get(materials[k][l], (0.5, 0.5, 0.5))) * LAYER_SHADE[l]
    base_rgb = np.where((kind == AIR)[..., None], 1.0, rgb_t[lay_c, gid])
    if bedrock == "sealed":
        base_rgb = np.where((kind == SOLID)[..., None], 0.30, base_rgb)

    return SoilMedium(grid_size=n, dz_mm=float(dz_mm), kind=kind, cap=cap, ks_mm_h=ks, psi_mm=psi,
                      sfc=sfc, layer=layer, depth_mm=np.where(in_ground, d * dz_mm, 0.0),
                      base_rgb=base_rgb, surface_row=surface_row, soil_id=soil_id, codes=codes,
                      materials=materials)


def flat_soil(texture, grid_size=60, dz_mm=5.0, surface_row=10, ds: SoilDataset = AGS) -> SoilMedium:
    """Suelo plano y homogéneo de una sola textura (para validar contra Green-Ampt)."""
    n = grid_size
    psi, ks, th = get_material(texture, ds)
    rows = np.arange(n)[:, None] * np.ones((1, n), int)
    ground = rows >= surface_row
    kind = np.where(ground, SOIL, AIR)
    full = lambda v: np.where(ground, float(v), 0.0)
    return SoilMedium(
        grid_size=n, dz_mm=float(dz_mm), kind=kind, cap=np.where(ground, th, 1.0), ks_mm_h=full(ks),
        psi_mm=full(psi), sfc=np.where(ground, field_capacity_fraction(psi), 0.0),
        layer=np.where(ground, 0, -1), depth_mm=np.where(ground, (rows - surface_row) * dz_mm, 0.0),
        base_rgb=np.where(ground[..., None], np.array(MATERIAL_RGB.get(texture, (0.5, 0.4, 0.3))), 1.0),
        surface_row=np.full(n, surface_row), soil_id=np.zeros(n, int), codes=["HOM"],
        materials=[[texture] * 5])
