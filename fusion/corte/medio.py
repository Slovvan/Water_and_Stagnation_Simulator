"""Construye el medio del corte (water.soil_terrain.SoilMedium) para una ventana de 2 celdas aéreas (la lupa).

La ventana mide 2 m (200 columnas de 10 mm): la mitad izquierda es la celda aérea elegida y la derecha su vecina en la
dirección escogida. Las propiedades de cada columna salen del mapa aéreo (mismos grupos, perfiles y multiplicadores de
Ks y θe de esa celda), de modo que ambas vistas comparten datos. El relieve sale de la elevación aérea (4 puntos del
transecto) más un microrelieve fractal.
"""
from __future__ import annotations

import numpy as np

from core.mapa import Mapa
from water.soil_data import AGS, PROFILES, SoilDataset
from water.soil_terrain import (AIR, BASEMENT, LAYER_SHADE, MATERIAL_RGB, MIN_DEPTH_MM, SOIL, SOLID, SoilMedium,
                                field_capacity_fraction, fractal_noise, get_material, resolve_profile_materials, zscore)

N_CORTE = 200
DZ_MM = 10.0                       # 200 celdas × 10 mm = 2 m: justo dos celdas aéreas de 1 m
TOP_FILAS = 50                     # aire sobre el punto más alto (espacio para lluvia y charcos)
DIRECCIONES = {"E": (0, 1), "O": (0, -1), "S": (1, 0), "N": (-1, 0)}


def celdas_del_transecto(mapa: Mapa, fila: int, col: int, direccion: str):
    """Devuelve las 2 celdas de la ventana (izquierda -> derecha) y 4 alturas para el perfil (vecinas incluidas).
    La ventana es la celda elegida y su vecina en la dirección escogida; si esa vecina cae fuera del mapa, la ventana
    se corre una celda hacia adentro."""
    dr, dc = DIRECCIONES[direccion]
    n = mapa.n
    a, b = (fila, col), (fila + dr, col + dc)
    if not (0 <= b[0] < n and 0 <= b[1] < n):
        a, b = (fila - dr, col - dc), (fila, col)
    puntos = [(a[0] - dr, a[1] - dc), a, b, (b[0] + dr, b[1] + dc)]
    puntos = [(int(np.clip(r, 0, n - 1)), int(np.clip(c, 0, n - 1))) for r, c in puntos]
    return puntos[1], puntos[2], [float(mapa.elevacion_m[r, c]) for r, c in puntos]


def _perfil(codigo: str, mapa: Mapa, ds: SoilDataset):
    """Materiales (4 horizontes + sustrato) y fracciones del perfil. 'HOM' = suelo de una sola textura (pruebas)."""
    if codigo == "HOM":
        return [mapa.nombres["HOM"]] * 4 + ["Roca fracturada"], (0.25, 0.25, 0.25, 0.25)
    return resolve_profile_materials(codigo, ds) + ["Roca fracturada"], PROFILES[codigo].fractions


def construir_medio(mapa: Mapa, fila: int, col: int, direccion: str = "E", ks_sigma_ln: float = 0.0,
                    bedrock: str = "fractured", micro_mm: float = 20.0, semilla: int = 0,
                    ds: SoilDataset = AGS) -> tuple[SoilMedium, dict]:
    n, dz = N_CORTE, DZ_MM
    rng = np.random.default_rng([abs(int(semilla)), int(fila), int(col), sorted(DIRECCIONES).index(direccion)])
    a, b, alturas = celdas_del_transecto(mapa, fila, col, direccion)
    x = (np.arange(n) + 0.5) * dz / 1000.0                                      # m dentro de la ventana
    z_m = np.interp(x, [-0.5, 0.5, 1.5, 2.5], alturas)                          # perfil del terreno (m)
    micro = (fractal_noise((1, n), 40, 3, rng)[0] - 0.5) * micro_mm / 1000.0    # microrelieve ±micro_mm/2
    z_m = z_m + micro
    rel_filas = (z_m.max() - z_m) * 1000.0 / dz
    surface_row = np.clip(np.rint(TOP_FILAS + rel_filas).astype(int), 1, n - 8)

    # grupo, profundidad y multiplicadores de cada columna: los de su celda aérea
    mitad = n // 2
    celda_col = [a if c < mitad else b for c in range(n)]
    codigos = []
    for (r, c) in (a, b):
        cod = mapa.codes[mapa.soil_id[r, c]]
        if cod not in codigos:
            codigos.append(cod)
    sid = np.array([codigos.index(mapa.codes[mapa.soil_id[r, c]]) for r, c in celda_col])
    prof_mm = np.array([mapa.profundidad_mm[r, c] for r, c in celda_col])
    prof_mm = np.maximum(prof_mm, MIN_DEPTH_MM)

    nC = len(codigos)
    psi_t, ks_t, th_t, frac = np.zeros((5, nC)), np.zeros((5, nC)), np.zeros((5, nC)), np.zeros((4, nC))
    materials = []
    for k, c in enumerate(codigos):
        mats, fr = _perfil(c, mapa, ds)
        materials.append(mats)
        for l, m in enumerate(mats):
            psi_t[l, k], ks_t[l, k], th_t[l, k] = get_material(m, ds)
        frac[:, k] = fr
    # multiplicador de Ks y θe de la celda aérea respecto al valor nominal de su horizonte superficial
    mult_ks = np.array([mapa.ks_mm_h[r, c] / ks_t[0, codigos.index(mapa.codes[mapa.soil_id[r, c]])] for r, c in celda_col])
    mult_th = np.array([mapa.theta_e[r, c] / th_t[0, codigos.index(mapa.codes[mapa.soil_id[r, c]])] for r, c in celda_col])

    depth_cells = prof_mm / dz
    bounds = np.cumsum(frac[:, sid], axis=0) * depth_cells[None, :]
    rows = np.arange(n)[:, None]
    d = rows - surface_row[None, :]
    in_ground = d >= 0
    dd = d + 0.5
    layer = sum((dd >= bounds[k][None, :]).astype(int) for k in range(4))
    layer = np.where(in_ground, layer, -1)
    lay_c = np.clip(layer, 0, 4)
    gid = np.broadcast_to(sid[None, :], (n, n))

    kind = np.where(in_ground, SOIL, AIR)
    if bedrock == "sealed":
        kind = np.where(in_ground & (layer == BASEMENT), SOLID, kind)
    soil_mask = kind == SOIL
    local = np.exp(ks_sigma_ln * zscore(fractal_noise((n, n), 14, 3, rng))) if ks_sigma_ln > 0 else 1.0
    ks = ks_t[lay_c, gid] * mult_ks[None, :] * local
    th = np.clip(th_t[lay_c, gid] * mult_th[None, :], 0.03, 0.6)
    psi = psi_t[lay_c, gid]
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

    medio = SoilMedium(grid_size=n, dz_mm=float(dz), kind=kind, cap=cap, ks_mm_h=ks, psi_mm=psi, sfc=sfc, layer=layer,
                       depth_mm=np.where(in_ground, d * dz, 0.0), base_rgb=base_rgb, surface_row=surface_row,
                       soil_id=sid, codes=codigos, materials=materials)
    info = dict(celdas=(a, b), direccion=direccion, alturas_m=alturas, pendiente=float((alturas[2] - alturas[1])),
                columna_celda=np.array([0 if c < mitad else 1 for c in range(n)]))
    return medio, info
