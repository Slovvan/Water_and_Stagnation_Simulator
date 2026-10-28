"""Mapa aéreo: elevación (m) y suelos por celda, con los mismos datos y reglas que el corte (water.soil_data)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.escenario import Escenario
from water.soil_data import AGS, CATENA_ORDER, CONFIANZA_ASIGNACION, PROFILES, SoilDataset
from water.soil_terrain import (MIN_DEPTH_MM, fractal_noise, get_material, norm01,
                                resolve_profile_materials, zscore)


@dataclass
class Mapa:
    n: int
    dx_m: float
    elevacion_m: np.ndarray       # (n, n) metros
    soil_id: np.ndarray           # (n, n) índice en `codes`
    codes: list
    ks_mm_h: np.ndarray           # Ks del horizonte superficial (controla la infiltración)
    psi_mm: np.ndarray            # succión del frente húmedo
    theta_e: np.ndarray           # porosidad efectiva del horizonte superficial
    almacen_sat_mm: np.ndarray    # agua que cabe en todo el perfil a saturación [mm]
    profundidad_mm: np.ndarray    # profundidad del perfil (4 horizontes) [mm]
    k_fondo_mm_h: float           # fuga por el sustrato rocoso cuando el perfil se llena [mm/h]
    nombres: dict

    @property
    def pendiente(self) -> np.ndarray:
        gy, gx = np.gradient(self.elevacion_m, self.dx_m)
        return np.hypot(gx, gy)


def _propiedades_por_codigo(codes, ds: SoilDataset):
    k = len(codes)
    top = np.zeros((3, k))
    alm = np.zeros(k)
    dep = np.zeros((2, k))
    for i, c in enumerate(codes):
        mats = resolve_profile_materials(c, ds)
        prof = PROFILES[c]
        top[:, i] = get_material(mats[0], ds)                      # (psi, ks, theta_e) del horizonte superficial
        alm[i] = sum(f * get_material(m, ds)[2] for f, m in zip(prof.fractions, mats))
        dep[:, i] = prof.depth_mm
    return top, alm, dep


def generar_mapa(esc: Escenario, ds: SoilDataset = AGS) -> Mapa:
    n = esc.n
    rng = np.random.default_rng(esc.seed)
    base = max(1.0 / esc.escala, 2.0)
    elev01 = fractal_noise((n, n), base, int(esc.octavas), rng, esc.persistencia)
    elevacion = (elev01 * esc.amplitud_m).astype(float)

    # Grupos de suelo con los % EXACTOS del dataset, ordenados por posición en el paisaje (catena)
    codes = [c for c in CATENA_ORDER if c in ds.pct] + [c for c in ds.pct if c not in CATENA_ORDER]
    w = np.array([ds.pct[c] for c in codes], float)
    w /= w.sum()
    cortes = np.round(np.cumsum(w) * n * n).astype(int)
    cortes[-1] = n * n
    pos = norm01(0.55 * elev01 + 0.45 * fractal_noise((n, n), max(base * 0.5, 2.0), 3, rng))
    orden = np.argsort(pos, axis=None, kind="stable")
    plano = np.empty(n * n, int)
    ini = 0
    for k, fin in enumerate(cortes):
        plano[orden[ini:fin]] = k
        ini = fin
    soil_id = plano.reshape(n, n)

    top, alm, dep = _propiedades_por_codigo(codes, ds)
    gy, gx = np.gradient(elevacion, esc.dx_m)
    pendiente = np.hypot(gx, gy)
    prof = dep[0][soil_id] + dep[1][soil_id] * zscore(fractal_noise((n, n), 12, 3, rng))
    prof = np.maximum(prof * np.clip(1.0 - 0.8 * pendiente, 0.35, 1.0), MIN_DEPTH_MM)

    ks = top[1][soil_id] * np.exp(esc.ks_sigma_ln * zscore(fractal_noise((n, n), 6, 3, rng)))
    th = np.clip(top[2][soil_id] * (1 + 0.04 * zscore(fractal_noise((n, n), 8, 2, rng))), 0.03, 0.6)
    psi = top[0][soil_id]
    k_fondo = get_material("Roca fracturada", ds)[1] if esc.bedrock == "fractured" else 0.0
    return Mapa(n=n, dx_m=esc.dx_m, elevacion_m=elevacion, soil_id=soil_id, codes=codes, ks_mm_h=ks, psi_mm=psi,
                theta_e=th, almacen_sat_mm=prof * alm[soil_id], profundidad_mm=prof, k_fondo_mm_h=float(k_fondo),
                nombres=dict(ds.names))


def mapa_homogeneo(textura: str, n: int = 20, profundidad_mm: float = 5000.0, k_fondo_mm_h: float = 0.0,
                   dx_m: float = 1.0, elevacion_m: np.ndarray | None = None, ds: SoilDataset = AGS) -> Mapa:
    """Suelo de una sola textura (para verificar contra Green-Ampt). Terreno plano por defecto."""
    psi, ks, th = get_material(textura, ds)
    uno = np.ones((n, n))
    elev = np.zeros((n, n)) if elevacion_m is None else np.asarray(elevacion_m, float)
    return Mapa(n=n, dx_m=dx_m, elevacion_m=elev, soil_id=np.zeros((n, n), int), codes=["HOM"],
                ks_mm_h=ks * uno, psi_mm=psi * uno, theta_e=th * uno, almacen_sat_mm=profundidad_mm * th * uno,
                profundidad_mm=profundidad_mm * uno, k_fondo_mm_h=float(k_fondo_mm_h), nombres={"HOM": textura})


def resumen_mapa(mapa: Mapa) -> dict:
    """Porcentajes del mapa generado. La confianza sigue la hoja Relacion_Ags del xlsx (Media / Baja / N/A)."""
    total = mapa.n * mapa.n
    pct = {c: 100.0 * np.count_nonzero(mapa.soil_id == k) / total for k, c in enumerate(mapa.codes)}
    conf = {"Media": 0.0, "Baja": 0.0, "N/A": 0.0}
    for c, p in pct.items():
        conf[CONFIANZA_ASIGNACION.get(c, "N/A")] += p
    dom = max(pct, key=pct.get)
    return {"pct": pct, "confianza": conf, "dominante": mapa.nombres.get(dom, dom), "dominante_pct": pct[dom],
            "con_parametros_pct": 100.0 - conf["N/A"]}
