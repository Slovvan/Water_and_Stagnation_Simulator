"""
soil_data.py
==============
Datos de entrada del suelo procedural (Aguascalientes).

Todo lo que viene del dataset está marcado como [DATASET]; todo lo que NO viene
del dataset y es una decisión del simulador está marcado como [SUPUESTO] para que
sea fácil de localizar y cambiar.

Unidades:  psi [mm] · Ks [mm/h] · theta_e [-] (porosidad efectiva)
"""
from __future__ import annotations

from dataclasses import dataclass

N_LAYERS = 4  # capas verticales por columna de suelo (fijo para poder vectorizar)

# --------------------------------------------------------------------------- #
# [DATASET] Rawls et al. (1983), hoja Rawls_1983:  (psi, Ks, theta_e)
# --------------------------------------------------------------------------- #
RAWLS_1983: dict[str, tuple[float, float, float]] = {
    "Clay":            (316.3,   0.3, 0.385),
    "Silty Clay":      (292.2,   0.5, 0.423),
    "Sandy Clay":      (239.0,   0.6, 0.321),
    "Clay Loam":       (208.8,   1.0, 0.309),
    "Silty Clay Loam": (273.0,   1.0, 0.432),
    "Sandy Clay Loam": (218.5,   1.5, 0.330),
    "Silt Loam":       (166.8,   3.4, 0.486),
    "Loam":            ( 88.9,   7.6, 0.434),
    "Sandy Loam":      (110.1,  10.9, 0.412),
    "Loamy Sand":      ( 61.3,  29.9, 0.401),
    "Sand":            ( 49.5, 117.8, 0.417),
}

# --------------------------------------------------------------------------- #
# [SUPUESTO] Materiales sin equivalente en Rawls.
# El dataset dice que Leptosoles "debe modelarse como material somero propio del
# simulador": aquí se define. Valores ilustrativos, calibrar si hay datos locales.
# --------------------------------------------------------------------------- #
CUSTOM_MATERIALS: dict[str, tuple[float, float, float]] = {
    "Leptosol":        (60.0, 15.0, 0.30),   # suelo somero, pedregoso, drena bien
    "Roca fracturada": (150.0, 0.5, 0.05),   # roca / tepetate: casi impermeable
}


# --------------------------------------------------------------------------- #
# [DATASET] Grupos de suelo de Aguascalientes (hoja Relacion_Ags)
# --------------------------------------------------------------------------- #
@dataclass
class SoilDataset:
    rawls: dict[str, tuple[float, float, float]]
    pct: dict[str, float]            # código WRB -> % de superficie estatal
    texture: dict[str, str | None]   # código WRB -> textura Rawls (None = sin equivalente)
    names: dict[str, str]            # código WRB -> nombre


AGS = SoilDataset(
    rawls=RAWLS_1983,
    pct={"PH": 33.69, "DU": 19.57, "LP": 18.48, "RG": 9.02, "CM": 5.61,
         "KS": 3.08, "LV": 2.72, "CL": 0.97, "FL": 0.67},
    texture={"PH": "Loam", "DU": "Clay", "LP": None, "RG": "Sandy Loam",
             "CM": "Loam", "KS": "Loam", "LV": "Clay Loam", "CL": "Loam",
             "FL": "Silt Loam"},
    names={"PH": "Phaeozems", "DU": "Durisoles", "LP": "Leptosoles",
           "RG": "Regosoles", "CM": "Cambisoles", "KS": "Kastanozems",
           "LV": "Luvisoles", "CL": "Calcisoles", "FL": "Fluvisoles"},
)


def load_excel(path: str) -> SoilDataset:
    """Lee el xlsx del proyecto (valores calculados). Si cambias la columna D
    de Relacion_Ags (textura asignada), el simulador usa tu nueva asignación."""
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True, read_only=True)

    rawls: dict[str, tuple[float, float, float]] = {}
    for row in wb["Rawls_1983"].iter_rows(min_row=2, max_row=12, values_only=True):
        name, psi, ks, phi = row[:4]
        if name and isinstance(psi, (int, float)):
            rawls[str(name)] = (float(psi), float(ks), float(phi))

    pct, tex, names = {}, {}, {}
    for row in wb["Relacion_Ags"].iter_rows(min_row=2, max_row=40, values_only=True):
        code, name, p, t = row[:4]
        if not code or not isinstance(p, (int, float)):
            continue
        pct[str(code)] = float(p)
        names[str(code)] = str(name)
        tex[str(code)] = t if t in rawls else None
    return SoilDataset(rawls=rawls, pct=pct, texture=tex, names=names)


# --------------------------------------------------------------------------- #
# [SUPUESTO] Perfil vertical de cada grupo: 4 capas (superficial -> sustrato).
#   "@"  = usa la textura que el dataset asigna a ese grupo (Relacion_Ags!D)
#   otro = nombre de textura Rawls o de CUSTOM_MATERIALS
# fractions = fracción de la profundidad total que ocupa cada capa (suman 1).
# depth_mm = (media, desviación) de la profundidad total del suelo.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Profile:
    depth_mm: tuple[float, float]
    textures: tuple[str, str, str, str]
    fractions: tuple[float, float, float, float]


PROFILES: dict[str, Profile] = {
    # Suelos medios y profundos: la textura del dataset en el cuerpo del perfil
    "PH": Profile((1200, 250), ("@", "@", "@", "Sandy Loam"), (.15, .25, .35, .25)),
    "KS": Profile((1200, 250), ("@", "@", "@", "Sandy Loam"), (.15, .25, .35, .25)),
    "CM": Profile((1100, 250), ("@", "@", "@", "Sandy Loam"), (.15, .25, .35, .25)),
    "CL": Profile((1000, 250), ("@", "@", "@", "@"),          (.15, .25, .35, .25)),
    # Regosol: material suelto y permeable
    "RG": Profile((1000, 250), ("@", "@", "@", "Loamy Sand"), (.15, .25, .35, .25)),
    # Fluvisol: depósito aluvial, arena en el fondo
    "FL": Profile((1500, 300), ("@", "@", "@", "Sandy Loam"), (.15, .25, .35, .25)),
    # Luvisol: arcilla acumulada en el subsuelo (horizonte argílico)
    "LV": Profile((1300, 250), ("Loam", "@", "@", "Loam"),    (.15, .30, .35, .20)),
    # Durisol: suelo medio sobre duripán ("@" = Clay del dataset) y roca debajo.
    # El dataset avisa que el limitante NO es la textura sino la capa cementada.
    "DU": Profile((900, 200),  ("Loam", "Loam", "@", "Roca fracturada"), (.20, .25, .35, .20)),
    # Leptosol: somero (<~30 cm) sobre roca
    "LP": Profile((300, 100),  ("Leptosol", "Leptosol", "Roca fracturada", "Roca fracturada"),
                  (.30, .30, .20, .20)),
}

# [SUPUESTO] Posición en el paisaje (catena), de zonas bajas a cimas.
# Fluvisoles en el fondo del valle, Leptosoles en lomas y cimas.
CATENA_ORDER = ["FL", "PH", "KS", "CM", "LV", "CL", "DU", "RG", "LP"]


# --------------------------------------------------------------------------- #
# [DATASET] Confianza de la equivalencia WRB -> textura USDA (hoja Relacion_Ags, columna I).
# Superficie estatal: Media 45.10 % (PH, CM, KS, LV) · Baja 30.23 % (DU, RG, CL, FL) · N/A 18.48 % (LP).
# La equivalencia es un supuesto de los autores, no un dato medido.
# --------------------------------------------------------------------------- #
CONFIANZA_ASIGNACION: dict[str, str] = {
    "PH": "Media", "DU": "Baja", "LP": "N/A", "RG": "Baja", "CM": "Media",
    "KS": "Media", "LV": "Media", "CL": "Baja", "FL": "Baja",
}
