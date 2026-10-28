"""Corte sincronizado: la vista lateral de una línea del mapa aéreo (por defecto, su lado derecho).

NO es otra simulación: dibuja el estado del MISMO motor aéreo a lo largo de la línea (relieve, horizontes del suelo,
frente de humedad por la infiltración acumulada F y agua superficial H). Por eso está sincronizado con el mapa en todo
momento y no cuesta pasos de física. El eje vertical está exagerado (una celda mide 1 m de ancho y la escala vertical es
de unos 3 m en 800 px).
"""
from __future__ import annotations

import numpy as np

from aereo.motor import MotorAereo
from app.render import CIELO, color_agua_mm, mojar
from core.escenario import Escenario
from core.mapa import Mapa
from corte.medio import _perfil
from water.soil_data import AGS
from water.soil_terrain import LAYER_SHADE, MATERIAL_RGB, get_material

FILAS = 400                    # resolución vertical de la imagen (se escala a 800 px)
MARGEN_AGUA_M = 0.15           # espacio sobre la cima para lluvia y charcos


class CorteSincronizado:
    def __init__(self, mapa: Mapa, esc: Escenario, eje: str = "col", indice: int | None = None):
        n = mapa.n
        self.mapa, self.eje, self.n = mapa, eje, n
        self.indice = n - 1 if indice is None else int(np.clip(indice, 0, n - 1))
        idx = np.arange(n)
        self.rr, self.cc = (idx, np.full(n, self.indice)) if eje == "col" else (np.full(n, self.indice), idx)
        rr, cc = self.rr, self.cc
        self.z = mapa.elevacion_m[rr, cc].astype(float)
        self.depth = mapa.profundidad_mm[rr, cc].astype(float)
        sid = mapa.soil_id[rr, cc]

        frac, th_l, colores, self.materiales = np.zeros((4, n)), np.zeros((4, n)), np.zeros((5, n, 3)), []
        por_codigo = {}
        for k, code in enumerate(mapa.codes):
            if k not in sid:
                continue
            mats, fr = _perfil(code, mapa, AGS)
            por_codigo[k] = (mats, np.array(fr), np.array([get_material(m, AGS)[2] for m in mats]),
                             np.array([np.array(MATERIAL_RGB.get(m, (0.5, 0.5, 0.5))) * LAYER_SHADE[l] * 255.0
                                       for l, m in enumerate(mats)]))
        for c in range(n):
            mats, fr, th, rgb = por_codigo[sid[c]]
            frac[:, c], th_l[:, c], colores[:, c] = fr, th[:4], rgb
        self.colores = colores
        self.bounds = np.cumsum(frac, axis=0) * self.depth[None, :]                    # mm, fondo de cada horizonte
        self.espacio_cum = np.cumsum((1.0 - esc.s0) * th_l * frac * self.depth[None, :], axis=0)   # mm de agua que caben
        self.s0 = esc.s0

        # --- geometría vertical fija (no cambia con el tiempo) ---
        zmax, zmin = float(self.z.max()), float(self.z.min())
        self.top = zmax + MARGEN_AGUA_M
        self.bottom = zmin - float(self.depth.max()) / 1000.0 - 0.10
        self.dzv = (self.top - self.bottom) / FILAS                                    # m por fila de imagen
        e = self.top - (np.arange(FILAS) + 0.5) * self.dzv
        d = (self.z[None, :] - e[:, None]) * 1000.0                                   # mm bajo la superficie
        self.d = d
        self.aire = d < 0
        capa = np.clip(sum((d >= self.bounds[k][None, :]).astype(int) for k in range(4)), 0, 4)
        self.base = np.where(self.aire[..., None], 0.0, colores[capa, np.arange(n)[None, :]]).astype(np.float32)
        self.cielo = CIELO.astype(np.float32)
        self.fila_sup = (self.top - self.z) / self.dzv                                  # fila (decimal) de la superficie
        suelo = ~self.aire & (self.d < self.depth[None, :])
        self.d_suelo = np.where(suelo, self.d, np.inf).astype(np.float32)               # fuera del suelo nunca está «mojado»
        w_seco, w_mojado = self.s0 ** 1.5, 1.0
        roca = ~self.aire & ~suelo                                                      # sustrato bajo el perfil: sin teñir
        tinte = lambda m: np.where(suelo[..., None], mojar(self.base, m), np.where(roca[..., None], self.base, self.cielo))
        self.img_seca = np.clip(tinte(w_seco), 0, 255).astype(np.uint8)
        self.img_mojada = np.clip(tinte(w_mojado), 0, 255).astype(np.uint8)
        self._filas = np.arange(FILAS, dtype=np.float32)[:, None] + 0.5
        self._hv = np.zeros(n)                                                          # agua suavizada para dibujar
        todos = [m for k in sorted(por_codigo) for m in por_codigo[k][0]]                 # todos los horizontes presentes
        self.mats_col = [por_codigo[sid[c]][0] for c in range(n)]
        self.codigo_col = [self.mapa.codes[sid[c]] for c in range(n)]
        self.leyenda = [(f"{m} (Ks {get_material(m, AGS)[1]:g})", tuple(MATERIAL_RGB.get(m, (0.5, 0.5, 0.5)))) for m in dict.fromkeys(todos)]
        self.exageracion = (self.n * self.mapa.dx_m) / (self.top - self.bottom)   # px verticales por m / px horizontales por m

    # ---- estado del motor -> profundidad del frente ------------------------
    def frente_mm(self, F: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Profundidad (mm) hasta donde llega el agua infiltrada: se llenan los horizontes de arriba hacia abajo."""
        Fs = np.maximum(F - D, 0.0)
        prof = np.zeros(self.n)
        pb = np.zeros(self.n)
        pc = np.zeros(self.n)
        for l in range(4):
            b, c = self.bounds[l], self.espacio_cum[l]
            esp = c - pc
            dentro = (Fs > pc) & (Fs <= c)
            f = np.divide(Fs - pc, esp, out=np.zeros(self.n), where=esp > 0)
            prof = np.where(dentro, pb + f * (b - pb), prof)
            pb, pc = b, c
        return np.where(Fs > self.espacio_cum[3], self.depth, prof)

    def datos(self, motor: MotorAereo):
        return motor.F[self.rr, self.cc], motor.D[self.rr, self.cc], motor.H[self.rr, self.cc]

    def imagen(self, motor: MotorAereo, suavizar: bool = False):
        """Devuelve (imagen RGB uint8 de FILAS x n, superficie_y por columna en filas de imagen).
        suavizar=True (la interfaz) filtra en el tiempo el agua que se dibuja para que la película fina no parpadee."""
        F, D, H = self.datos(motor)
        if suavizar:
            self._hv = self._hv + 0.4 * (H - self._hv)
            H = np.where(self._hv > 0.1, self._hv, 0.0)
        mojado = self.d_suelo < self.frente_mm(F, D).astype(np.float32)[None, :]
        img = np.where(mojado[..., None], self.img_mojada, self.img_seca)
        # agua sobre el suelo, a escala (sin grosor mínimo: los charcos nivelados quedan horizontales)
        filas_agua = H / 1000.0 / self.dzv
        con_agua = np.nonzero(H > 0.05)[0]
        if con_agua.size:
            col = np.clip(color_agua_mm(H[con_agua]), 0, 255).astype(np.uint8)
            r = self._filas
            sup = self.fila_sup[con_agua][None, :]
            agua = (r < sup) & (r > sup - filas_agua[con_agua][None, :])
            sub = img[:, con_agua]
            img[:, con_agua] = np.where(agua[..., None], col[None, :, :], sub)
            # película fina (menos de una fila): se pinta la fila del suelo para que se vea que está mojado
            fina = filas_agua[con_agua] < 1.0
            if fina.any():
                cf = con_agua[fina]
                fr = np.clip(self.fila_sup[cf].astype(int), 0, FILAS - 1)
                img[fr, cf] = col[fina]
        return img, self.fila_sup - filas_agua

    def info(self, motor: MotorAereo, j: int, fila: int) -> list[str]:
        """Texto del popup al señalar la columna j (celda de la línea) y la fila de imagen `fila`."""
        j, fila = int(np.clip(j, 0, self.n - 1)), int(np.clip(fila, 0, FILAS - 1))
        r, c = int(self.rr[j]), int(self.cc[j])
        F, D, H = motor.F[r, c], motor.D[r, c], motor.H[r, c]
        cod = self.codigo_col[j]
        nombre = self.mapa.nombres.get(cod, cod)
        cab = f"{nombre} ({cod}) · celda ({r}, {c}) · {j * self.mapa.dx_m:.0f} m"
        sup = self.fila_sup[j]
        if self.aire[fila, j]:
            filas_agua = H / 1000.0 / self.dzv
            if fila + 0.5 > sup - filas_agua:
                return [cab, f"Agua superficial: {H:.1f} mm", f"Nivel del agua: {self.z[j] + H / 1000.0:.3f} m (terreno {self.z[j]:.3f} m)"]
            return [cab, f"Aire · elevación {self.top - (fila + 0.5) * self.dzv:.2f} m", f"Agua en la superficie de esta celda: {H:.2f} mm"]
        d = float(self.d[fila, j])
        capa = int(np.clip(sum(d >= self.bounds[k][j] for k in range(4)), 0, 4))
        mat = self.mats_col[j][capa]
        psi, ks, th = get_material(mat, AGS)
        ini = 0.0 if capa == 0 else float(self.bounds[capa - 1][j])
        fin = float(self.bounds[capa][j]) if capa < 4 else float("inf")
        tramo = f"{ini:.0f}–{fin:.0f} mm" if capa < 4 else f"desde {ini:.0f} mm"
        fr = float(self.frente_mm(motor.F[self.rr, self.cc], motor.D[self.rr, self.cc])[j])
        if capa == 4:
            estado = "Roca: sin infiltración propia"
        else:
            estado = f"Mojado (frente a {fr:.0f} mm)" if d < fr else f"Seco (S0 = {100 * self.s0:.0f}%)"
        return [cab, f"{mat} · {'sustrato' if capa == 4 else f'horizonte {capa + 1} de 4'} · {tramo}",
                f"Ks {ks:.2f} mm/h · ψ {psi:.0f} mm · θe {th:.3f}", f"Profundidad bajo el terreno: {d:.0f} mm", estado]

    def resumen(self, motor: MotorAereo) -> dict:
        F, D, H = self.datos(motor)
        frente = self.frente_mm(F, D)
        cap = self.espacio_cum[3]
        nombres = {}
        for c in range(self.n):
            nom = self.mapa.nombres.get(self.mapa.codes[self.mapa.soil_id[self.rr[c], self.cc[c]]], "?")
            nombres[nom] = nombres.get(nom, 0) + 1
        return {"suelos": sorted(((v * 100.0 / self.n, k) for k, v in nombres.items()), reverse=True),
                "frente_medio_mm": float(frente.mean()), "frente_max_mm": float(frente.max()),
                "infiltrado_medio_mm": float(F.mean()), "agua_media_mm": float(H.mean()), "agua_max_mm": float(H.max()),
                "encharcado_pct": 100.0 * float((H > 1.0).mean()), "perfiles_llenos": int(((F - D) >= cap - 1e-6).sum()),
                "desnivel_m": float(self.z.max() - self.z.min())}
