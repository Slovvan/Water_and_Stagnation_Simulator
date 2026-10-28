"""
soil_rules.py
=============
Reglas del CA para un medio con aire + suelo poroso (+ roca). Generalizan tus gravity_Rule y
lateral_Flow_Rule: si TODAS las celdas son aire (capacidad 1) dan exactamente el mismo resultado.

Estado: grid[r, c] = agua en la celda (unidades de dz_mm). Capacidad: aire 1 · suelo theta_e · roca 0.

GRAVEDAD (vertical)
  flujo = min( agua que puede salir de arriba,  velocidad de arriba,  velocidad de abajo,  espacio libre abajo )
  - Suelo arriba: solo sale el exceso sobre capacidad de campo, a K = Ks·Se² (Se = saturación efectiva).
  - Frente húmedo (Green-Ampt): si desde arriba llega agua libre (aire) o suelo saturado, el flujo hacia
    una celda de suelo se multiplica por (1 + psi/L), con L = profundidad del centro de esa celda bajo la superficie. Es el mismo Ks·(1 + psi·Δθ/F) de tu hoja, porque F/Δθ = profundidad del frente.
  - Una capa poco permeable (duripán, roca) limita el paso con SU Ks: el agua se acumula encima.

LATERAL (horizontal)
  - Aire-aire y suelo-suelo: las celdas del par tienden a igualar su saturación (con tu patrón de pares
    pares/impares), limitado por la conductividad del suelo. El aire solo se mueve si está «apoyado».
  - Aire-suelo: el agua libre se mete al suelo por el costado (talud) a su Ks; el suelo no «suda» al aire.
"""
import numpy as np

SAT_FROM = 0.90   # a partir de esta saturación una celda empieza a transmitir como columna saturada (rampa hasta 1.0)
EPS = 1e-9


class SoilGravityRule:
    def __init__(self, medium, scale, substeps=1, flow_rate=1.0, bottom_drain=True):
        self.m, self.flow_rate, self.bottom_drain = medium, flow_rate, bottom_drain
        kind, cap = medium.kind, medium.cap
        soil, air = kind == 1, kind == 0
        # Ks en unidades de agua por subpaso
        ks = medium.ks_mm_h * scale.hours_per_step / scale.dz_mm / substeps
        sfc = medium.sfc
        cap_safe = np.where(cap > 0, cap, 1.0)
        up, lo = slice(None, -1), slice(1, None)
        self.cap_up, self.cap_lo = cap[up], cap[lo]
        self.cap_up_safe = cap_safe[up]
        self.soil_up, self.soil_lo, self.air_up = soil[up], soil[lo], air[up]
        self.ks_up, self.ks_lo = ks[up], ks[lo]
        self.sfc_up = sfc[up]
        L = medium.depth_mm[lo] + 0.5 * scale.dz_mm                      # mm hasta el centro de la celda receptora (frente)
        self.boost = np.where(self.soil_lo, 1.0 + medium.psi_mm[lo] / L, 1.0)
        self.ks_last, self.sfc_last, self.cap_last = ks[-1], sfc[-1], cap[-1]
        self.soil_last = soil[-1]

    def apply(self, current):
        up, lo = current[:-1], current[1:]
        free = np.maximum(self.cap_lo - lo, 0.0)
        S_up = up / self.cap_up_safe
        excess = np.where(self.soil_up, np.maximum(up - self.sfc_up * self.cap_up, 0.0), up)

        # peso de «columna saturada / agua libre arriba»: 1 para aire, rampa 0→1 entre SAT_FROM y 1 para suelo
        w = np.where(self.air_up, 1.0, np.clip((S_up - SAT_FROM) / (1.0 - SAT_FROM), 0.0, 1.0))
        extra = (self.boost - 1.0) * w                                    # ganancia Green-Ampt (1 + psi/L) − 1
        Se = np.clip((S_up - self.sfc_up) / (1.0 - self.sfc_up), 0.0, 1.0)
        donor = np.where(self.soil_up, self.ks_up * (Se ** 2 + extra), self.flow_rate)
        recv = np.where(self.soil_lo, self.ks_lo * (1.0 + extra), self.flow_rate)

        flow = np.minimum(np.minimum(excess, free), np.minimum(donor, recv))
        np.maximum(flow, 0.0, out=flow)
        updated = current.copy()
        updated[:-1] -= flow
        updated[1:] += flow
        return updated

    def drain_bottom(self, current):
        """Percolación profunda: la última fila drena fuera del dominio a su Ks. Devuelve (grid, agua_perdida)."""
        if not self.bottom_drain:
            return current, 0.0
        last = current[-1]
        S = last / np.where(self.cap_last > 0, self.cap_last, 1.0)
        Se = np.clip((S - self.sfc_last) / (1.0 - self.sfc_last + EPS), 0.0, 1.0)
        out = np.minimum(self.ks_last * Se ** 2, np.maximum(last - self.sfc_last * self.cap_last, 0.0))
        out = np.where(self.soil_last, out, 0.0)
        updated = current.copy()
        updated[-1] -= out
        return updated, float(out.sum())


class SoilLateralRule:
    def __init__(self, medium, scale, substeps=1, flow_rate=1.0, anisotropy=1.0, edge_runoff=False):
        self.m, self.flow_rate, self.edge_runoff = medium, flow_rate, edge_runoff
        n_rows, n_cols = medium.kind.shape
        kind, cap, sfc = medium.kind, medium.cap, medium.sfc
        ks = medium.ks_mm_h * scale.hours_per_step / scale.dz_mm / substeps * anisotropy
        soil, air = kind == 1, kind == 0
        ground_below = np.ones_like(soil)
        ground_below[:-1] = (kind[1:] != 0)
        static_support = soil | ground_below                            # suelo, o aire sobre suelo/roca/piso
        self.pre = []
        for parity in (0, 1):
            L, R = slice(parity, n_cols - 1, 2), slice(parity + 1, n_cols, 2)
            cl, cr = cap[:, L], cap[:, R]
            p = dict(L=L, R=R, cl=cl, cr=cr, tot=np.where(cl + cr > 0, cl + cr, 1.0),
                     valid=(cl > 0) & (cr > 0),
                     soil_l=soil[:, L], soil_r=soil[:, R], air_l=air[:, L], air_r=air[:, R],
                     sfc_l=sfc[:, L], sfc_r=sfc[:, R], ks_l=ks[:, L], ks_r=ks[:, R],
                     cls=np.where(cl > 0, cl, 1.0), crs=np.where(cr > 0, cr, 1.0),
                     sup_l=static_support[:, L], sup_r=static_support[:, R])
            p["mixed"] = (p["air_l"] & p["soil_r"]) | (p["soil_l"] & p["air_r"])
            self.pre.append(p)
        self.air_edge = np.stack([air[:, 0], air[:, -1]], axis=1)       # (filas, 2)

    def apply(self, current, step):
        p = self.pre[step % 2]
        L, R = p["L"], p["R"]
        wl, wr = current[:, L], current[:, R]

        below = np.empty_like(current)                                   # «apoyada»: debajo hay agua llena
        below[:-1] = current[1:]
        below[-1] = 1.0
        active = p["sup_l"] | p["sup_r"] | (below[:, L] >= 1.0 - EPS) | (below[:, R] >= 1.0 - EPS)

        f_eq = wl - (wl + wr) * p["cl"] / p["tot"]                       # >0: izquierda -> derecha
        f_mix = np.where(p["air_l"], wl, -wr)                            # solo el agua libre entra al suelo
        f = np.where(p["valid"], np.where(p["mixed"], f_mix, f_eq), 0.0)

        pos = f > 0
        Sl, Sr = wl / p["cls"], wr / p["crs"]
        Sel = np.clip((Sl - p["sfc_l"]) / (1.0 - p["sfc_l"] + EPS), 0.0, 1.0)
        Ser = np.clip((Sr - p["sfc_r"]) / (1.0 - p["sfc_r"] + EPS), 0.0, 1.0)
        avail = np.where(pos,
                         np.where(p["soil_l"], np.maximum(wl - p["sfc_l"] * p["cl"], 0.0), wl),
                         np.where(p["soil_r"], np.maximum(wr - p["sfc_r"] * p["cr"], 0.0), wr))
        donor = np.where(pos,
                         np.where(p["soil_l"], p["ks_l"] * Sel ** 2, self.flow_rate),
                         np.where(p["soil_r"], p["ks_r"] * Ser ** 2, self.flow_rate))
        recv = np.where(pos,
                        np.where(p["soil_r"], p["ks_r"], self.flow_rate),
                        np.where(p["soil_l"], p["ks_l"], self.flow_rate))
        free = np.where(pos, p["cr"] - wr, p["cl"] - wl)

        mag = np.minimum(np.minimum(np.abs(f), avail), np.minimum(np.minimum(donor, recv), np.maximum(free, 0.0)))
        flow = np.sign(f) * mag * active
        updated = current.copy()
        updated[:, L] -= flow
        updated[:, R] += flow
        return updated

    def drain_edges(self, current):
        """Escorrentía limitada a flow_rate por borde y subpaso."""
        if not self.edge_runoff:
            return current, 0.0
        updated = current.copy()
        removed = 0.0
        for k, col in enumerate((0, -1)):
            remaining = self.flow_rate
            for row in range(updated.shape[0] - 1, -1, -1):
                if not self.air_edge[row, k]:
                    continue
                take = min(updated[row, col], remaining)
                updated[row, col] -= take
                removed += take
                remaining -= take
                if remaining <= EPS:
                    break
        return updated, removed
