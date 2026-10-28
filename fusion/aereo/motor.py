"""
Motor aéreo incremental (vista en planta). Mismo reloj, datos y unidades que el corte.

Por paso (dt_s):
  1. Lluvia: i·dt (mm) sobre cada celda (con ruido relativo opcional) mientras t < duracion.
  2. Infiltración Green-Ampt con F acumulada: la capacidad del paso sale de integrar dF/dt = Ks(1 + ψΔθ/F),
        g(F) = F − A·ln(1 + F/A),  g(F_nueva) = g(F) + Ks·dt,   A = ψ·Δθ,  Δθ = (1 − S0)·θe
     y se infiltra min(agua disponible en la celda, capacidad). El encharcamiento aparece solo cuando la
     lluvia supera la capacidad, y el agua encharcada o recibida de otras celdas se sigue infiltrando.
     Si el perfil se llena (almacenamiento), solo pasa la fuga del sustrato (k_fondo): exceso de saturación.
  3. Reparto superficial conservativo hacia vecinos más bajos (8 vecinos), con bordes abiertos o cerrados.

Balance (mm·celda):  lluvia = agua superficial + infiltración acumulada + salida por bordes.
El reparto superficial es un reparto rápido por nivel de agua, no un cálculo de velocidad de escorrentía.
"""
from __future__ import annotations

import numpy as np

try:                                    # scipy es opcional: acelera el etiquetado de charcos
    from scipy.ndimage import label as _label_scipy
except Exception:                       # pragma: no cover
    _label_scipy = None

from core.escenario import Escenario
from core.mapa import Mapa

UMBRAL_CHARCO_MM = 0.02        # primer charco medible (mismo criterio que el corte)
UMBRAL_AREA_MM = 1.0           # área encharcada: > 1 mm
POZA_MM = 2.0                  # celdas con más de 2 mm forman charcos y se nivelan (superficie horizontal)
NIVELAR_CADA = 2               # pasos entre nivelaciones de charcos
OMEGA = 0.14                   # conductancia del reparto: sum(pesos)·OMEGA < 1 garantiza estabilidad
_D = 1.0 / np.sqrt(2.0)
VECINOS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, _D), (-1, 1, _D), (1, -1, _D), (1, 1, _D)]
assert OMEGA * sum(w for _, _, w in VECINOS) < 1.0


def _g(F, A):
    return F - A * np.log1p(F / A)


def capacidad_ga(F, A, ks_dt):
    """Infiltración máxima del paso (mm) por Green-Ampt: F_nueva − F, con g(F_nueva) = g(F) + Ks·dt.
    g es creciente y convexa: Newton desde sqrt(2·A·T) cae a la derecha de la raíz y converge monótono (4 pasadas
    dan un residuo relativo < 1e-11 en el rango de A y Ks·dt del proyecto)."""
    T = _g(F, A) + ks_dt
    Fn = np.sqrt(2.0 * A * T) + 1e-300
    for _ in range(4):
        Fn = Fn - (_g(Fn, A) - T) * (Fn + A) / Fn
    return np.maximum(Fn - F, 0.0)


def etiquetar(mascara: np.ndarray):
    """Componentes 4-conexas de una máscara booleana. Devuelve (etiquetas 0..k, k)."""
    if _label_scipy is not None:
        return _label_scipy(mascara)
    filas, cols = np.nonzero(mascara)
    if filas.size == 0:
        return np.zeros(mascara.shape, int), 0
    r0, r1, c0, c1 = filas.min(), filas.max() + 1, cols.min(), cols.max() + 1
    m = mascara[r0:r1, c0:c1]
    lab = np.where(m, np.arange(1, m.size + 1).reshape(m.shape), 0)
    for _ in range(m.shape[0] + m.shape[1]):                     # propagación del mínimo; converge en ~el diámetro
        nuevo = lab.copy()
        for eje, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            vec = np.roll(lab, sh, axis=eje)
            if eje == 0:
                (vec[0, :] if sh == 1 else vec[-1, :])[...] = 0
            else:
                (vec[:, 0] if sh == 1 else vec[:, -1])[...] = 0
            nuevo = np.where(m & (vec > 0), np.minimum(nuevo, np.where(vec > 0, vec, nuevo)), nuevo)
        if np.array_equal(nuevo, lab):
            break
        lab = nuevo
    _, inv = np.unique(lab[m], return_inverse=True)
    out = np.zeros(mascara.shape, int)
    sub = np.zeros(m.shape, int)
    sub[m] = inv + 1
    out[r0:r1, c0:c1] = sub
    return out, int(inv.max()) + 1


def nivel_de_charco(z: np.ndarray, volumen_m: float) -> float:
    """Nivel horizontal L tal que sum(max(L − z, 0)) = volumen (m·celda): llenado de un charco sobre el terreno z."""
    zs = np.sort(z)
    csum = np.cumsum(zs)
    k = np.arange(1, zs.size + 1)
    vol_en = k * zs - csum                      # volumen necesario para subir el nivel hasta zs[k-1]
    m = int(np.searchsorted(vol_en, volumen_m, side="right"))
    return float((volumen_m + csum[m - 1]) / m)


class MotorAereo:
    def __init__(self, mapa: Mapa, esc: Escenario):
        self.mapa, self.esc = mapa, esc
        n = mapa.n
        self.n = n
        self.dt_h = esc.dt_s / 3600.0
        self.z = np.asarray(mapa.elevacion_m, float)
        self.ks = np.asarray(mapa.ks_mm_h, float)
        self.A = np.asarray(mapa.psi_mm, float) * (1.0 - esc.s0) * np.asarray(mapa.theta_e, float)
        self.cap = (1.0 - esc.s0) * np.asarray(mapa.almacen_sat_mm, float)   # espacio del perfil [mm]
        self.kb_dt = float(mapa.k_fondo_mm_h) * self.dt_h
        self.ks_dt = self.ks * self.dt_h
        self.rng = np.random.default_rng(esc.semilla_lluvia)
        self.intensidad_mm_h = float(esc.intensidad_mm_h)
        self.duracion_min = float(esc.duracion_min)
        self.lluvia_activa = True          # interruptor en vivo: la lluvia se puede detener y reanudar
        self.t_fin_lluvia = 0.0            # último minuto con lluvia
        self.paso = 0
        self.H = np.zeros((n, n))          # agua superficial [mm]
        self.F = np.zeros((n, n))          # infiltración acumulada [mm]
        self.D = np.zeros((n, n))          # fuga profunda acumulada [mm] (parte de F)
        self.t_charco = np.full((n, n), np.inf)   # min del primer charco en cada celda
        self.lluvia_cel = 0.0              # lluvia acumulada (mm·celda)
        self.P = np.zeros((n, n))          # lluvia acumulada por celda [mm]
        self.recibido = np.zeros((n, n))   # agua que la celda recibió de sus vecinas, acumulada [mm]
        self.enviado = np.zeros((n, n))    # agua que la celda mandó a sus vecinas, acumulada [mm]
        self.salida_lado = np.zeros(4)     # salida por borde (mm·celda): arriba, abajo, izquierda, derecha
        self.salida_cel = 0.0              # salida por bordes (mm·celda)
        self.pico_sup = 0.0                # mayor agua superficial total (mm·celda)
        self.t_pico_min = 0.0
        self.t_mitad_min = np.nan          # min hasta que el agua superficial baja a la mitad del pico (tras llover)
        # anillo fantasma de 1 celda alrededor del mapa (estático):
        #   bordes abiertos -> el terreno continúa con su pendiente (2·z_borde − z_interior): sale agua solo si baja hacia
        #                      afuera o si el charco rebasa ese nivel;
        #   bordes cerrados -> +inf: ningún flujo hacia afuera.
        z = self.z
        if esc.bordes_abiertos:
            zp = np.pad(z, 1, mode="reflect", reflect_type="odd")        # extrapolación lineal: 2·z0 − z1
            self._lp = zp.copy()
        else:
            self._lp = np.full((n + 2, n + 2), np.inf)
        self._hp = np.zeros((n + 2, n + 2))
        self.aporte = np.zeros((n, n))     # mm por paso que cada celda recibe de sus vecinas (suavizado ~1 min)
        self._aporte_paso = np.zeros((n, n))
        self.envio = np.zeros((n, n))      # mm por paso que cada celda manda a sus vecinas (suavizado ~1 min)
        self._envio_paso = np.zeros((n, n))
        self._buf = [np.empty((n, n)) for _ in VECINOS]
        self._tot = np.empty((n, n))

    # ---- control en vivo ---------------------------------------------------
    def set_intensidad(self, mm_h: float):
        if not np.isfinite(mm_h) or mm_h < 0:
            raise ValueError("La intensidad debe ser un número no negativo (mm/h)")
        self.intensidad_mm_h = float(mm_h)

    def set_duracion(self, minutos: float):
        """Minutos que llueve (desde t = 0). float('inf') = lluvia continua."""
        if np.isnan(minutos) or minutos < 0:
            raise ValueError("La duración debe ser un número no negativo (min), o inf para lluvia continua")
        self.duracion_min = float(minutos)

    def prolongar(self, minutos: float):
        """Alarga la lluvia `minutos` más a partir de ahora (y la reanuda si estaba detenida)."""
        self.lluvia_activa = True
        if np.isfinite(self.duracion_min):
            self.duracion_min = max(self.duracion_min, self.tiempo_min) + float(minutos)

    def detener_lluvia(self):
        self.lluvia_activa = False

    def reanudar_lluvia(self, continua: bool = True):
        """Vuelve a llover; si ya había pasado la duración, sigue sin límite (continua=True) o por 30 min."""
        self.lluvia_activa = True
        if self.tiempo_min >= self.duracion_min:
            self.duracion_min = float("inf") if continua else self.tiempo_min + 30.0

    @property
    def aporte_neto(self) -> np.ndarray:
        """mm por paso que cada celda recibe de más (recibido − enviado, >= 0): lo que se acumula en ella."""
        return np.maximum(self.aporte - self.envio, 0.0)

    @property
    def tiempo_min(self) -> float:
        return self.paso * self.esc.dt_s / 60.0

    @property
    def llueve(self) -> bool:
        return self.lluvia_activa and self.tiempo_min < self.duracion_min

    # ---- física ------------------------------------------------------------
    def _lluvia(self):
        r = self.intensidad_mm_h * self.dt_h
        if r <= 0 or not self.llueve:
            return
        if self.esc.ruido_lluvia > 0:
            f = np.clip(1.0 + self.esc.ruido_lluvia * self.rng.standard_normal(self.H.shape), 0.0, None)
            self.H += r * f
            self.P += r * f
            self.lluvia_cel += r * float(f.sum())
        else:
            self.H += r
            self.P += r
            self.lluvia_cel += r * self.n * self.n

    def _infiltrar(self):
        espacio = np.maximum(self.cap - (self.F - self.D), 0.0)
        lim = espacio + self.kb_dt
        H, F = self.H, self.F
        # si H <= f(F+H)·dt la capacidad de Green-Ampt (que es >= f(F+H)·dt) no limita: se infiltra todo lo que haya
        piso = self.ks_dt * (1.0 + self.A / (F + H + 1e-12))
        dif = np.nonzero(H > piso)
        infil = np.minimum(H, lim)
        if dif[0].size:
            cap_ga = capacidad_ga(F[dif], self.A[dif], self.ks_dt[dif])
            infil[dif] = np.minimum(H[dif], np.minimum(cap_ga, lim[dif]))
        self.H -= infil
        self.F += infil
        lleno = (self.F - self.D) >= self.cap - 1e-9            # perfil saturado: solo pasa la fuga del sustrato
        fuga = np.where(lleno, np.minimum(self.kb_dt, self.F - self.D), 0.0)
        self.D += fuga

    def _rutear(self):
        n, H = self.n, self.H
        L = self.z + H * 1e-3
        Lp, Hp, tot = self._lp, self._hp, self._tot
        Lp[1:-1, 1:-1] = L
        tot[:] = 0.0
        coef = 1e3 * OMEGA
        for (dr, dc, w), d in zip(VECINOS, self._buf):
            np.subtract(L, Lp[1 + dr:1 + dr + n, 1 + dc:1 + dc + n], out=d)      # (+inf en el anillo cerrado -> negativo)
            np.maximum(d, 0.0, out=d)
            d *= coef * w                                                          # mm que querría salir hacia ese vecino
            tot += d
        escala = np.where(tot > H, H / np.where(tot > 0, tot, 1.0), 1.0)           # nunca sale más agua de la que hay
        Hp[:] = 0.0
        for (dr, dc, w), d in zip(VECINOS, self._buf):
            d *= escala
            Hp[1 + dr:1 + dr + n, 1 + dc:1 + dc + n] += d
        saliente = tot * escala
        interior = Hp[1:-1, 1:-1]
        lados = (float(Hp[0, :].sum()), float(Hp[-1, :].sum()), float(Hp[1:-1, 0].sum()), float(Hp[1:-1, -1].sum()))
        self.salida_cel += sum(lados)
        self.salida_lado += lados
        self._aporte_paso += interior
        self._envio_paso += saliente
        self.recibido += interior
        self.enviado += saliente
        self.H = H - saliente + interior
        return float(saliente.max())

    def _nivelar_charcos(self):
        """Los charcos (celdas conectadas con más de POZA_MM) quedan con la superficie del agua HORIZONTAL: el volumen de cada
        charco se reparte con un solo nivel sobre su terreno. Solo redistribuye dentro del charco: conserva la masa."""
        H = self.H
        mascara = H > POZA_MM
        if mascara.sum() < 2:
            return
        lab, k = etiquetar(mascara)
        if k == 0:
            return
        idx = np.flatnonzero(mascara)
        etq = lab.ravel()[idx]
        orden = np.argsort(etq, kind="stable")
        idx, etq = idx[orden], etq[orden]
        cortes = np.flatnonzero(np.diff(etq)) + 1
        zf, Hf = self.z.ravel(), H.ravel()                       # vistas: H.ravel() comparte memoria con H
        rec, env = self.recibido.ravel(), self.enviado.ravel()
        for celdas in np.split(idx, cortes):
            if celdas.size < 2:
                continue
            zc = zf[celdas]
            vol = float(Hf[celdas].sum()) * 1e-3                 # m·celda
            nivel = nivel_de_charco(zc, vol)
            nuevo = np.maximum(nivel - zc, 0.0) * 1e3
            delta = nuevo - Hf[celdas]                           # + recibe agua del charco, − la entrega
            rec[celdas] += np.maximum(delta, 0.0)
            env[celdas] += np.maximum(-delta, 0.0)
            Hf[celdas] = nuevo

    def step(self):
        self._lluvia()
        self._infiltrar()
        t_next = (self.paso + 1) * self.esc.dt_s / 60.0
        nuevo = (self.H > UMBRAL_CHARCO_MM) & np.isinf(self.t_charco)
        if nuevo.any():
            self.t_charco[nuevo] = t_next
        self._aporte_paso[:] = 0.0
        self._envio_paso[:] = 0.0
        if self.H.max() > 1e-9:
            for _ in range(int(self.esc.pasos_ruteo)):
                if self._rutear() < 1e-6:                      # mm movidos en la pasada: ya casi nada se mueve
                    break
        if self.paso % NIVELAR_CADA == 0 and self.H.max() > POZA_MM:
            self._nivelar_charcos()
        self.aporte = 0.8 * self.aporte + 0.2 * self._aporte_paso
        self.envio = 0.8 * self.envio + 0.2 * self._envio_paso
        self.paso += 1
        if self.llueve:
            self.t_fin_lluvia = self.tiempo_min
            self.t_mitad_min = np.nan
        sup = float(self.H.sum())
        if sup > self.pico_sup:
            self.pico_sup, self.t_pico_min = sup, self.tiempo_min
        elif np.isnan(self.t_mitad_min) and not self.llueve and self.pico_sup > 0 and sup <= 0.5 * self.pico_sup:
            self.t_mitad_min = self.tiempo_min - max(self.t_fin_lluvia, self.t_pico_min)

    def avanzar(self, pasos: int):
        for _ in range(int(pasos)):
            self.step()

    def avanzar_hasta(self, minutos: float):
        self.avanzar(max(int(round(minutos * 60.0 / self.esc.dt_s)) - self.paso, 0))

    # ---- medidas -----------------------------------------------------------
    def error_balance(self) -> float:
        """Debe ser ~0: lluvia − (agua superficial + infiltración acumulada + salida por bordes)  [mm·celda]."""
        return float(self.lluvia_cel - self.H.sum() - self.F.sum() - self.salida_cel)

    def metricas(self) -> dict:
        m3 = self.mapa.dx_m ** 2 / 1000.0                      # mm·celda -> m³
        n2 = self.n * self.n
        return {
            "tiempo_min": self.tiempo_min,
            "lluvia_mm": self.lluvia_cel / n2,
            "vol_lluvia_m3": self.lluvia_cel * m3,
            "vol_infiltrado_m3": float(self.F.sum()) * m3,
            "vol_almacenado_suelo_m3": float((self.F - self.D).sum()) * m3,
            "vol_fuga_profunda_m3": float(self.D.sum()) * m3,
            "vol_superficial_m3": float(self.H.sum()) * m3,
            "vol_salida_m3": self.salida_cel * m3,
            "prof_max_mm": float(self.H.max()),
            "prof_media_mm": float(self.H.mean()),
            "area_encharcada_pct": 100.0 * float((self.H > UMBRAL_AREA_MM).mean()),
            "primer_charco_min": float(self.t_charco.min()),
            "pico_superficial_mm": self.pico_sup / n2,
            "t_mitad_min": self.t_mitad_min,
            "error_balance_mm_celda": self.error_balance(),
        }


def tp_formula_min(psi_mm: float, ks_mm_h: float, theta_e: float, i_mm_h: float, s0: float = 0.30) -> float:
    """t_p de Green-Ampt (min), la fórmula del documento y del xlsx. Infinito si i <= Ks."""
    if i_mm_h <= ks_mm_h:
        return float("inf")
    return psi_mm * (1 - s0) * theta_e * ks_mm_h / (i_mm_h - ks_mm_h) / i_mm_h * 60.0
