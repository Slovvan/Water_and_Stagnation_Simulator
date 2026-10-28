"""
surface_leveling.py
===================
Nivelación hidrostática del agua libre (charcos y escorrentía) sobre el perfil del terreno.

Las reglas del CA mueven el agua libre como máximo una celda por subpaso y la igualan por difusión,
así que un charco de 1 m tardaría horas simuladas en ponerse horizontal (en la realidad: segundos).
Esta regla reparte, en cada subpaso, el agua libre de cada cuenca en su estado de equilibrio:
  - el agua de una ladera baja hasta el fondo de su cuenca (descenso por el terreno),
  - cada charco queda horizontal,
  - si una cuenca se llena hasta su divisoria más baja, el exceso pasa a la cuenca vecina
    (o sale del dominio si la divisoria da a un borde abierto) y, si ambas se llenan, se unen.

El terreno no cambia, así que las cuencas, sus divisorias y sus áreas de aporte se calculan una vez.
Conserva la masa: lo que sale por los bordes se devuelve como escorrentía.
"""
import numpy as np

TOL = 1e-12


class SurfaceLeveling:
    def __init__(self, medium, open_edges=True, flow_rate=1.0):
        n = medium.grid_size
        self.n = n
        self.air = medium.kind == 0
        self.surface_row = np.asarray(medium.surface_row, int)
        z = (n - self.surface_row).astype(float)          # altura (celdas) del suelo de cada columna
        self.z = z
        self.open_edges = bool(open_edges)
        self.flow_rate = float(flow_rate)
        rows = np.arange(n)[:, None]
        self.k_above = self.surface_row[None, :] - 1 - rows  # 0 = celda de aire justo sobre el suelo

        edge = -np.inf if self.open_edges else np.inf
        runs = []                                           # tramos de altura constante: (inicio, fin)
        s = 0
        for c in range(1, n + 1):
            if c == n or z[c] != z[s]:
                runs.append((s, c - 1))
                s = c
        n_runs = len(runs)
        direction = np.zeros(n_runs, int)                   # -1 izquierda · +1 derecha · 0 mínimo
        for r, (a, b) in enumerate(runs):
            h = z[a]
            left = z[a - 1] if a > 0 else edge
            right = z[b + 1] if b < n - 1 else edge
            if left < h or right < h:
                direction[r] = -1 if left <= right else 1

        # hojas ordenadas de izquierda a derecha: [SALIDA_IZQ] + mínimos + [SALIDA_DER]
        minima = [r for r in range(n_runs) if direction[r] == 0]
        leaves = []                                         # (inicio, fin) del fondo; None = salida
        if self.open_edges:
            leaves.append((-1, -1))
        run_leaf = {}
        for r in minima:
            run_leaf[r] = len(leaves)
            leaves.append(runs[r])
        if self.open_edges:
            leaves.append((n, n))
        out_left, out_right = (0, len(leaves) - 1) if self.open_edges else (-1, -1)

        def target(r):
            seen = []
            while r not in run_leaf:
                seen.append(r)
                nxt = r + direction[r]
                if nxt < 0:
                    leaf = out_left
                    break
                if nxt >= n_runs:
                    leaf = out_right
                    break
                r = nxt
            else:
                leaf = run_leaf[r]
            for q in seen:
                run_leaf[q] = leaf
            return leaf

        leaf_of = np.empty(n, int)
        for r, (a, b) in enumerate(runs):
            leaf_of[a:b + 1] = target(r)
        self.leaf_of = leaf_of
        self.n_leaves = len(leaves)
        self.is_out = np.zeros(self.n_leaves, bool)
        if self.open_edges:
            self.is_out[[out_left, out_right]] = True

        # área de aporte de cada hoja (contigua en 1D) y divisorias entre hojas consecutivas
        self.catch = []
        for j in range(self.n_leaves):
            cols = np.nonzero(leaf_of == j)[0]
            self.catch.append((cols.min(), cols.max()) if cols.size else None)
        self.ridge = np.array([z[leaves[j][1] + 1: leaves[j + 1][0]].max()
                               for j in range(self.n_leaves - 1)])

    # ------------------------------------------------------------------ #
    def _cols(self, g):
        lo = next(self.catch[j][0] for j in range(g[0], g[1] + 1) if self.catch[j] is not None)
        hi = next(self.catch[j][1] for j in range(g[1], g[0] - 1, -1) if self.catch[j] is not None)
        return lo, hi

    def _capacity(self, g, level):
        lo, hi = self._cols(g)
        return float(np.maximum(level - self.z[lo:hi + 1], 0.0).sum())

    def _outlets(self, group):
        outlets = []
        if self.is_out[0] and group[0] == 0:
            outlets.append(0)
        right = self.n_leaves - 1
        if self.is_out[right] and group[1] == right:
            outlets.append(right)
        return outlets

    def _distribute(self, water):
        """water[j] = agua libre que drena a la hoja j. Devuelve (grupos finales, escorrentía)."""
        groups = [[j, j, float(water[j])] for j in range(self.n_leaves)]   # [hoja_ini, hoja_fin, agua]
        runoff = 0.0
        outlet_runoff = np.zeros(self.n_leaves, dtype=float)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(groups):
                g = groups[i]
                outlets = self._outlets(g)
                if outlets:
                    for outlet in outlets:
                        available = max(self.flow_rate - outlet_runoff[outlet], 0.0)
                        drained = min(g[2], available)
                        runoff += drained
                        outlet_runoff[outlet] += drained
                        g[2] -= drained
                    if g[2] <= TOL:
                        i += 1
                        continue
                rl = self.ridge[g[0] - 1] if i > 0 else np.inf
                rr = self.ridge[g[1]] if i < len(groups) - 1 else np.inf
                spill = min(rl, rr)
                if np.isfinite(spill):
                    cap = self._capacity(g, spill)
                    if g[2] > cap + TOL:
                        nb = i - 1 if rl <= rr else i + 1
                        groups[nb][2] += g[2] - cap
                        g[2] = cap
                        changed = True
                        other = groups[nb]
                        ridge = rl if nb < i else rr
                        if nb < i:
                            far = self.ridge[other[0] - 1] if nb > 0 else np.inf
                        else:
                            far = self.ridge[other[1]] if nb < len(groups) - 1 else np.inf
                        # se unen solo si la divisoria común es también la más baja de la vecina
                        if far >= ridge and other[2] >= self._capacity(other, ridge) - TOL:
                            a, b = (other, g) if nb < i else (g, other)
                            merged = [a[0], b[1], a[2] + b[2]]
                            k = min(i, nb)
                            groups[k:k + 2] = [merged]
                            i = max(k - 1, 0)
                            continue
                i += 1
        return groups, runoff

    @staticmethod
    def _level(z, water):
        zs = np.sort(z)
        csum = np.cumsum(zs)
        k = np.arange(1, zs.size + 1)
        vol_at = k * zs - csum                      # agua necesaria para subir el nivel hasta zs[k-1]
        m = int(np.searchsorted(vol_at, water, side="right"))   # columnas sumergidas
        return (water + csum[m - 1]) / m

    def apply(self, grid):
        air_water = np.where(self.air, grid, 0.0)
        per_column = air_water.sum(axis=0)
        if per_column.max() <= TOL:
            return grid, 0.0
        water = np.bincount(self.leaf_of, weights=per_column, minlength=self.n_leaves)
        groups, runoff = self._distribute(water)

        depth = np.zeros(self.n)
        for g in groups:
            if g[2] <= TOL:
                continue
            lo, hi = self._cols(g)
            zc = self.z[lo:hi + 1]
            depth[lo:hi + 1] = np.maximum(self._level(zc, g[2]) - zc, 0.0)
        # el agua de cada columna queda apilada sobre el suelo (celdas llenas + una parcial arriba)
        stacked = np.clip(depth[None, :] - self.k_above, 0.0, 1.0)
        updated = np.where(self.air, stacked, grid)
        return updated, runoff
