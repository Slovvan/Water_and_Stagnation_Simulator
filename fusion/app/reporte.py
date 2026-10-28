"""Reporte final de la simulación: un HTML autocontenido (tablas, gráficas SVG e imágenes incrustadas) y un CSV con la
serie temporal. Solo usa la biblioteca estándar y NumPy."""
from __future__ import annotations

import base64
import datetime as dt
import html
import struct
import zlib
from pathlib import Path

import numpy as np

from app.corte_sync import CorteSincronizado
from corte.medio import _perfil
from water.soil_data import AGS, CONFIANZA_ASIGNACION, CUSTOM_MATERIALS, RAWLS_1983
from water.soil_terrain import get_material

COLORES = ["#2a70c8", "#2e9e6b", "#d97b29", "#7a4fb8", "#c0392b", "#8a6d3b", "#15a0a5", "#d4a017", "#5d6d7e"]


# ------------------------------------------------------------------ PNG sin dependencias
def png_bytes(rgb: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def trozo(tipo, datos):
        return struct.pack(">I", len(datos)) + tipo + datos + struct.pack(">I", zlib.crc32(tipo + datos) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + trozo(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + trozo(b"IDAT", zlib.compress(raw, 6)) + trozo(b"IEND", b""))


def _img_html(rgb, ancho, alt):
    b64 = base64.b64encode(png_bytes(rgb)).decode()
    return f'<img alt="{html.escape(alt)}" width="{ancho}" src="data:image/png;base64,{b64}">'


# ------------------------------------------------------------------ gráficas SVG
def _ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1.0
    paso = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(paso))
    paso = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= paso), default=paso)
    ini = np.floor(lo / paso) * paso
    out, k = [ini], 1
    while out[-1] < hi - 1e-9:                         # hasta cubrir `hi` (la última marca es >= hi)
        out.append(ini + k * paso)
        k += 1
    return [float(v) for v in out]


def svg_lineas(series, titulo, xlab, ylab, w=640, h=290, escalon=False):
    """series: lista de (nombre, xs, ys, color)."""
    ml, mr, mb = 56, 14, 44
    filas_leyenda, lxs, lx = 1, [], 56 + 8
    for nombre, *_ in series:                                         # la leyenda salta de renglón si no cabe
        if lx + 22 + 6.2 * len(nombre) > w - 14:
            filas_leyenda, lx = filas_leyenda + 1, 56 + 8
        lxs.append((lx, filas_leyenda))
        lx += 22 + 6.2 * len(nombre)
    mt = 34 + 15 * filas_leyenda
    xs_all = np.concatenate([np.asarray(s[1], float) for s in series if len(s[1])] or [np.array([0.0, 1.0])])
    ys_all = np.concatenate([np.asarray(s[2], float) for s in series if len(s[2])] or [np.array([0.0, 1.0])])
    x0, x1 = float(xs_all.min()), float(xs_all.max())
    y1 = float(ys_all.max()) if ys_all.size else 1.0
    y1 = y1 * 1.08 if y1 > 0 else 1.0
    tx, ty = _ticks(x0, x1 if x1 > x0 else x0 + 1), _ticks(0.0, y1)
    X = lambda v: ml + (v - x0) / max(x1 - x0, 1e-9) * (w - ml - mr)
    Y = lambda v: h - mb - v / max(ty[-1], 1e-9) * (h - mt - mb)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="sans-serif" font-size="11">',
         f'<text x="{w / 2}" y="16" text-anchor="middle" font-size="13" font-weight="bold">{html.escape(titulo)}</text>']
    for v in ty:
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e3e7ea"/><text x="{ml - 5}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#555">{v:g}</text>')
    for v in tx:
        o.append(f'<text x="{X(v):.1f}" y="{h - mb + 15}" text-anchor="middle" fill="#555">{v:g}</text>')
    o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}" stroke="#888"/><line x1="{ml}" x2="{ml}" y1="{mt}" y2="{h - mb}" stroke="#888"/>')
    o.append(f'<text x="{(ml + w - mr) / 2}" y="{h - 6}" text-anchor="middle" fill="#333">{html.escape(xlab)}</text>')
    o.append(f'<text transform="translate(13 {(mt + h - mb) / 2}) rotate(-90)" text-anchor="middle" fill="#333">{html.escape(ylab)}</text>')
    for (nombre, xs, ys, col), (lx, fila_l) in zip(series, lxs):
        if len(xs) == 0:
            continue
        pts = []
        for k, (a, b) in enumerate(zip(xs, ys)):
            if escalon and k:
                pts.append(f"{X(a):.1f},{Y(ys[k - 1]):.1f}")
            pts.append(f"{X(a):.1f},{Y(b):.1f}")
        o.append(f'<polyline fill="none" stroke="{col}" stroke-width="1.8" points="{" ".join(pts)}"/>')
        o.append(f'<rect x="{lx}" y="{16 + 15 * fila_l}" width="10" height="3" fill="{col}"/><text x="{lx + 13}" y="{21 + 15 * fila_l}" fill="#333">{html.escape(nombre)}</text>')
    o.append("</svg>")
    return "".join(o)


def svg_barras(etiquetas, valores, titulo, ylab, w=640, h=290, colores=None, fmt="{:.1f}"):
    ml, mr, mt, mb = 56, 14, 36, 74
    vmax = max(max(valores, default=0.0), 1e-9) * 1.10
    ty = _ticks(0.0, vmax)
    Y = lambda v: h - mb - v / ty[-1] * (h - mt - mb)
    n = len(valores)
    ancho = (w - ml - mr) / max(n, 1)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="sans-serif" font-size="11">',
         f'<text x="{w / 2}" y="16" text-anchor="middle" font-size="13" font-weight="bold">{html.escape(titulo)}</text>']
    for v in ty:
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e3e7ea"/><text x="{ml - 5}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="#555">{v:g}</text>')
    for k, (e, v) in enumerate(zip(etiquetas, valores)):
        x = ml + k * ancho + ancho * 0.15
        col = (colores or COLORES)[k % len(colores or COLORES)]
        o.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{ancho * 0.7:.1f}" height="{h - mb - Y(v):.1f}" fill="{col}"/>')
        o.append(f'<text x="{x + ancho * 0.35:.1f}" y="{Y(v) - 4:.1f}" text-anchor="middle" fill="#333">{fmt.format(v)}</text>')
        o.append(f'<text transform="translate({x + ancho * 0.35:.1f} {h - mb + 10}) rotate(35)" fill="#333">{html.escape(e)}</text>')
    o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}" stroke="#888"/>')
    o.append(f'<text transform="translate(13 {(mt + h - mb) / 2}) rotate(-90)" text-anchor="middle" fill="#333">{html.escape(ylab)}</text></svg>')
    return "".join(o)


# ------------------------------------------------------------------ datos
def tabla_grupos(ctrl) -> list[dict]:
    mo, m = ctrl.motor, ctrl.mapa
    m3 = m.dx_m ** 2 / 1000.0
    filas = []
    for k, cod in enumerate(m.codes):
        sel = m.soil_id == k
        n = int(sel.sum())
        if n == 0:
            continue
        P, F = float(mo.P[sel].mean()), float(mo.F[sel].mean())
        ch = np.isfinite(mo.t_charco[sel])
        mats, _ = _perfil(cod, m, AGS)
        filas.append(dict(
            codigo=cod, nombre=m.nombres.get(cod, cod), superficial=mats[0], confianza=CONFIANZA_ASIGNACION.get(cod, "N/A"),
            celdas=n, area_pct=100.0 * n / m.n ** 2, area_m2=n * m.dx_m ** 2,
            ks_med=float(np.median(m.ks_mm_h[sel])), ks_min=float(m.ks_mm_h[sel].min()), ks_max=float(m.ks_mm_h[sel].max()),
            prof_med=float(m.profundidad_mm[sel].mean()), lluvia_mm=P, infil_mm=F, infil_pct=100.0 * F / P if P > 0 else 0.0,
            vol_infil=float(mo.F[sel].sum()) * m3, vol_lluvia=float(mo.P[sel].sum()) * m3,
            lleno_pct=100.0 * float(((mo.F - mo.D) >= mo.cap - 1e-6)[sel].mean()), charco_pct=100.0 * float(ch.mean()),
            t_charco=float(np.median(mo.t_charco[sel][ch])) if ch.any() else float("nan"),
            recibido=float(mo.recibido[sel].sum()) * m3, enviado=float(mo.enviado[sel].sum()) * m3,
            sup=float(mo.H[sel].sum()) * m3, fuga=float(mo.D[sel].sum()) * m3))
    return sorted(filas, key=lambda f: -f["area_pct"])


def _fmt(v, d=1, vacio="—"):
    return vacio if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{d}f}".replace(",", " ")


def _tabla(cab, filas, clases=None):
    h = "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cab) + "</tr></thead><tbody>"
    for f in filas:
        h += "<tr>" + "".join(f"<td>{c}</td>" for c in f) + "</tr>"
    return h + "</tbody></table>"


def _origen(mat):
    if mat in RAWLS_1983:
        return "Rawls et al. (1983), tabla copiada de la wiki de TUFLOW"
    if mat in CUSTOM_MATERIALS:
        return "supuesto del simulador (sin equivalente en Rawls)"
    return "—"


# ------------------------------------------------------------------ reporte
def generar_reporte(ctrl, carpeta: str | Path = "reportes", nombre: str | None = None) -> dict:
    mo, m, esc = ctrl.motor, ctrl.mapa, ctrl.esc
    ahora = dt.datetime.now()
    base = nombre or f"reporte_{ahora:%Y%m%d_%H%M%S}"
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_html, ruta_csv = carpeta / f"{base}.html", carpeta / f"{base}_serie.csv"
    S = ctrl.serie
    t = [s["t"] for s in S]
    n2 = m.n ** 2
    met = mo.metricas()
    m3 = m.dx_m ** 2 / 1000.0
    lluvia_m3 = met["vol_lluvia_m3"]
    pct = lambda v: 100.0 * v / lluvia_m3 if lluvia_m3 > 0 else 0.0
    grupos = tabla_grupos(ctrl)
    lados = [float(v) * m3 for v in mo.salida_lado]

    # --- CSV
    cab = ["t_min", "lluvia_mm", "infiltrado_mm", "fuga_profunda_mm", "superficie_mm", "salida_mm", "intensidad_mm_h",
           "prof_max_charco_mm", "area_encharcada_pct", "perfiles_llenos_pct"] + [f"F_{c}_mm" for c in m.codes]
    with open(ruta_csv, "w", encoding="utf-8") as fh:
        fh.write(",".join(cab) + "\n")
        for s in S:
            fila = [s["t"], s["lluvia"], s["infiltrado"], s["fuga"], s["superficie"], s["salida"], s["intensidad"],
                    s["prof_max"], s["area_pct"], s["lleno_pct"]] + list(s["F_grupo"])
            fh.write(",".join(f"{v:.6g}" for v in fila) + "\n")

    # --- gráficas
    g = []
    if S:
        g.append(svg_lineas([("Lluvia", t, [s["lluvia"] for s in S], "#2a70c8"), ("Infiltrado", t, [s["infiltrado"] for s in S], "#2e9e6b"),
                             *([("Salió del mapa", t, [s["salida"] for s in S], "#d97b29")] if esc.bordes_abiertos else []),
                             ("En superficie", t, [s["superficie"] for s in S], "#7a4fb8")],
                            "Balance de agua acumulado (lámina sobre todo el mapa)", "tiempo (min)", "mm"))
        g.append(svg_lineas([("Intensidad", t, [s["intensidad"] for s in S], "#2a70c8")], "Lluvia aplicada", "tiempo (min)", "mm/h", escalon=True))
        g.append(svg_lineas([("Área encharcada", t, [s["area_pct"] for s in S], "#2a70c8"), ("Perfiles llenos", t, [s["lleno_pct"] for s in S], "#c0392b")],
                            "Área encharcada (>1 mm) y perfiles llenos", "tiempo (min)", "% del mapa"))
        g.append(svg_lineas([("Profundidad máxima", t, [s["prof_max"] for s in S], "#1b5aa8")], "Profundidad máxima del charco", "tiempo (min)", "mm"))
        top = [k for k, c in enumerate(m.codes) if (m.soil_id == k).any()]
        top = sorted(top, key=lambda k: -(m.soil_id == k).sum())[:9]
        g.append(svg_lineas([(m.nombres.get(m.codes[k], m.codes[k]), t, [s["F_grupo"][k] for s in S], COLORES[i % 9]) for i, k in enumerate(top)],
                            "Infiltración acumulada media por grupo de suelo", "tiempo (min)", "mm", h=320))
    g_barras = svg_barras([f["nombre"] for f in grupos], [f["infil_mm"] for f in grupos], "Infiltración media por grupo de suelo (final)", "mm") if grupos else ""
    g_pct = svg_barras([f["nombre"] for f in grupos], [f["infil_pct"] for f in grupos], "Porcentaje de la lluvia que se infiltró, por grupo", "% de la lluvia", fmt="{:.0f}") if grupos else ""
    mov = [("Recibido de otras celdas", sum(f["recibido"] for f in grupos), "#7a4fb8"), ("Enviado a otras celdas", sum(f["enviado"] for f in grupos), "#d97b29"),
           *([("Salió del mapa", met["vol_salida_m3"], "#c0392b")] if esc.bordes_abiertos else []),
           ("Fuga profunda", met["vol_fuga_profunda_m3"], "#2e9e6b"), ("Quedó en superficie", met["vol_superficial_m3"], "#2a70c8")]
    g_flujo = svg_barras([a for a, _, _ in mov], [b for _, b, _ in mov], "Agua que se movió (m³)", "m³", colores=[c for _, _, c in mov], fmt="{:.0f}")

    # --- imágenes finales
    for _ in range(40):                                              # que el agua suavizada alcance el estado final
        ctrl.vista_aerea.suavizar(mo.H)
    img_mapa = np.repeat(np.repeat(ctrl.vista_aerea.imagen(mo.H, suavizar=False), 5, 0), 5, 1)
    cs = CorteSincronizado(m, esc, ctrl.eje, ctrl.indice)
    corte, _ = cs.imagen(mo)
    img_corte = np.repeat(np.repeat(corte, 2, 0), 5, 1)

    # --- tablas
    filas_g = [[f"<b>{html.escape(f['nombre'])}</b> ({f['codigo']})", html.escape(f["superficial"]), f["confianza"], _fmt(f["area_pct"]), _fmt(f["area_m2"], 0),
                f"{f['ks_med']:.1f} ({f['ks_min']:.1f}–{f['ks_max']:.1f})", _fmt(f["prof_med"], 0), _fmt(f["infil_mm"]), _fmt(f["infil_pct"], 0),
                _fmt(f["vol_infil"]), _fmt(f["lleno_pct"], 0), _fmt(f["charco_pct"], 0), _fmt(f["t_charco"])] for f in grupos]
    tabla1 = _tabla(["Grupo", "Horizonte superficial", "Confianza", "Área %", "Área m²", "Ks mm/h (mediana, mín–máx)", "Perfil mm", "Infiltrado mm", "% de la lluvia",
                     "Infiltrado m³", "Perfiles llenos %", "Celdas con charco %", "1.er charco (min)"], filas_g)
    filas_f = [[html.escape(f["nombre"]), _fmt(f["vol_lluvia"]), _fmt(f["recibido"]), _fmt(f["enviado"]), _fmt(f["enviado"] - f["recibido"]),
                _fmt(f["vol_infil"]), _fmt(f["sup"]), _fmt(f["fuga"], 2)] for f in grupos]
    tabla2 = _tabla(["Grupo", "Lluvia m³", "Recibido m³", "Enviado m³", "Neto exportado m³", "Infiltrado m³", "En superficie m³", "Fuga profunda m³"], filas_f)
    mats = {}
    for k, cod in enumerate(m.codes):
        if (m.soil_id == k).any():
            for mt in _perfil(cod, m, AGS)[0]:
                mats.setdefault(mt, set()).add(cod)
    filas_m = []
    for mt, cods in mats.items():
        psi, ks, th = get_material(mt, AGS)
        filas_m.append([html.escape(mt), f"{ks:g}", f"{psi:g}", f"{th:g}", html.escape(_origen(mt)), html.escape(", ".join(sorted(cods)))])
    tabla3 = _tabla(["Material", "Ks mm/h", "ψ mm", "θe", "Origen del valor", "Presente en"], filas_m)
    filas_b = [["Arriba (norte)", _fmt(lados[0])], ["Abajo (sur)", _fmt(lados[1])], ["Izquierda (oeste)", _fmt(lados[2])], ["Derecha (este)", _fmt(lados[3])], ["Total", _fmt(sum(lados))]]
    tabla4 = (_tabla(["Borde del mapa", "Agua que salió (m³)"], filas_b) if esc.bordes_abiertos
              else '<p><b>Bordes cerrados:</b> el agua no sale del mapa; lo que no se infiltra se acumula en los charcos.</p>')

    tch = met["primer_charco_min"]
    infil_pct = pct(met["vol_infiltrado_m3"])
    resumen = _tabla(["Concepto", "Valor"], [
        ["Tiempo simulado", f"{met['tiempo_min']:.0f} min ({met['tiempo_min'] / 60:.2f} h)"],
        ["Lluvia total", f"{met['lluvia_mm']:.1f} mm = {lluvia_m3:,.0f} m³".replace(",", " ")],
        ["Infiltrado en el suelo", f"{met['vol_infiltrado_m3']:,.0f} m³ ({infil_pct:.0f}% de la lluvia)".replace(",", " ")],
        ["  de ello, fuga por el fondo (sustrato)", f"{met['vol_fuga_profunda_m3']:.2f} m³"],
        ["Quedó en superficie al final", f"{met['vol_superficial_m3']:.1f} m³ ({pct(met['vol_superficial_m3']):.1f}%)"],
        ["Salió por los bordes del mapa", (f"{met['vol_salida_m3']:,.0f} m³ ({pct(met['vol_salida_m3']):.0f}%)".replace(",", " ") if esc.bordes_abiertos else "0 (bordes cerrados)")],
        ["Error del balance de masa", f"{met['error_balance_mm_celda']:.1e} mm·celda"],
        ["Profundidad máxima de charco (al final / pico)", f"{met['prof_max_mm']:.1f} mm / {max([s['prof_max'] for s in S], default=0):.1f} mm"],
        ["Área encharcada (>1 mm) al final / máxima", f"{met['area_encharcada_pct']:.1f}% / {max([s['area_pct'] for s in S], default=0):.1f}%"],
        ["Primer charco en el mapa", "no se produjo" if not np.isfinite(tch) else f"{tch:.1f} min"],
        ["Tiempo para que el agua superficial baje a la mitad de su pico", "—" if np.isnan(met["t_mitad_min"]) else f"{met['t_mitad_min']:.0f} min tras dejar de llover"]])
    cfg = _tabla(["Parámetro", "Valor"], [
        ["Semilla / mapa", f"{esc.seed} / {m.n} × {m.n} celdas de {m.dx_m:g} m"],
        ["Relieve", f"escala {esc.escala:g}, {esc.octavas} octavas, persistencia {esc.persistencia:g}, amplitud {esc.amplitud_m:g} m"],
        ["Saturación inicial S0", f"{esc.s0:g}"], ["Variabilidad de Ks (σ de ln Ks)", f"{esc.ks_sigma_ln:g}"],
        ["Bordes del mapa", "abiertos (el terreno continúa; sale agua solo si baja hacia afuera)" if esc.bordes_abiertos else "cerrados (el agua no sale)"],
        ["Fondo del perfil", "roca fracturada: fuga de %.1f mm/h cuando el perfil se llena" % m.k_fondo_mm_h if m.k_fondo_mm_h > 0 else "roca sellada: sin fuga"],
        ["Paso de tiempo", f"{esc.dt_s:g} s · {esc.pasos_ruteo} pasadas de reparto superficial por paso"],
        ["Ruido de la lluvia entre celdas", f"{esc.ruido_lluvia:g}"],
        ["Lluvia", "continua" if not np.isfinite(mo.duracion_min) else f"hasta los {mo.duracion_min:.0f} min"]])
    limites = [
        "El reparto del agua superficial es rápido y por niveles: <b>no</b> es una velocidad real de escorrentía ni da tiempos de traslado.",
        "La infiltración usa Green-Ampt con el horizonte superficial y limita por almacenamiento del perfil; no ve capas profundas con otra Ks "
        "(en Luvisoles y tormentas de más de ~3 h sobreestima la infiltración).",
        "No hay evaporación, transpiración, intercepción, macroporos, sellado superficial ni nivel freático.",
        "Los charcos se nivelan en horizontal dentro del mapa; el relieve es procedural, no un modelo digital de elevación real.",
        "La equivalencia grupo WRB → textura USDA es un supuesto (solo 45.10% de la superficie estatal con confianza media; "
        "30.23% baja; 18.48% Leptosoles sin equivalente).",
        "Los valores de Rawls provienen de una reproducción (TUFLOW), no del artículo original; Leptosol y roca fracturada son supuestos ilustrativos.",
        "No hay validación física con observaciones independientes: los resultados son comparativos, no predicciones de un sitio.",
        "El corte lateral dibuja lo que calcula el mapa; el detalle de 2 m usa otro motor (autómata) que se inicializa desde el mapa."]
    verif = ("Verificación del motor: la infiltración acumulada coincide con la solución implícita de Green-Ampt con una desviación máxima "
             "de 0.01% en las 11 texturas de Rawls y el Leptosol (<code>python -m tests.verificar_materiales</code>).")

    css = ("body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f3f5f7;color:#1c2a33}main{max-width:1180px;margin:auto;padding:24px}"
           "h1{margin:0 0 4px}h2{margin:30px 0 10px;border-bottom:2px solid #2a70c8;padding-bottom:4px}"
           "table{border-collapse:collapse;background:#fff;font-size:13px;margin:8px 0}th,td{border:1px solid #d5dbe0;padding:5px 8px;text-align:right}"
           "th{background:#e8eef4}td:first-child,th:first-child{text-align:left}.g{display:flex;flex-wrap:wrap;gap:14px}"
           ".g>*{background:#fff;border:1px solid #d5dbe0;padding:6px}.nota{background:#fff8e1;border-left:4px solid #e0a800;padding:8px 12px;margin:10px 0}"
           "small{color:#5a6a75}")
    cuerpo = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Reporte de la simulación</title><style>{css}</style></head><body><main>
<h1>Reporte de la simulación de encharcamiento</h1><small>Generado el {ahora:%d/%m/%Y %H:%M} · {met['tiempo_min']:.0f} min simulados</small>
<h2>1. Resumen del agua</h2>{resumen}
<h2>2. Condiciones de la simulación</h2>{cfg}
<h2>3. Cómo evolucionó</h2><div class="g">{''.join(g) if g else '<p>Aún no hay datos en el tiempo (la simulación no avanzó).</p>'}</div>
<h2>4. Materiales y absorción de cada uno</h2><div class="nota">Cada grupo de suelo infiltra con el horizonte superficial indicado. «Perfiles llenos»: el suelo ya no admite más agua
(sale solo la fuga del sustrato); «Celdas con charco»: alguna vez tuvieron más de 0.02 mm de agua encima.</div>{tabla1}
<div class="g">{g_barras}{g_pct}</div>
<h2>5. Agua que fluyó a través</h2><div class="nota">«Recibido» y «Enviado» suman lo que cada celda pasó a/desde sus vecinas durante todas las pasadas del reparto
(incluye ir y venir dentro de un charco); «Neto exportado» = enviado − recibido: positivo = el grupo entrega agua a otros.</div>{tabla2}
<div class="g">{g_flujo}<div>{tabla4}</div></div>
<h2>6. Estado final</h2><div class="g"><div><b>Mapa</b> (agua en azul)<br>{_img_html(img_mapa, 420, 'Mapa final')}</div>
<div><b>Corte lateral</b> de la línea {html.escape(ctrl.nombre_linea)}<br>{_img_html(img_corte, 560, 'Corte final')}</div></div>
<h2>7. Materiales del modelo y origen de sus valores</h2>{tabla3}
<h2>8. Límites del modelo</h2><ul>{''.join(f'<li>{x}</li>' for x in limites)}</ul><p>{verif}</p>
<p><small>Serie temporal completa: {html.escape(ruta_csv.name)}</small></p></main></body></html>"""
    ruta_html.write_text(cuerpo, encoding="utf-8")
    return {"html": ruta_html, "csv": ruta_csv}
