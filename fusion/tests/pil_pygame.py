"""pygame de DESARROLLO basado en PIL: sirve para sacar capturas del lienzo de la interfaz sin pygame real.
No es parte de la aplicación; solo implementa lo que usa app/ui.py."""
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tests import fake_pygame as fp

_FUENTES = [f for pat in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
            for f in glob.glob(pat)]
_NEGRITAS = glob.glob("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


class Surface:
    def __init__(self, size, flags=0):
        self.size = (int(size[0]), int(size[1]))
        self.img = Image.new("RGBA", self.size, (0, 0, 0, 0 if flags & fp.SRCALPHA else 255))

    def fill(self, color):
        self.img.paste(tuple(color) + ((255,) if len(color) == 3 else ()), (0, 0, *self.size))

    def blit(self, src, pos):
        pos = (int(pos[0]), int(pos[1]))
        self.img.alpha_composite(src.img, dest=(max(pos[0], 0), max(pos[1], 0)), source=(max(-pos[0], 0), max(-pos[1], 0))) \
            if pos[0] < self.size[0] and pos[1] < self.size[1] else None

    def get_size(self): return self.size
    def get_width(self): return self.size[0]
    def get_height(self): return self.size[1]


class Font:
    def __init__(self, tam, negrita=False):
        ruta = (_NEGRITAS if negrita and _NEGRITAS else _FUENTES)[0]
        self.f = ImageFont.truetype(ruta, max(int(tam * 0.92), 8))

    def size(self, texto):
        b = self.f.getbbox(texto or " ")
        return (b[2], max(b[3], 14))

    def render(self, texto, aa, color):
        w, h = self.size(texto)
        s = Surface((w + 2, h + 4), fp.SRCALPHA)
        ImageDraw.Draw(s.img).text((0, 0), texto, font=self.f, fill=tuple(color) + (255,))
        return s


class _Draw:
    def rect(self, surf, color, rect, width=0, border_radius=0):
        d = ImageDraw.Draw(surf.img, "RGBA")
        x, y, w, h = rect
        box = (x, y, x + w - 1, y + h - 1)
        kw = dict(fill=tuple(color) if width == 0 else None, outline=tuple(color) if width else None, width=max(width, 1))
        (d.rounded_rectangle(box, radius=border_radius, **kw) if border_radius else d.rectangle(box, **kw))

    def line(self, surf, color, a, b, width=1):
        ImageDraw.Draw(surf.img, "RGBA").line([tuple(a), tuple(b)], fill=tuple(color), width=int(width))

    def lines(self, surf, color, cerrado, pts, width=1):
        ImageDraw.Draw(surf.img, "RGBA").line([tuple(q) for q in pts], fill=tuple(color), width=int(width))

    def circle(self, surf, color, c, r, width=0):
        ImageDraw.Draw(surf.img, "RGBA").ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=tuple(color) if width == 0 else None,
                                                  outline=tuple(color) if width else None)


def crear(guion, escritorio=(1920, 1080)):
    pg, st = fp.crear(guion, escritorio)

    def set_mode(tam, flags=0):
        st["modos"].append((tuple(tam), flags))
        st["pantalla"] = Surface(tam)
        return st["pantalla"]

    pg.Surface = Surface
    pg.display.set_mode = set_mode
    pg.display.get_surface = lambda: st["pantalla"]
    pg.font = type("F", (), {"Font": staticmethod(lambda n, t, neg=False: Font(t, neg)),
                             "SysFont": staticmethod(lambda nombres, t, neg=False: Font(t, neg))})
    pg.draw = _Draw()
    pg.surfarray = type("S", (), {"make_surface": staticmethod(lambda a: _desde(a))})
    pg.transform = type("T", (), {"scale": staticmethod(lambda s, tam: _esc(s, tam, Image.NEAREST)),
                                  "smoothscale": staticmethod(lambda s, tam: _esc(s, tam, Image.BICUBIC))})
    return pg, st


def _desde(a):
    s = Surface((a.shape[0], a.shape[1]))
    s.img = Image.fromarray(np.ascontiguousarray(a.swapaxes(0, 1))).convert("RGBA")
    return s


def _esc(s, tam, modo):
    o = Surface(tam)
    o.img = s.img.resize((int(tam[0]), int(tam[1])), modo)
    return o
