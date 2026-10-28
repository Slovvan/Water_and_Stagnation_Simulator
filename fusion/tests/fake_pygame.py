"""pygame simulado: solo valida que la interfaz llama a la API con argumentos de forma correcta y que el flujo no falla.
No sustituye una prueba con pygame real."""
from types import SimpleNamespace as NS
import numpy as np

QUIT, KEYDOWN, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION = 256, 768, 1025, 1026, 1024
K_TAB, K_SPACE, K_r, K_d, K_l, K_f, K_s, K_F11, K_ESCAPE, K_RIGHT, K_LEFT, K_DOWN, K_UP = 9, 32, 114, 100, 108, 102, 115, 1073741892, 27, 1, 2, 3, 4
RESIZABLE, FULLSCREEN, VIDEORESIZE = 16, -2147483648, 32769
SRCALPHA = 65536


class Surface:
    def __init__(self, size, flags=0):
        assert len(size) == 2 and all(int(v) > 0 for v in size), size
        self.size = (int(size[0]), int(size[1]))
        self.blits = 0

    def fill(self, color):
        assert len(color) in (3, 4)

    def blit(self, src, pos):
        assert isinstance(src, Surface) and len(pos) == 2
        self.blits += 1

    def get_size(self): return self.size
    def get_width(self): return self.size[0]
    def get_height(self): return self.size[1]


class Font:
    def __init__(self, nombre, tam, negrita=False): assert tam > 0
    def size(self, texto): return (max(len(texto), 1) * 7, 16)
    def render(self, texto, aa, color):
        assert isinstance(texto, str) and len(color) == 3
        return Surface((max(len(texto), 1) * 7, 16))


class Clock:
    def tick(self, fps): return 16


class _Draw:
    calls = {"rect": 0, "line": 0, "circle": 0, "lines": 0}

    def lines(self, surf, color, cerrado, puntos, width=1):
        assert isinstance(surf, Surface) and len(puntos) >= 2 and all(len(q) == 2 for q in puntos)
        self.calls["lines"] += 1

    def rect(self, surf, color, rect, width=0, border_radius=0):
        assert isinstance(surf, Surface) and len(color) in (3, 4) and len(rect) == 4
        self.calls["rect"] += 1

    def line(self, surf, color, a, b, width=1):
        assert isinstance(surf, Surface) and len(a) == 2 and len(b) == 2
        self.calls["line"] += 1

    def circle(self, surf, color, centro, radio, width=0):
        assert isinstance(surf, Surface) and len(centro) == 2
        self.calls["circle"] += 1


def make_surface(arr):
    assert arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8 and arr.flags["C_CONTIGUOUS"]
    return Surface((arr.shape[0], arr.shape[1]))


def scale(surf, size):
    assert isinstance(surf, Surface)
    return Surface(size)


def smoothscale(surf, size):
    assert isinstance(surf, Surface) and len(size) == 2 and size[0] > 0 and size[1] > 0
    return Surface(size)


def crear(guion, escritorio=(1920, 1080)):
    """guion: lista de listas de eventos; cada llamada a event.get() entrega la siguiente lista."""
    estado = {"i": 0, "pantalla": None, "flips": 0, "cerrado": False, "modos": [], "iconificada": 0}

    def get():
        i = estado["i"]
        estado["i"] += 1
        return guion[i] if i < len(guion) else []

    def set_mode(tam, flags=0):
        estado["modos"].append((tuple(tam), flags))
        estado["pantalla"] = Surface(tam)
        return estado["pantalla"]

    def iconify():
        estado["iconificada"] += 1
        return True

    def flip(): estado["flips"] += 1
    def quit_(): estado["cerrado"] = True

    pg = NS(QUIT=QUIT, KEYDOWN=KEYDOWN, MOUSEBUTTONDOWN=MOUSEBUTTONDOWN, MOUSEBUTTONUP=MOUSEBUTTONUP,
            MOUSEMOTION=MOUSEMOTION, K_TAB=K_TAB, K_SPACE=K_SPACE, K_r=K_r, K_d=K_d, K_l=K_l, K_f=K_f, K_s=K_s, K_F11=K_F11, RESIZABLE=RESIZABLE, FULLSCREEN=FULLSCREEN, VIDEORESIZE=VIDEORESIZE, K_ESCAPE=K_ESCAPE, K_RIGHT=K_RIGHT,
            K_LEFT=K_LEFT, K_DOWN=K_DOWN, K_UP=K_UP, SRCALPHA=SRCALPHA, Surface=Surface,
            init=lambda: None, quit=quit_,
            display=NS(set_mode=set_mode, set_caption=lambda t: None, flip=flip, iconify=iconify, get_surface=lambda: estado["pantalla"],
                       get_desktop_sizes=lambda: [escritorio], Info=lambda: NS(current_w=escritorio[0], current_h=escritorio[1])),
            font=NS(Font=Font, SysFont=lambda nombres, tam, negrita=False: Font(None, tam, negrita)), time=NS(Clock=Clock), draw=_Draw(),
            surfarray=NS(make_surface=make_surface), transform=NS(scale=scale, smoothscale=smoothscale), event=NS(get=get))
    return pg, estado


def ev(tipo, **kw):
    return NS(type=tipo, **kw)
