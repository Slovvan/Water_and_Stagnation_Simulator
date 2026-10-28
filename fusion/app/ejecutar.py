"""Punto de entrada: crea la aplicación con pygame real."""


def main():
    import pygame
    from app.ui import Aplicacion
    Aplicacion(pygame).ejecutar()


if __name__ == "__main__":
    main()
