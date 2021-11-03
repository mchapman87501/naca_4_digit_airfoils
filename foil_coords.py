#!/usr/bin/env python3
from dataclasses import dataclass
import math
import re
import typing as tp


@dataclass(frozen=True)
class NACAFoil:
    id_str: str
    max_camber: float
    camber_pos: float
    thickness: float


def parse_id(airfoil_id: str) -> NACAFoil:
    pattern = r"^NACA\s*(\d)(\d)(\d{2})$"
    expr = re.compile(pattern, re.I)
    if m := expr.match(airfoil_id):
        max_camber_fract = int(m.group(1)) / 100.0
        camber_pos_fract = int(m.group(2)) / 10.0
        thickness_fract = int(m.group(3)) / 100.0
        return NACAFoil(
            id_str=airfoil_id,
            max_camber=max_camber_fract,
            camber_pos=camber_pos_fract,
            thickness=thickness_fract,
        )
    raise ValueError(f"{airfoil_id} is not a valid NACA identifier")


@dataclass(frozen=True)
class Coord:
    x: float
    y: float


def frange(v0: float, vf: float, dv: float) -> tp.Sequence[float]:
    # For internal use - never mind parameter validation.
    v = v0
    if vf > v0:
        while v < vf:
            yield v
            v += dv
    else:
        while v >= vf:
            yield v
            v += dv


class FoilMaker:
    def __init__(self, foil: NACAFoil) -> None:
        self._foil = foil

    def gen_coordinates(self) -> tp.Sequence[Coord]:
        num_steps = 10
        dx = 1.0 / num_steps
        # Generate upper surface from 0.0 to 1.0
        for x in frange(0.0, 1.0, dx):
            yield self.upper_surface(x)
        for x in frange(x, 0.0, -dx):
            yield self.lower_surface(x)

    def yc(self, x: float) -> float:
        foil = self._foil
        m = foil.max_camber
        p = foil.camber_pos
        p_sqr = p * p
        if 0.0 <= x < p:
            return (m / p_sqr) * (2.0 * p * x - x ** 2)
        if p <= x < 1.0:
            return (m / (1.0 - p) ** 2) * (1.0 - 2.0 * p * x - x ** 2)
        raise ValueError(f"x must be in 0.0 ..< 1.0")

    def dyc_dx(self, x: float) -> float:
        foil = self._foil
        m = foil.max_camber
        p = foil.camber_pos
        p_sqr = p * p
        if 0.0 <= x < p:
            return (2.0 * m / p_sqr) * (p - x)
        if p <= x < 1.0:
            return (2.0 * m / (1.0 - p) ** 2) * (p - x)
        raise ValueError(f"x must be in 0.0 ..< 1.0")

    def half_thickness(self, x: float) -> float:
        t = self._foil.thickness
        return (t / 0.2) * (
            0.2969 * x ** 0.5
            - 0.126 * x
            - 0.3516 * x ** 2
            + 0.2843 * x ** 3
            - 0.1015 * x ** 4
        )

    def upper_surface(self, x: float) -> Coord:
        dyc_dx = self.dyc_dx(x)
        theta = math.atan(dyc_dx)
        yt = self.half_thickness(x)
        xu = x - yt * math.sin(theta)
        yu = self.yc(x) + yt * math.cos(theta)
        return Coord(x=xu, y=yu)

    def lower_surface(self, x: float) -> Coord:
        dyc_dx = self.dyc_dx(x)
        theta = math.atan(dyc_dx)
        yt = self.half_thickness(x)
        xl = x + yt * math.sin(theta)
        yl = self.yc(x) - yt * math.cos(theta)
        return Coord(x=xl, y=yl)


def main() -> None:
    """Mainline for standalone execution."""
    spec = parse_id("NACA2412")
    maker = FoilMaker(spec)
    for point in maker.gen_coordinates():
        print(f"{point.x}, {point.y}")


if __name__ == "__main__":
    main()
