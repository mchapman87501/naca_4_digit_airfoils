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


def gen_xvals(num_steps: int) -> tp.Sequence[float]:
    # Reduce flat segments near leading edge:
    beta = 0.0
    d_beta = math.pi / num_steps
    while beta <= math.pi:
        yield (1.0 - math.cos(beta)) / 2.0
        beta += d_beta


class FoilMaker:
    def __init__(self, foil: NACAFoil) -> None:
        self._foil = foil

    def gen_env_coordinates(self) -> tp.Sequence[Coord]:
        num_steps = 20
        # Generate upper surface from 0.0 to 1.0
        xvals = list(gen_xvals(num_steps))
        for x in xvals:
            yield self.upper_surface(x)
        for x in reversed(xvals):
            yield self.lower_surface(x)

    def yc(self, x: float) -> float:
        foil = self._foil
        m = foil.max_camber
        p = foil.camber_pos
        p_sqr = p * p
        if 0.0 <= x < p:
            return (m / p_sqr) * (2.0 * p * x - x ** 2)
        if p <= x <= 1.0:
            return (m / (1.0 - p) ** 2) * (
                1.0 - 2.0 * p + 2.0 * p * x - x ** 2
            )
        raise ValueError(f"x must be in 0.0 ... 1.0")

    def dyc_dx(self, x: float) -> float:
        foil = self._foil
        m = foil.max_camber
        p = foil.camber_pos
        p_sqr = p * p
        if 0.0 <= x < p:
            return (2.0 * m / p_sqr) * (p - x)
        if p <= x <= 1.0:
            return (2.0 * m / (1.0 - p) ** 2) * (p - x)
        raise ValueError(f"x must be in 0.0 ... 1.0")

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
        yc = self.yc(x)
        xu = x - yt * math.sin(theta)
        yu = yc + yt * math.cos(theta)
        return Coord(x=xu, y=yu)

    def lower_surface(self, x: float) -> Coord:
        dyc_dx = self.dyc_dx(x)
        theta = math.atan(dyc_dx)
        yt = self.half_thickness(x)
        yc = self.yc(x)
        xl = x + yt * math.sin(theta)
        yl = yc - yt * math.cos(theta)
        return Coord(x=xl, y=yl)

    def gen_camber_coords(self) -> tp.Sequence[Coord]:
        num_steps = 20
        # Generate upper surface from 0.0 to 1.0
        xvals = list(gen_xvals(num_steps))
        for x in xvals:
            y = self.yc(x)
            yield Coord(x=x, y=y)


def main() -> None:
    """Mainline for standalone execution."""
    spec = parse_id("NACA2412")
    print(
        f"Max camber {spec.max_camber} occurs at {spec.camber_pos}.  Thickness is {spec.thickness}."
    )
    maker = FoilMaker(spec)
    print("# Envelope")
    for point in maker.gen_env_coordinates():
        print(f"{point.x:.4f}, {point.y:.4f}")
    print("")
    print("# Camber line")
    for point in maker.gen_camber_coords():
        print(f"{point.x:.4f}, {point.y:.4f}")


if __name__ == "__main__":
    main()
