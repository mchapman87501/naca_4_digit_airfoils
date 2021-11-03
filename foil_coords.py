#!/usr/bin/env python3
import argparse
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
    """
    FoilMaker generates (x, y) coordinates from a NACAFoil.
    """

    def __init__(self, foil: NACAFoil, num_points: int) -> None:
        """Initialize a new instance.

        Args:
            foil (NACAFoil): the foil for which to generate coordinates
            num_points (int): the number of points to produce
        """
        self._foil = foil
        self._num_points = num_points

    def gen_env_coordinates(self) -> tp.Sequence[Coord]:
        """Generate (x, y) envelope/hull coordinates.

        Yields:
            Iterator[tp.Sequence[Coord]]: envelope (x, y) Coords
        """
        num_steps = self._num_points // 2
        # Generate upper surface from 0.0 to 1.0
        xvals = list(gen_xvals(num_steps))
        for x in xvals:
            yield self._upper_surface(x)
        for x in reversed(xvals):
            yield self._lower_surface(x)

    def _y_chord(self, x: float) -> float:
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

    def _dyc_dx(self, x: float) -> float:
        foil = self._foil
        m = foil.max_camber
        p = foil.camber_pos
        p_sqr = p * p
        if 0.0 <= x < p:
            return (2.0 * m / p_sqr) * (p - x)
        if p <= x <= 1.0:
            return (2.0 * m / (1.0 - p) ** 2) * (p - x)
        raise ValueError(f"x must be in 0.0 ... 1.0")

    def _half_thickness(self, x: float) -> float:
        t = self._foil.thickness
        return (t / 0.2) * (
            0.2969 * x ** 0.5
            - 0.126 * x
            - 0.3516 * x ** 2
            + 0.2843 * x ** 3
            - 0.1015 * x ** 4
        )

    def _upper_surface(self, x: float) -> Coord:
        dyc_dx = self._dyc_dx(x)
        theta = math.atan(dyc_dx)
        yt = self._half_thickness(x)
        yc = self._y_chord(x)
        xu = x - yt * math.sin(theta)
        yu = yc + yt * math.cos(theta)
        return Coord(x=xu, y=yu)

    def _lower_surface(self, x: float) -> Coord:
        dyc_dx = self._dyc_dx(x)
        theta = math.atan(dyc_dx)
        yt = self._half_thickness(x)
        yc = self._y_chord(x)
        xl = x + yt * math.sin(theta)
        yl = yc - yt * math.cos(theta)
        return Coord(x=xl, y=yl)

    def gen_camber_coords(self) -> tp.Sequence[Coord]:
        """Generate (x, y) coordinates of the mean camber line.

        Yields:
            Iterator[tp.Sequence[Coord]]: (x, y) coords of the camber line)
        """
        num_steps = self._num_points
        # Generate upper surface from 0.0 to 1.0
        xvals = list(gen_xvals(num_steps))
        for x in xvals:
            y = self._y_chord(x)
            yield Coord(x=x, y=y)


def _parse_cmdline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate coordinates for a NACA airfoil"
    )
    parser.add_argument(
        "naca_4_digit",
        nargs="+",
        help="NACA 4-digit foil specifier, e.g., 'NACA2412'",
    )
    parser.add_argument(
        "-p",
        "--num-points",
        type=int,
        default=24,
        help="number of coordinates to output",
    )
    parser.add_argument(
        "-c",
        "--camber",
        action="store_true",
        default=False,
        help="Also output coordinates of the mean camber line.",
    )

    return parser.parse_args()


def main() -> None:
    """Mainline for standalone execution."""
    args = _parse_cmdline()
    for foil_id in args.naca_4_digit:
        print(f"# {foil_id}")
        spec = parse_id(foil_id)
        maker = FoilMaker(spec, args.num_points)
        print("# Envelope")
        for point in maker.gen_env_coordinates():
            print(f"{point.x:.4f}, {point.y:.4f}")
        print()
        if args.camber:
            print("# Mean Camber Line")
            for point in maker.gen_camber_coords():
                print(f"{point.x:.4f}, {point.y:.4f}")
            print()


if __name__ == "__main__":
    main()
