#!/usr/bin/env python3
"""This simple script prints/plots (x, y) coordinates for NACA airfoils."""

import argparse
from dataclasses import dataclass
import math
import re
import typing as tp

import matplotlib.pyplot as plt


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


CoordIter = tp.Iterable[Coord]
CoordSeq = tp.Sequence[Coord]


def gen_xvals(num_steps: int) -> tp.Iterable[float]:
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

    def gen_env_coordinates(self) -> CoordIter:
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
            return (m / p_sqr) * (2.0 * p * x - x**2)
        if p <= x <= 1.0:
            return (m / (1.0 - p) ** 2) * (
                1.0 - 2.0 * p + 2.0 * p * x - x**2
            )
        raise ValueError("x must be in 0.0 ... 1.0")

    def _dyc_dx(self, x: float) -> float:
        foil = self._foil
        m = foil.max_camber
        p = foil.camber_pos
        p_sqr = p * p
        if 0.0 <= x < p:
            return (2.0 * m / p_sqr) * (p - x)
        if p <= x <= 1.0:
            return (2.0 * m / (1.0 - p) ** 2) * (p - x)
        raise ValueError("x must be in 0.0 ... 1.0")

    def _half_thickness(self, x: float) -> float:
        t = self._foil.thickness
        return (t / 0.2) * (
            0.2969 * x**0.5
            - 0.126 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
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

    def gen_camber_coords(self) -> CoordIter:
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


class CoordPrinter:
    """Print airfoil shape coordinates."""

    def __init__(self, swift_syntax: bool) -> None:
        """Initialize a new instance.

        Args:
            swift_syntax (bool): Format output as Swift-lang code
                                 (if True) or as comma-separated values
        """
        self._swift_syntax = swift_syntax
        self._comment = "//" if swift_syntax else "#"
        coord_prefix = "    (" if self._swift_syntax else ""
        coord_suffix = ")," if self._swift_syntax else ""
        self._item_fmt = f"{coord_prefix}{{x:.4f}}, {{y:.4f}}{coord_suffix}"

    def print_foil(
        self, foil_id: str, envelope: CoordSeq, camber_line: CoordSeq
    ) -> None:
        """Print coords of an airfoil envelope and its camber line (if given).

        Args:
            foil_id (str): NACA airfoil ID, e.g., 'NACA2412'
            envelope (CoordSeq): (x, y) coords of airfoil envelope
            camber_line (CoordSeq): (x, y) coords of foil camber line
                                    If camber_line is empty it is not printed.
        """
        print(self._comment, foil_id)
        self._print_envelope(envelope)
        if camber_line:
            self._print_camber(camber_line)

    def _print_envelope(self, envelope: CoordIter) -> None:
        if self._swift_syntax:
            print("let envelope = [")
        else:
            print("# Envelope")

        fmt = self._item_fmt
        for point in envelope:
            print(fmt.format(x=point.x, y=point.y))

        if self._swift_syntax:
            print("]")
        print()

    def _print_camber(self, camber_line: CoordIter) -> None:
        if self._swift_syntax:
            print("let camber_line = [")
        else:
            print("# Mean Camber Line")

        fmt = self._item_fmt
        for point in camber_line:
            print(fmt.format(x=point.x, y=point.y))
        if self._swift_syntax:
            print("]")
        print()


class Plotter:
    """Plots foil shapes using matplotlib."""

    def plot(
        self, foil_id: str, envelope: CoordSeq, camber_line: CoordSeq
    ) -> None:
        """Plot an airfoil envelope and camber line (if given).

        Args:
            foil_id (str): NACA airfoil ID, e.g., 'NACA2412'
            envelope (CoordSeq): (x, y) coords of airfoil envelope
            camber_line (CoordSeq): (x, y) coords of foil camber line
                                    If camber_line is empty it is not plotted.
        """
        _ = plt.figure()
        plt.title(foil_id)
        plt.axis("equal")

        self._plot_envelope(envelope)
        if camber_line:
            self._plot_camber(camber_line)

    def _plot_envelope(self, envelope: CoordSeq) -> None:
        x = [p.x for p in envelope]
        y = [p.y for p in envelope]
        plt.plot(x, y, marker="o", linestyle="-", color="b")

    def _plot_camber(self, camber_line: CoordSeq) -> None:
        x = [p.x for p in camber_line]
        y = [p.y for p in camber_line]
        plt.plot(x, y, "k--")


def _parse_cmdline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate coordinates for NACA airfoils."
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
    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        default=False,
        help=(
            "Display an image of the foil envelope "
            "(and mean camber line, if requested)."
        ),
    )
    parser.add_argument(
        "-s",
        "--swift",
        action="store_true",
        default=False,
        help="Print coordinates as a Swift-lang code fragment.",
    )

    return parser.parse_args()


def main() -> None:
    """Mainline for standalone execution."""
    args = _parse_cmdline()

    printer = CoordPrinter(args.swift)
    plotter = Plotter()

    for foil_id in args.naca_4_digit:
        spec = parse_id(foil_id)
        maker = FoilMaker(spec, args.num_points)

        env = list(maker.gen_env_coordinates())
        camber = [] if not args.camber else list(maker.gen_camber_coords())
        printer.print_foil(foil_id, env, camber)

        if args.render:
            plotter.plot(foil_id, env, camber)

    if args.render:
        plt.show()


if __name__ == "__main__":
    main()
