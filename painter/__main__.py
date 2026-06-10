"""Entry point: python -m painter [path/to/model.stl|.3mf|.obj]"""

import sys

from . import mesh
from .viewer import Viewer


def main() -> None:
    if len(sys.argv) > 1:
        m = mesh.load(sys.argv[1])
    else:
        print("No model given - showing demo torus. Usage: python -m painter model.stl")
        m = mesh.demo()
    Viewer(m).run()


if __name__ == "__main__":
    main()
