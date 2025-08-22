"""Module entry point for :mod:`qats.app`.

Executing ``python -m qats.app`` will invoke :func:`qats.app.main` and
start the graphical user interface.
"""

from . import main

if __name__ == "__main__":
    main()
