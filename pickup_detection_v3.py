"""Step 6E corrected launcher.

This keeps the historical filename while delegating to the modular
single-target implementation in ``bottle_monitor.py``.
"""

from bottle_monitor import main


if __name__ == "__main__":
    main()
