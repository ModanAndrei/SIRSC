"""Compatibility entry point for the CPU training module."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.neural_network.train import main


if __name__ == "__main__":
    main()