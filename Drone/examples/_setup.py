"""Lets the examples import the `drone` package when run as `python examples/NN_name.py`."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUTPUT = os.path.join(ROOT, "output")
