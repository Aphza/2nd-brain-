"""Alias of :mod:`ai_metadata`.

In the original (compiled) project this module was a byte-for-byte duplicate of
``ai_metadata.py``. To avoid maintaining the provider-rotation and memory logic
in two places, it now re-exports everything from ``ai_metadata`` and shares a
single implementation.
"""

from ai_metadata import *  # noqa: F401,F403
from ai_metadata import main

if __name__ == "__main__":
    main()
