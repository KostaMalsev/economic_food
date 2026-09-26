"""Backward-compatible entry point for the regression method.

New code should import from ``analysis.regression.model``.
"""

from analysis.regression.model import FamilyGroupAnalyzer, main

__all__ = ["FamilyGroupAnalyzer", "main"]


if __name__ == "__main__":
    main()
