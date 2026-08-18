"""Vendored Spectrum runtime without upstream-wide node registration.

H3 Relay registers only the Spectrum model wrapper it uses. Runtime hooks are
installed on the cloned model by that wrapper rather than globally at import.
"""

from .nodes import SpectrumApplyMiniMaxH3

__all__ = ["SpectrumApplyMiniMaxH3"]
