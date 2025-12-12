"""
Backward compatibility module for old pickle files.

This module was reorganized into Modules.parameters.constants.
This file exists to maintain compatibility with pickle files that reference
the old module path 'Modules.constants'.
"""

# Re-export everything from the new location
from Modules.parameters.constants import *

# Explicitly re-export the main classes for clarity
from Modules.parameters.constants import (
    SimulationParameters,
    HayParameters,
)

__all__ = ['SimulationParameters', 'HayParameters']
