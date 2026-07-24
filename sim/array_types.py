"""
Shared array type aliases for the Fig. 14 derivation.

Every numpy array in this project plays one of a few well-defined roles. These
aliases name the role and carry the intended shape *inside the type* via
`Annotated`, so hovering a parameter shows e.g. "shape (competitions, territories)"
right there. A variable docstring under each alias adds a short example.

Shape convention:
    grid    -> shape (competitions, territories)   [one value per territory, per competition]
    vector  -> shape (competitions,)               [one value per competition]
    flexible-> scalar or array of any shape        [elementwise helpers like utility()]
"""
from __future__ import annotations

from typing import Annotated, TypeAlias

import numpy as np
from numpy.typing import NDArray

# --- grids: one value per (competition, territory) ---

ResourceValueGrid: TypeAlias = Annotated[NDArray[np.integer], "shape (competitions, territories)"]
"""Resource value each territory holds. Values 1..MAX_RESOURCE_VALUE.
Example (2 competitions x 3 territories): [[20, 55, 90], [1, 100, 50]]."""

PreferenceGrid: TypeAlias = Annotated[NDArray[np.floating], "shape (competitions, territories)"]
"""How much a strategy wants each territory (higher = more wanted); ties allowed.
Example - Truth's worth for [[20, 55, 90]]: [[32.5, 96.9, 13.5]] (it most wants territory 1)."""

AvailabilityGrid: TypeAlias = Annotated[NDArray[np.bool_], "shape (competitions, territories)"]
"""True where a territory is still free to be claimed, False where already taken.
Example - after the first mover took territory 1: [[True, False, True]]."""

# --- vectors: one value per competition ---

TerritoryChoices: TypeAlias = Annotated[NDArray[np.intp], "shape (competitions,)"]
"""Index of the territory chosen in each competition.
Example (3 competitions): [1, 0, 2]."""

Payoffs: TypeAlias = Annotated[NDArray[np.floating], "shape (competitions,)"]
"""Utility collected in each competition.
Example (3 competitions): [96.9, 100.0, 88.2]."""

# --- categories: describing how resource values are grouped into bands ---

Boundaries: TypeAlias = Annotated[NDArray[np.floating], "shape (num_boundaries,), sorted ascending"]
"""Category cut-points, sorted low->high. B boundaries define B+1 contiguous bands.
Flexible in count: 2 for CR3, 4 for IF3, any number. Example - CR3's cuts: [30.0, 70.0]."""

CategoryValues: TypeAlias = Annotated[NDArray[np.intp], "same shape as the input resource values"]
"""Band index each resource value falls into, 0 (lowest) .. B (highest); shape matches the input.
Example - values [10, 50, 90] under boundaries [30, 70]: [0, 1, 2]."""

BandExpectedUtilities: TypeAlias = Annotated[NDArray[np.floating], "shape (num_bands,) = (num_boundaries + 1,)"]
"""Average utility of a territory whose value lands in each band (empty bands = -inf).
Example - boundaries [30, 70]: [worth of low band, worth of middle band, worth of high band]."""

ZoneLabels: TypeAlias = Annotated[NDArray[np.intp], "shape (num_zones,) = (num_boundaries + 1,)"]
"""Maps each contiguous zone to a perceptual label; a label may repeat (non-contiguous).
Example - CR3 (identity): [0, 1, 2]. IF3 (fold tails & shoulders): [0, 1, 2, 1, 0]."""

# --- flexible: elementwise helpers that accept a scalar or any-shaped array ---

ResourceValues: TypeAlias = Annotated[NDArray[np.integer] | float, "scalar or array of any shape"]
"""Resource value(s) for an elementwise helper - a single number or any-shaped array.
Example: 50, or [20, 55, 90], or a full ResourceValueGrid."""

UtilityValues: TypeAlias = Annotated[NDArray[np.floating], "same shape as the input"]
"""Worth value(s) returned elementwise, matching the shape of the input resource values."""
