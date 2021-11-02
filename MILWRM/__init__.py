# -*- coding: utf-8 -*-
"""
Multiplex Image Labeling With Regional Morphology
"""
from .MILWRM import (
    mxif_labeler,
    st_labeler,
)
from .MxIF import img
from .ST import (
    map_pixels,
    trim_image,
    assemble_pita,
    show_pita,
)

__all__ = [
    "img",
    "map_pixels",
    "trim_image",
    "assemble_pita",
    "show_pita",
    "mxif_labeler",
    "st_labeler",
]

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
