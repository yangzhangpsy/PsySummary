"""Compatibility helpers for PsySummary code shared with PsyBuilder."""

import os

from app.psyDataFunc import PsyDataFunc


class Func:
    """Delegate shared image and output operations to the standalone app."""

    _IMAGE_ALIASES = {
        "common/icon.png": "icon.png",
        "menu/checked": "checked.png",
    }

    @staticmethod
    def getImageObject(image_path, type=0, size=None):
        image_path = Func._IMAGE_ALIASES.get(image_path, os.path.basename(image_path))
        return PsyDataFunc.getImageObject(image_path, type, size)

    @staticmethod
    def printOut(information, information_type=0):
        return PsyDataFunc.printOut(information, information_type)
