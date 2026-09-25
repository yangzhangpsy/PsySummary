"""Standalone PsySummary application settings used by the shared GUI code."""

import os
import platform


class Info:
    """Provide the small subset of PsyBuilder settings required by PsySummary."""

    OS_TYPE = {"Windows": 0, "Darwin": 1}.get(platform.system(), 2)
    FILE_DIRECTORY = ""
    UserPath = os.path.expanduser("~")
    ConfigFile = os.path.join(UserPath, ".psysummary.ini")
