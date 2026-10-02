import os
import platform


class Info:
    """Standalone paths and platform settings used by Data Summary."""

    OS_TYPE = {'Windows': 0, 'Darwin': 1}.get(platform.system(), 2)
    FILE_DIRECTORY = ''
    UserPath = os.path.expanduser('~')
    ConfigFile = os.path.join(UserPath, '.psysummary.ini')
