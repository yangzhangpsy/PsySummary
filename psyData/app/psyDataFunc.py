import os
import re

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap, QIcon, QMovie

from app.psyDataInfo import PsyDataInfo


class PsyDataFunc(object):
    """
    This class is used to store the information of the data file.
    """

    @staticmethod
    def printOut(information: str, information_type: int = 0, showTime=True):
        """
        print information in output.
        :param information: output string
        :param information_type: 0 none
                                 1 success
                                 2 fail
                                 3 compile error
                                 4 warning
        :param showTime: True for showing the date info
        """
        PsyDataInfo.PsyData.output.printOut(information, information_type, showTime)

    @staticmethod
    def genScript(script: (str, list)):
        """
        append analysis script to the end of the script.
        :param script: analysis script
        """
        if isinstance(script, list):
            for cScript in script:
                PsyDataInfo.PsyData.script_dock.printOut(cScript)
        elif isinstance(script, str):
            PsyDataInfo.PsyData.script_dock.printOut(script)

    @staticmethod
    def list2Script(var_list: list, var_list_name: str):
        if len(var_list) > 0:
            temp_script = '"' + '", "'.join(var_list) + '"'
        else:
            temp_script = ''
        return f"{var_list_name} = [{temp_script}]"


    @staticmethod
    def getImageObject(image_path: str, type: int = 0, size: QSize = None) -> QPixmap or QIcon:
        """
        get image from its relative path, return qt image object, include QPixmap or QIcon.
        @param image_path: its relative path
        @param type: 0: pixmap (default),
                     1: icon
                     2: QMovie
        @return: Qt image object
        """

        path = os.path.join(PsyDataInfo.MAIN_DIR, "images", *(re.split(r'[\\/]', image_path)))

        # PsyDataInfo.PsyData.output.printOut(f"{path}", 0, True)

        if type == 0:
            if size:
                return QPixmap(path).scaled(size, transformMode=Qt.SmoothTransformation)
            return QPixmap(path)
        elif type == 1:
            if size:
                return QIcon(QPixmap(path).scaled(size, transformMode=Qt.SmoothTransformation))
            return QIcon(path)
        elif type == 2:
            if size:
                return QMovie(path).setScaledSize(size)
            return QMovie(path)


