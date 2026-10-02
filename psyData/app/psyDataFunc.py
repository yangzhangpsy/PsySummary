import os
import re

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap, QIcon, QMovie

from app.psyDataInfo import PsyDataInfo


class PsyDataFunc(object):
    """
    This class is used to store the information of the data file.
    """

    _imageObjectCache = {}

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
        dock = PsyDataInfo.PsyData.script_dock
        previous_text = dock.text_edit.toPlainText()
        try:
            if isinstance(script, list):
                for cScript in script:
                    dock.printOut(cScript)
            elif isinstance(script, str):
                dock.printOut(script)
        except Exception:
            dock.text_edit.setPlainText(previous_text)
            raise

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
        size_key = (size.width(), size.height()) if size else None
        cache_key = (path, type, size_key)

        if type in (0, 1) and cache_key in PsyDataFunc._imageObjectCache:
            cached_object = PsyDataFunc._imageObjectCache[cache_key]
            return QPixmap(cached_object) if type == 0 else QIcon(cached_object)

        if type == 0:
            if size:
                image_object = QPixmap(path).scaled(size, transformMode=Qt.SmoothTransformation)
            else:
                image_object = QPixmap(path)
            PsyDataFunc._imageObjectCache[cache_key] = image_object
            return QPixmap(image_object)
        elif type == 1:
            if size:
                image_object = QIcon(QPixmap(path).scaled(size, transformMode=Qt.SmoothTransformation))
            else:
                image_object = QIcon(path)
            PsyDataFunc._imageObjectCache[cache_key] = image_object
            return QIcon(image_object)
        elif type == 2:
            movie = QMovie(path)
            if size:
                movie.setScaledSize(size)
            return movie
