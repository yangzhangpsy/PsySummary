from PyQt5.QtWidgets import QDialog
from app.psyDataFunc import PsyDataFunc


class Dialog(QDialog):
    """
    not using.
    """

    def __init__(self, parent=None):
        super(Dialog, self).__init__(parent)
        # set its icon
        self.setWindowIcon(PsyDataFunc.getImageObject("icon.png", type=1))
