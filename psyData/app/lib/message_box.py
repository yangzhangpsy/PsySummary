from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from app.psyDataFunc import PsyDataFunc


class MessageBox(QMessageBox):
    def __init__(self, *__args):
        super(MessageBox, self).__init__(*__args)
        # 将图标设置为psy icon
        self.setWindowIcon(PsyDataFunc.getImageObject("icon.png", type=1))
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        # self.setAttribute('-topmost',True)
