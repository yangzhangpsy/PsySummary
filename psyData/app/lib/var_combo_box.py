import re

from PyQt5.QtCore import QDataStream, QIODevice, Qt, QRegExp, pyqtSignal
from PyQt5.QtGui import QRegExpValidator, QFont
from PyQt5.QtWidgets import QComboBox, QListView

from app.lib import MessageBox

class VarComboBox(QComboBox):
    focusLost = pyqtSignal()

    Attribute = r"^\[[_\d\.\w]+\]$"
    Float = r"^(-?\d+)(\.\d+)?$"
    Integer = r"^-?\d+$"
    Percentage = r"^(100|[1-9]?\d?)%$|0$"
    FloatPercentage = r"^(([1-9]{1}\d*)|(0{1}))(\.\d{0,2})?%$"

    def __init__(self, editable: bool = False, parent=None):
        super(VarComboBox, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setEditable(editable)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.currentTextChanged.connect(self.searchVariable)

        self.focusLost.connect(self.checkValidity)
        self.valid_data: str = self.currentText()
        self.reg_exp = ""
        self.AttributesToWidget = "3"
        self.DEFAULT_FONT = ""

        list_view = QListView()
        self.setView(list_view)

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(self.AttributesToWidget):
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        data = e.mimeData().data(self.AttributesToWidget)
        stream = QDataStream(data, QIODevice.ReadOnly)
        text = f"[{stream.readQString()}]"
        index = self.findText(text, Qt.MatchExactly)
        if index == -1:
            self.addItem(text)
        self.setCurrentText(text)

    # def setCurrentText(self, text: str) -> None:
    #     index = self.findText(text, Qt.MatchExactly)
    #     if index == -1:
    #         self.addItem(text)

    def addCurrentText(self, text: str):
        index = self.findText(text, Qt.MatchExactly)
        if index == -1:
            self.addItem(text)
        self.setCurrentText(text)

    # 检查变量
    def searchVariable(self, current_text: str):
        if current_text.startswith("[") and current_text.endswith("]"):
            self.setStyleSheet("color: blue")
            self.setFont(QFont(self.DEFAULT_FONT, 9, QFont.Bold))
        else:
            self.setStyleSheet("color: black")
            self.setFont(QFont(self.DEFAULT_FONT, 9, QFont.Normal))

    def setReg(self, reg_exp: str or list or tuple):
        if isinstance(reg_exp, str):
            self.reg_exp = f"{reg_exp}|{VarComboBox.Attribute}"
        elif isinstance(reg_exp, list) or isinstance(reg_exp, tuple):
            self.reg_exp = f"{'|'.join(reg_exp)}|{VarComboBox.Attribute}"
        self.setValidator(QRegExpValidator(QRegExp(self.reg_exp), self))

    def focusOutEvent(self, e):
        self.focusLost.emit()
        QComboBox.focusOutEvent(self, e)

    def checkValidity(self):
        cur = self.currentText()
        if self.reg_exp != "" and re.fullmatch(self.reg_exp, cur) is None:
            self.setCurrentText(self.valid_data)
            MessageBox.warning(self, "Invalid", f"Invalid Parameter '{cur}'\nFormat must conform to\n{self.reg_exp}")
        else:
            self.valid_data = cur
