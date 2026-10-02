import datetime
import os
import re
import time

from PyQt5.QtCore import QDir, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAction, QApplication, QFileDialog, QMessageBox, QTextEdit,
)

from app.lib import DockWidget


class Output(DockWidget):
    """
    This widget is used to output information about states of software.
    """
    realVisibleChanged = pyqtSignal(bool)

    def __init__(self, tabifyDock: bool = False, export_default_filename=None):
        super(Output, self).__init__()
        self.tabify_dock = tabifyDock
        # title
        self.setWindowTitle("Output")
        # main widget is a widget_name edit
        self.text_edit = OutputTextEdit(export_default_filename)
        self.text_edit.setReadOnly(True)
        self.scroll_bar = self.text_edit.verticalScrollBar()
        self.error_beep_pending = False
        self.last_error_beep_time = 0.0
        # first str is work path of this software
        self.text_edit.setHtml(f"<b>{QDir().currentPath()}</b>")
        self.text_edit.append('<p style="font:5px;color:white">.</p>')
        self.setWidget(self.text_edit)

        if self.tabify_dock:
            self.real_visible = False
            self.visibilityChanged.connect(self.setRealVisible)

    def setRealVisible(self, visible: bool):
        current_visible = not self.visibleRegion().isEmpty()
        if current_visible != self.real_visible:
            self.real_visible = current_visible
            self.realVisibleChanged.emit(current_visible)

    def printOut(self, information: str, information_type: int = 0, showTime=True) -> None:
        """
        print information in its widget_name edit
        :param information:
        :param information_type: 0 none
                                 1 success
                                 2 fail
                                 3 compile error
                                 4 warning
        :return:
        """
        if showTime:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.text_edit.append(f"<p>{timestamp}</p>")

        information = re.sub(r'\n', '<br>', information)
        # none
        if information_type == 0:
            self.text_edit.append(f"<p>{information}</p>")
        elif information_type == 1:
            self.text_edit.append(f'<b style="color:rgb(73,156,84)">[success]</b> {information}')
        elif information_type == 2:
            self.text_edit.append(f'<b style="color:rgb(199,84,80)">[fail]</b> {information}')
        elif information_type == 4:
            self.text_edit.append(f'<b style="color:rgb(199,84,80)">[warning]</b> {information}')
        elif information_type == 3:
            self.text_edit.append(f'<b style="color:rgb(255,84,80)">[error]</b> {information}')
            self.requestErrorBeep()
        self.text_edit.append('<p style="font:5px;color:white">.</p>')
        # to the bottom
        self.scroll_bar.setSliderPosition(self.scroll_bar.maximum())

    def requestErrorBeep(self) -> None:
        """Play one asynchronous alert for a burst of related error messages."""
        if self.error_beep_pending or time.monotonic() - self.last_error_beep_time < 1.0:
            return
        self.error_beep_pending = True
        QTimer.singleShot(0, self.playErrorBeep)

    def playErrorBeep(self) -> None:
        self.error_beep_pending = False
        self.last_error_beep_time = time.monotonic()
        QApplication.beep()

    def clear(self):
        """
        clear current_text
        :return:
        """
        self.text_edit.clearMe()
        # self.text_edit.setHtml(f"<b>{QDir().currentPath()}</b>")
        # self.text_edit.append('<p style="font:5px;color:white">.</p>')


class OutputTextEdit(QTextEdit):
    def __init__(self, export_default_filename=None):
        super(OutputTextEdit, self).__init__()
        self.setObjectName("OutputQTextEdit")
        self.export_default_filename = export_default_filename

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.openMenu)

    def openMenu(self, e):
        menu = self.createStandardContextMenu()
        menu.addSeparator()

        clearAction = QAction("Clear", self)
        clearAction.triggered.connect(self.clearMe)

        menu.addAction(clearAction)
        if self.export_default_filename:
            export_action = QAction("Export Log...", self)
            export_action.triggered.connect(self.exportLog)
            menu.addAction(export_action)
        menu.exec_(self.mapToGlobal(e))

    def exportLog(self):
        """Export the visible output as a UTF-8 plain-text log."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Export Output Log',
            self.export_default_filename,
            'Text Files (*.txt)',
        )
        if not file_path:
            return
        root, extension = os.path.splitext(file_path)
        if extension.lower() != '.txt':
            file_path = root + '.txt' if extension else file_path + '.txt'
        try:
            lines = []
            block = self.document().begin()
            while block.isValid():
                line = block.text()
                if line.strip() != '.':
                    lines.append(line)
                block = block.next()
            content = '\n'.join(lines).rstrip()
            if content:
                content += '\n'
            with open(file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(content)
        except OSError as error:
            QMessageBox.warning(self, 'Export Log Error', str(error))

    def clearMe(self):
        self.clear()
        self.setHtml(f"<b>{QDir().currentPath()}</b>")
        self.append('<p style="font:5px;color:white">.</p>')
