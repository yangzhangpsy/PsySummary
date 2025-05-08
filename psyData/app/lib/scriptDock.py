import html
import os
import shutil

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QTextEdit, QAction, QApplication, QFileDialog

from app.lib.dock_widget import DockWidget
from app.psyDataInfo import PsyDataInfo


class ScriptDock(DockWidget):
    """
    This widget is used to display information about analysis script.
    """
    realVisibleChanged = pyqtSignal(bool)

    def __init__(self):
        super(ScriptDock, self).__init__()
        # title
        self.setWindowTitle("Script")
        # main widget is a widget_name edit
        self.text_edit = OutputTextEdit()
        self.text_edit.setReadOnly(True)
        self.real_visible = False
        self.scroll_bar = self.text_edit.verticalScrollBar()
        self.text_format = "<p style='line-height:1px; width:100% ; white-space: pre-wrap; margin:0 5px; '>"
        # first str is work path of this software
        self.text_edit.setHtml(f"{self.text_format}from aggregateData import AggregateData</p>")
        self.text_edit.append(f"{self.text_format}aggData = AggregateData()</p>")
        # self.text_edit.append(f"<p>{information}</p>")
        self.setWidget(self.text_edit)
        self.visibilityChanged.connect(self.setRealVisible)

    def clear(self):
        """
        clear current_text
        :return:
        """
        self.text_edit.clearMe()

    def printOut(self, information: str):
        self.text_edit.append(
            f"<p style='line-height:1px; width:100% ; white-space: pre-wrap; margin:0 auto; '>{html.escape(information)}</p>")

    def setRealVisible(self, visible: bool):
        current_visible = not self.visibleRegion().isEmpty()
        if current_visible != self.real_visible:
            self.real_visible = current_visible
            self.realVisibleChanged.emit(current_visible)


class OutputTextEdit(QTextEdit):

    def __init__(self):
        super(OutputTextEdit, self).__init__()
        self.setObjectName("OutputQTextEdit")

        self.text_format = "<p style='line-height:1px; width:100% ; white-space: pre-wrap; margin:0 5px; '>"

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.openMenu)

    def openMenu(self, e):
        menu = self.createStandardContextMenu()
        menu.addSeparator()

        clearAction = QAction("Clear", self)
        clearAction.triggered.connect(self.clearMe)

        # copyAction = QAction("Copy", self)
        # copyAction.triggered.connect(self.copy)

        exportAction = QAction("Export", self)
        exportAction.triggered.connect(self.export)

        menu.addAction(clearAction)
        # menu.addAction(copyAction)
        menu.addAction(exportAction)

        menu.exec_(self.mapToGlobal(e))

    def copy(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.toPlainText())

    def export(self):
        try:
            export_full_filename, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'Python Files (*.py)')
            if export_full_filename:
                with open(export_full_filename, 'w') as f:
                    f.write(self.toPlainText())

                # copy the aggregateData.py file
                current_directory = os.path.join(PsyDataInfo.MAIN_DIR, "exportFiles")

                output_path = os.path.dirname(export_full_filename)

                sourceFile = os.path.join(current_directory, 'aggregateData.py')
                shutil.copyfile(sourceFile, os.path.join(output_path, 'aggregateData.py'))

                sourceFile = os.path.join(current_directory, 'rtDist.py')
                shutil.copyfile(sourceFile, os.path.join(output_path, 'rtDist.py'))
        except Exception as e:
            print(e)

    def clearMe(self):
        self.clear()
        self.setHtml(f"{self.text_format}from .aggregateData import AggregateData</p>")
        self.append(f"{self.text_format}aggData = AggregateData()</p>")
