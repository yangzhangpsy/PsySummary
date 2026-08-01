import builtins
import keyword
import os
import re
import shutil

from PyQt5.QtCore import QRegularExpression, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QTextEdit

from app.lib.dock_widget import DockWidget


INITIAL_SCRIPT = 'from aggregateData import AggregateData\naggData = AggregateData()'


def makeTextFormat(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    textFormat = QTextCharFormat()
    textFormat.setForeground(QColor(color))
    if bold:
        textFormat.setFontWeight(QFont.Bold)
    textFormat.setFontItalic(italic)
    return textFormat


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Apply Python syntax colors without changing the document's plain text."""

    def __init__(self, document):
        super(PythonSyntaxHighlighter, self).__init__(document)

        keywords = '|'.join(re.escape(word) for word in keyword.kwlist)
        builtinNames = '|'.join(re.escape(name) for name in dir(builtins) if not name.startswith('_'))
        self.rules = [
            (QRegularExpression(r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?j?\b'),
             makeTextFormat('#F5871F'), 0),
            (QRegularExpression(r'\b[A-Za-z_]\w*(?=\s*\()'), makeTextFormat('#4271AE'), 0),
            (QRegularExpression(rf'\b(?:{builtinNames})\b'), makeTextFormat('#3E999F'), 0),
            (QRegularExpression(rf'\b(?:{keywords})\b'), makeTextFormat('#8959A8', bold=True), 0),
            (QRegularExpression(r'@[A-Za-z_]\w*'), makeTextFormat('#C99E00'), 0),
            (QRegularExpression(r'\b(?:def|class)\s+([A-Za-z_]\w*)'), makeTextFormat('#4271AE', bold=True), 1),
            (QRegularExpression(
                r'''(?i)(?:\b[rubf]{1,2})?(?:'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*")'''),
             makeTextFormat('#718C00'), 0),
            (QRegularExpression(r'#[^\n]*'), makeTextFormat('#8E908C', italic=True), 0),
        ]

    def highlightBlock(self, text: str) -> None:
        for pattern, textFormat, captureGroup in self.rules:
            matchIterator = pattern.globalMatch(text)
            while matchIterator.hasNext():
                match = matchIterator.next()
                start = match.capturedStart(captureGroup)
                length = match.capturedLength(captureGroup)
                if start >= 0 and length > 0:
                    self.setFormat(start, length, textFormat)


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
        self.text_edit.setPlainText(INITIAL_SCRIPT)
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
        self.text_edit.appendPlainTextLine(information)

    def setRealVisible(self, visible: bool):
        current_visible = not self.visibleRegion().isEmpty()
        if current_visible != self.real_visible:
            self.real_visible = current_visible
            self.realVisibleChanged.emit(current_visible)


class OutputTextEdit(QTextEdit):

    def __init__(self):
        super(OutputTextEdit, self).__init__()
        self.setObjectName("OutputQTextEdit")
        self.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.syntax_highlighter = PythonSyntaxHighlighter(self.document())

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

    def appendPlainTextLine(self, text: str) -> None:
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        if not self.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(str(text))
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def export(self):
        try:
            export_full_filename, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'Python Files (*.py)')
            if export_full_filename:
                with open(export_full_filename, 'w') as f:
                    f.write(self.toPlainText())

                # copy the aggregateData.py file
                current_directory = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exportFiles')

                output_path = os.path.dirname(export_full_filename)

                sourceFile = os.path.join(current_directory, 'aggregateData.py')
                shutil.copyfile(sourceFile, os.path.join(output_path, 'aggregateData.py'))

                sourceFile = os.path.join(current_directory, 'rtDist.py')
                shutil.copyfile(sourceFile, os.path.join(output_path, 'rtDist.py'))
        except Exception as e:
            print(e)

    def clearMe(self):
        self.setPlainText(INITIAL_SCRIPT)
