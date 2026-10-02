import builtins
import keyword
import os
import re
import shutil
import tempfile

from PyQt5.QtCore import QRegularExpression, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import QTextEdit, QAction, QApplication, QFileDialog, QMessageBox

from app.lib import DockWidget


INITIAL_SCRIPT = 'from aggregateData import AggregateData\naggData = AggregateData()'


def export_analysis_bundle(script_path, script_text, helper_sources):
    """Stage a complete export and restore prior files if publication fails."""
    script_path = os.path.abspath(script_path)
    directory, script_name = os.path.split(script_path)
    if script_name.casefold() in {name.casefold() for name in helper_sources}:
        raise ValueError('Choose a script filename different from the exported helper modules.')
    names = list(helper_sources) + [script_name]
    for name in names:
        destination = os.path.join(directory, name)
        if os.path.lexists(destination) and (os.path.islink(destination) or not os.path.isfile(destination)):
            raise ValueError(f'Cannot overwrite a directory or symbolic link: {destination}')
    staging = tempfile.mkdtemp(prefix='.psysummary-export-', dir=directory)
    backups, published = {}, []
    try:
        for name, source in helper_sources.items():
            shutil.copyfile(source, os.path.join(staging, name))
        with open(os.path.join(staging, script_name), 'w', encoding='utf-8') as stream:
            stream.write(script_text)
        backup_directory = os.path.join(staging, 'backups')
        os.mkdir(backup_directory)
        for name in names:
            destination = os.path.join(directory, name)
            if os.path.exists(destination):
                backup = os.path.join(backup_directory, name)
                shutil.copy2(destination, backup)
                backups[name] = backup
        # Publish the entry script last, after all of its dependencies.
        for name in names:
            os.replace(os.path.join(staging, name), os.path.join(directory, name))
            published.append(name)
    except Exception as error:
        recovery_errors = []
        for name in reversed(published):
            try:
                destination = os.path.join(directory, name)
                if name in backups:
                    os.replace(backups[name], destination)
                else:
                    os.remove(destination)
            except OSError as recovery_error:
                recovery_errors.append(str(recovery_error))
        if recovery_errors:
            raise RuntimeError(f'Export failed: {error}. Recovery is incomplete; '
                               f'keep the backup files in {staging}. '
                               + '; '.join(recovery_errors)) from error
        try:
            shutil.rmtree(staging)
        except OSError as cleanup_error:
            raise RuntimeError(f'Export failed: {error}. Output files were restored, '
                               f'but temporary files remain in {staging}: {cleanup_error}') from error
        raise
    try:
        shutil.rmtree(staging)
    except OSError as error:
        return f'Export completed, but temporary files remain in {staging}: {error}'
    return ''


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
                source_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                sources = {name: os.path.join(source_directory, 'exportFiles', name)
                           for name in ('aggregateData.py', 'rtDist.py')}
                for helper_name in ('cognitiveModels.py', 'cognitiveModelSpec.py', 'expression.py'):
                    sources[helper_name] = os.path.join(source_directory, helper_name)
                warning = export_analysis_bundle(export_full_filename, self.toPlainText(), sources)
                if warning:
                    QMessageBox.warning(self, 'Export Script', warning)
        except Exception as e:
            QMessageBox.warning(self, 'Export Script Error', str(e))

    def clearMe(self):
        self.setPlainText(INITIAL_SCRIPT)
