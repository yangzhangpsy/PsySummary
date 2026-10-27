"""Ordered filter rules with selection and drag lifetime owned by the view."""

from PyQt5 import sip
from PyQt5.QtCore import QItemSelectionModel, Qt
from PyQt5.QtGui import QDrag, QDragLeaveEvent, QPainter, QPen
from PyQt5.QtWidgets import QAbstractItemView, QMessageBox

from app.lib.list_widget import ListWidget


class OrderedFilterListWidget(ListWidget):
    """Commit internal moves only after the platform drag loop has returned."""

    MIME_TYPE = 'application/x-psybuilder-ordered-filters'

    def __init__(self, contextMenuType=1):
        super().__init__(contextMenuType)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        # Paint an insertion line even when Qt considers the cursor "on" a row.
        self.setDropIndicatorShown(False)
        self._drag_items = []
        self._drop_row = None
        self._pending_drop_row = None

    def startDrag(self, supported_actions):
        """Keep native drag cleanup independent of list mutations and host syncing."""
        if self._drag_items:
            return
        items = sorted(self.selectedItems(), key=self.row)
        if not items:
            return
        self._drag_items = items
        self._pending_drop_row = None
        drag = QDrag(self)
        mime = self.model().mimeData([self.indexFromItem(item) for item in items])
        mime.setData(self.MIME_TYPE, b'filters')
        drag.setMimeData(mime)
        rect = self.visualItemRect(items[0])
        drag.setPixmap(self.viewport().grab(rect))
        drag.setHotSpot(rect.center() - rect.topLeft())
        changed = False
        try:
            action = drag.exec_(Qt.MoveAction, Qt.MoveAction)
            if not sip.isdeleted(self) and action == Qt.MoveAction and self._pending_drop_row is not None:
                changed = self._move_items(items, self._pending_drop_row)
        finally:
            if not sip.isdeleted(self):
                super().dragLeaveEvent(QDragLeaveEvent())
                self._drag_items = []
                self._drop_row = None
                self._pending_drop_row = None
                self.setState(QAbstractItemView.NoState)
                self.viewport().update()
            if not sip.isdeleted(drag):
                drag.deleteLater()
        if changed:
            self.listChanged.emit('change')

    def _is_internal_drag(self, event):
        """Accept only this view's active rule drag, never external list content."""
        return (event.source() is self and bool(self._drag_items)
                and event.mimeData().hasFormat(self.MIME_TYPE))

    def _insertion_row(self, pos):
        """Use row midpoints so both halves and the space after the last row work."""
        item = self.itemAt(pos)
        if item is None:
            return self.count()
        return self.row(item) + int(pos.y() >= self.visualItemRect(item).center().y())

    def dragEnterEvent(self, event):
        if not self._is_internal_drag(event):
            event.ignore()
            return
        super().dragEnterEvent(event)
        event.setDropAction(Qt.MoveAction)
        event.accept()

    def dragMoveEvent(self, event):
        if not self._is_internal_drag(event):
            event.ignore()
            return
        # Retain Qt's edge autoscrolling; our midpoint rule controls insertion.
        super().dragMoveEvent(event)
        self._drop_row = self._insertion_row(event.pos())
        event.setDropAction(Qt.MoveAction)
        event.accept()
        self.viewport().update()

    def dragLeaveEvent(self, event):
        self._drop_row = None
        super().dragLeaveEvent(event)
        self.viewport().update()

    def dropEvent(self, event):
        """Record the drop without deleting or rebuilding items inside QDrag.exec_."""
        if not self._is_internal_drag(event):
            event.ignore()
            return
        self._pending_drop_row = self._insertion_row(event.pos())
        self._drop_row = None
        super().dragLeaveEvent(QDragLeaveEvent())
        event.setDropAction(Qt.MoveAction)
        event.accept()
        self.viewport().update()

    def _move_items(self, items, destination):
        """Move surviving item objects as one ordered block, preserving item data."""
        if any(sip.isdeleted(item) or self.row(item) < 0 for item in items):
            return False
        before = [self.item(row) for row in range(self.count())]
        rows = sorted(self.row(item) for item in items)
        items = [before[row] for row in rows]
        destination = max(0, min(destination, self.count()))
        insertion = destination - sum(row < destination for row in rows)
        selected_rows = set(rows)
        remaining = [item for row, item in enumerate(before) if row not in selected_rows]
        after = remaining[:insertion] + items + remaining[insertion:]
        if all(first is second for first, second in zip(before, after)):
            return False
        current = self.currentItem()
        for row in reversed(rows):
            self.takeItem(row)
        for offset, item in enumerate(items):
            self.insertItem(insertion + offset, item)
        self.clearSelection()
        self.setCurrentItem(current if any(current is item for item in items) else items[0],
                            QItemSelectionModel.NoUpdate)
        for item in items:
            item.setSelected(True)
        return True

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_row is None or not self.count():
            return
        row = min(self._drop_row, self.count() - 1)
        rect = self.visualItemRect(self.item(row))
        y = rect.top() if self._drop_row < self.count() else rect.bottom()
        painter = QPainter(self.viewport())
        painter.setPen(QPen(self.palette().highlight().color(), 2))
        painter.drawLine(0, max(1, y), self.viewport().width(), max(1, y))
        painter.end()

    def keyPressEvent(self, event):
        if event.key() != Qt.Key_Delete:
            super().keyPressEvent(event)
            return
        if self.selectedItems() and QMessageBox.question(
                self, 'Delete Item', 'Are you sure to delete the selected item(s)?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes:
            self.delete_item()
