from PyQt5.QtWidgets import (
    QApplication, QListWidget, QMessageBox, QComboBox, QHBoxLayout, QMenu,
    QWidget, QAbstractItemView,
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer

from app.lib import MessageBox
from app.lib.list_widget import ListWidget
from app.lib.cognitiveModelDialog import edit_cognitive_model
from app.cognitiveModelSpec import COGNITIVE_MODEL_NAMES, is_cognitive_model


MODEL_SPEC_ROLE = Qt.UserRole + 101

DATA_OPERATIONS = [
    'Mean',
    'Median',
    'Mode',
    'Count',
    'Max',
    'Min',
    'Variance',
    'Standard Error',
    'Standard Deviation',
    'Gamma (k, θ)',
    'Shifted Gamma (k, θ, shift)',
    'Weibull (k, θ)',
    'Shifted Weibull (k, θ, shift)',
    'LogNormal (k, θ)',
    'Shifted LogNormal (k, θ, shift)',
    'Wald (m, a)',
    'Ex-Wald (m, a, τ)',
    'Shifted Wald (m, a, shift)',
    'Ex-Gaussian (μ, σ, τ)',
    'Inv-Gaussian (μ, λ)',
    'Shifted Inv-Gaussian (μ, λ, shift)',
] + list(COGNITIVE_MODEL_NAMES)


# for row and column variables
class DraggableListWidget(ListWidget):
    contentList = []
    RowType, ColumnType, DataType, VariableType = range(4)

    def __init__(self, listType: int = 0, contextMenuType: int = 1):
        super(DraggableListWidget, self).__init__(contextMenuType)
        self.list_type = listType

        self.itemLabel = None
        self.combo_box = None
        self.model_dataframe = None
        self.filtered_data_provider = None
        self._pending_single_click_item = None
        self._last_mouse_button = Qt.NoButton
        self._single_click_timer = QTimer(self)
        self._single_click_timer.setSingleShot(True)
        self._single_click_timer.timeout.connect(self._handle_pending_single_click)

        self.setAcceptDrops(True)
        self.setDragEnabled(True)

        self.setSelectionMode(QAbstractItemView.SingleSelection)  # 单选模式
        # self.setSelectionMode(QAbstractItemView.ExtendedSelection)  # 多选模式

        if self.list_type == self.VariableType:
            self.setDragDropOverwriteMode(False)
            self.setSortingEnabled(True)

        if self.list_type == self.DataType:
            self.itemClicked.connect(self.itemSingleClick)
            self.itemDoubleClicked.connect(self.itemDoubleClick)

    def removeItem(self, item):
        self.contentList.remove(item.text())
        super().removeItem(item)

    def setModelContext(self, dataframe, filtered_data_provider=None):
        """Provide source data and an optional current-filter data provider for model dialogs."""
        self.model_dataframe = dataframe
        self.filtered_data_provider = filtered_data_provider

    def modelSpecifications(self):
        """Return structured specifications keyed by their compact display text."""
        return {
            self.item(index).text(): self.item(index).data(MODEL_SPEC_ROLE)
            for index in range(self.count())
            if self.item(index).data(MODEL_SPEC_ROLE)
        }

    def restoreModelSpecifications(self, specifications):
        """Attach saved structured specifications to matching Data items."""
        specifications = specifications or {}
        for index in range(self.count()):
            item = self.item(index)
            specification = specifications.get(item.text())
            if specification:
                item.setData(MODEL_SPEC_ROLE, specification)

    def dragEnterEvent(self, event):
        if event.source() is self:
            self.setDragDropMode(QListWidget.InternalMove)
        else:
            self.setDragDropMode(QListWidget.DragDrop)
        super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        source_Widget = event.source()

        if not isinstance(source_Widget, (DraggableListWidget, VariableDraggableListWidget)):
            return None

        """
        for allowable drag-drop widgets
        """
        if self.list_type == self.VariableType:
            items = source_Widget.selectedItems()
            for item in items:
                source_Widget.removeItem(item)

            return None

        if source_Widget is self:
            super().dropEvent(event)
        else:
            items = source_Widget.selectedItems()
            source_Widget.clearSelection()
            source_Widget.clearMask()
            for item in items:
                text = item.text()

                if self.list_type == self.DataType:
                    # text = text.split('@')[0]
                    pure_text = text.split('@')[0]

                    for cDescriptive in DATA_OPERATIONS:
                        text = pure_text + "@" + cDescriptive
                        if text not in self.contentList:
                            break

                if source_Widget.list_type == self.DataType:
                    text = text.split('@')[0]

                if not (text in self.contentList):
                    self.addItem(text)
                    self.contentList.append(text)

                    added_item = self.item(self.count() - 1)
                    operation = text.split('@', 1)[1] if '@' in text else ''
                    if self.list_type == self.DataType and is_cognitive_model(operation):
                        if not self._configure_model_item(added_item, operation):
                            self.takeItem(self.row(added_item))
                            self.contentList.remove(text)
                            continue

                    if source_Widget.list_type in [self.RowType, self.ColumnType, self.DataType]:
                        source_Widget.removeItem(item)
                else:
                    msg = MessageBox(QMessageBox.Warning, "warning", f"The variable {text} already in the target list")
                    msg.exec_()
                    break

    def itemSingleClick(self, item):
        """Delay the analysis selector so a second click can open model settings instead."""
        if self._last_mouse_button != Qt.LeftButton:
            return
        self._pending_single_click_item = item
        self._single_click_timer.start(QApplication.doubleClickInterval())

    def mousePressEvent(self, event):
        """Remember which mouse button produced the subsequent item-click signal."""
        self._last_mouse_button = event.button()
        super().mousePressEvent(event)

    def _handle_pending_single_click(self):
        """Open the operation selector for a valid item after the double-click interval."""
        item = self._pending_single_click_item
        self._pending_single_click_item = None
        if item is not None and self.row(item) >= 0:
            self._show_operation_selector(item)

    def itemDoubleClick(self, item):
        """Open settings when a cognitive-model Data item is double-clicked."""
        self._single_click_timer.stop()
        self._pending_single_click_item = None
        self._clear_operation_selector()
        try:
            currentText = item.text().split("@", 1)[1]
            if is_cognitive_model(currentText):
                self._configure_model_item(item, currentText)
        except (IndexError, ValueError):
            return

    def _show_operation_selector(self, item):
        """Replace one Data item temporarily with its analysis-operation combo box."""
        try:
            current_text = item.text().split('@', 1)[1]
            operations = list(DATA_OPERATIONS)
            operations.remove(current_text)
            operations.insert(0, current_text)
        except (IndexError, ValueError):
            return

        self._clear_operation_selector()
        self.combo_box = QComboBox()
        self.combo_box.addItems(operations)
        self.itemLabel = item
        item_widget = QWidget()
        layout = QHBoxLayout(item_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo_box)
        layout.setStretch(0, 1)
        self.setItemWidget(item, item_widget)
        self.combo_box.setMaximumWidth(int(item_widget.sizeHint().width() * 0.88))
        self.combo_box.activated[str].connect(self.confirmBox)
        self.combo_box.setFocus(Qt.MouseFocusReason)
        QTimer.singleShot(0, self.combo_box.showPopup)

    def _clear_operation_selector(self):
        """Remove the temporary analysis-operation combo box, if one is active."""
        if self.itemLabel is not None and self.row(self.itemLabel) >= 0:
            self.setItemWidget(self.itemLabel, None)
        if self.combo_box is not None:
            self.combo_box.deleteLater()
        self.itemLabel = None
        self.combo_box = None

    def ShowContextMenu(self, pos):
        """Show Data-item analysis actions in addition to the standard delete actions."""
        if self.list_type != self.DataType:
            super().ShowContextMenu(pos)
            return
        item = self.itemAt(pos)
        menu = self._create_data_context_menu(item)
        menu.exec_(self.mapToGlobal(pos))

    def _create_data_context_menu(self, item):
        """Build the Data-list context menu for the item under the pointer."""
        menu = QMenu(self)
        if item is not None:
            self.setCurrentItem(item)
            change_action = menu.addAction('Change Analysis...')
            change_action.triggered.connect(lambda: self._show_operation_selector(item))
            try:
                operation = item.text().split('@', 1)[1]
            except IndexError:
                operation = ''
            settings_action = menu.addAction('Model Settings...')
            settings_action.setEnabled(is_cognitive_model(operation))
            if is_cognitive_model(operation):
                settings_action.triggered.connect(
                    lambda: self._configure_model_item(item, operation))
            menu.addSeparator()
        delete_action = menu.addAction('Delete')
        delete_action.triggered.connect(self.delete_item)
        delete_all_action = menu.addAction('Delete All')
        delete_all_action.triggered.connect(self.delete_all_item)
        return menu

    def confirmBox(self, text):
        try:
            item = self.itemLabel
            prev_text = item.text()
            previous_specification = item.data(MODEL_SPEC_ROLE)
            self._clear_operation_selector()
            if is_cognitive_model(text):
                if not self._configure_model_item(
                        item, text,
                        previous_specification if previous_specification
                        and previous_specification.get('model') == text else None):
                    return
            else:
                item.setData(MODEL_SPEC_ROLE, None)
            new_text = prev_text.split("@", 1)[0] + "@" + text
            item.setText(new_text)
            self.contentList.remove(prev_text)
            self.contentList.append(new_text)
        except Exception as e:
            print(e)

    def _configure_model_item(self, item, model, specification=None):
        """Open model settings and attach accepted structured data to one item."""
        if self.model_dataframe is None or self.model_dataframe.empty:
            QMessageBox.warning(self, 'Model Settings', 'Load data before configuring a cognitive RT model.')
            return False
        dataframe = self.model_dataframe
        if self.filtered_data_provider is not None:
            try:
                dataframe = self.filtered_data_provider()
            except Exception as error:
                QMessageBox.warning(
                    self, 'Model Settings', f'Could not apply the current filters: {error}')
                return False
        if dataframe is None or dataframe.empty:
            QMessageBox.warning(
                self, 'Model Settings', 'No rows remain after applying the current filters.')
            return False
        rt_variable = item.text().split('@', 1)[0]
        if specification is None:
            saved_specification = item.data(MODEL_SPEC_ROLE)
            if saved_specification and saved_specification.get('model') == model:
                specification = saved_specification
        configured = edit_cognitive_model(
            dataframe, model, rt_variable, specification, self)
        if configured is None:
            return False
        item.setData(MODEL_SPEC_ROLE, configured)
        return True

    def clear(self, clearContentList=True):
        self._single_click_timer.stop()
        self._pending_single_click_item = None
        self._clear_operation_selector()
        if clearContentList:
            self.contentList = []
        super().clear()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:  # Delete key
            selected_items = self.selectedItems()
            if selected_items:
                reply = QMessageBox.question(self, 'Delete Item(s)', 'Are you sure to delete the selected item(s)?',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    for item in selected_items:
                        # self.takeItem(self.row(item))
                        self.removeItem(item)
                        # self.contentList.remove(item.text())


class VariableDraggableListWidget(ListWidget):
    RowType, ColumnType, DataType, VariableType = range(4)

    def __init__(self, listType: int = 0, contextMenuType: int = 2):
        super().__init__(contextMenuType)
        self.list_type = listType
        self.itemLabel = None
        self.combo_box = None
        self.setAcceptDrops(True)
        # self.setContextMenuPolicy(Qt.CustomContextMenu)

        self.setDragEnabled(True)  # 开启拖拽功能
        # self.setDragDropOverwriteMode(False)  # 不能拖入
        self.setDropIndicatorShown(False)
        self.setSortingEnabled(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)  # 多选模式

        # self.customContextMenuRequested.connect(self.show_context_menu)

    def dropEvent(self, event) -> None:
        source_Widget = event.source()
        if source_Widget.list_type in [self.RowType, self.DataType, self.ColumnType]:
            items = source_Widget.selectedItems()
            for item in items:
                source_Widget.removeItem(item)
        return None


# for filter list
class FilterDragListWidget(ListWidget):
    # 自定义信号
    # listChanged: pyqtSignal = pyqtSignal(str)

    def __init__(self, contextMenuType: int = 1):
        super().__init__(contextMenuType)

        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setSelectionMode(QListWidget.MultiSelection)

    # 判断拖动后是否完成交换位置，以便及时同步
    def dropEvent(self, event):
        old_row = self.currentRow()  # 保存拖动前的位置
        super().dropEvent(event)
        new_row = self.currentRow()  # 获取拖动后的位置
        if old_row != new_row:
            self.listChanged.emit("change")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:  # Delete key
            selected_items = self.selectedItems()
            if selected_items:
                reply = QMessageBox.question(self, 'Delete Item', 'Are you sure to delete the selected item(s)?',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    for item in selected_items:
                        self.removeItem(item)
                        # self.takeItem(self.row(item))

                    self.listChanged.emit("delete")


class MainFilterListWidget(ListWidget):
    def __init__(self, contextMenuType: int = 1):
        super().__init__(contextMenuType)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setSelectionMode(QListWidget.MultiSelection)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:  # Delete key
            selected_items = self.selectedItems()
            if selected_items:
                reply = QMessageBox.question(self, 'Delete Item', 'Are you sure to delete the selected item(s)?',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    for item in selected_items:
                        self.removeItem(item)
                        # self.takeItem(self.row(item))
