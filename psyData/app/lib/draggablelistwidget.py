from PyQt5.QtWidgets import QListWidget, QMessageBox, QComboBox, QHBoxLayout, QWidget, QAbstractItemView
from PyQt5.QtCore import Qt

from app.lib.message_box import MessageBox
from app.lib.list_widget import ListWidget


# for row and column variables
class DraggableListWidget(ListWidget):
    contentList = []
    RowType, ColumnType, DataType, VariableType = range(4)

    def __init__(self, listType: int = 0, contextMenuType: int = 1):
        super(DraggableListWidget, self).__init__(contextMenuType)
        self.list_type = listType

        self.itemLabel = None
        self.combo_box = None

        self.setAcceptDrops(True)
        self.setDragEnabled(True)

        self.setSelectionMode(QAbstractItemView.SingleSelection)  # 单选模式
        # self.setSelectionMode(QAbstractItemView.ExtendedSelection)  # 多选模式

        if self.list_type == self.VariableType:
            self.setDragDropOverwriteMode(False)
            self.setSortingEnabled(True)

        if self.list_type == self.DataType:
            self.itemDoubleClicked.connect(self.itemDoubleClick)

    def removeItem(self, item):
        self.contentList.remove(item.text())
        super().removeItem(item)

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

                    lst = ['Mean', 'Median', 'Mode', 'Count', 'Standard Deviation', 'Max', 'Min', 'Variance',
                           'Standard Error', 'ex-Gaussian']

                    for cDescriptive in lst:
                        text = pure_text + "@" + cDescriptive
                        if text not in self.contentList:
                            break

                if source_Widget.list_type == self.DataType:
                    text = text.split('@')[0]

                if not (text in self.contentList):
                    self.addItem(text)
                    self.contentList.append(text)

                    if source_Widget.list_type in [self.RowType, self.ColumnType, self.DataType]:
                        source_Widget.removeItem(item)
                else:
                    msg = MessageBox(QMessageBox.Warning, "warning", f"The variable {text} already in the target list")
                    msg.exec_()
                    break

    def itemDoubleClick(self, item):
        try:
            self.combo_box = QComboBox()
            lst = ['Mean',
                   'Median',
                   'Mode',
                   'Count',
                   'Max',
                   'Min',
                   'Variance',
                   'Standard Error',
                   'Standard Deviation',
                   'Gamma (k, θ)',
                   'Weibull (k, θ)',
                   'LogNormal (k, θ)',
                   'Wald (m, a)',
                   'Ex-Wald (m, a, τ)',
                   'Shifted Wald (m, a, shift)',
                   'Ex-Gaussian (μ, σ, τ)',
                   'Inv-Gaussian (μ, λ)',
                   'Shifted Inv-Gaussian (μ, λ, shift)',
                   ]
            currentText = item.text().split("@")[1]
            # 把currentText交换道lst的第一个
            lst.remove(currentText)
            lst.insert(0, currentText)
            self.combo_box.addItems(lst)

            self.itemLabel = item
            # 创建水平布局
            h_layout = QHBoxLayout()

            # 添加下拉框到水平布局
            h_layout.addWidget(self.combo_box)

            # 创建项小部件并设置布局
            item_widget = QWidget()
            item_widget.setLayout(h_layout)

            # 设置项小部件的拉伸因子以填充整个项的高度
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.setStretch(0, 1)

            # 将项小部件设置为当前项的小部件
            listWidgetItem = self.item(self.row(item))
            self.setItemWidget(listWidgetItem, item_widget)

            combo_box_width = item_widget.sizeHint().width() * 0.88
            self.combo_box.setMaximumWidth(int(combo_box_width))

            # 当下拉框的选择发生变化时，触发删除下拉框的槽函数
            self.combo_box.activated[str].connect(self.confirmBox)
        except:
            pass

    def confirmBox(self, text):
        try:
            self.combo_box.deleteLater()  # 删除下拉框
            prev_text = self.itemLabel.text()
            new_text = prev_text.split("@")[0] + "@" + text
            self.itemLabel.setText(new_text)
            self.contentList.remove(prev_text)
            self.contentList.append(new_text)
        except Exception as e:
            print(e)

    def clear(self, clearContentList=True):
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
