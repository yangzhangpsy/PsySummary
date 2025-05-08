from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QListWidget, QMenu, QAction


class ListWidget(QListWidget):
    listChanged: pyqtSignal = pyqtSignal(str)

    def __init__(self, contextMenuType: int = 1):
        super(ListWidget, self).__init__()

        self.context_menu_type = contextMenuType

        if self.context_menu_type > 0:
            self.setContextMenuPolicy(Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self.ShowContextMenu)

    def ShowContextMenu(self, pos):
        # 创建菜单
        context_menu = QMenu(self)

        if self.context_menu_type != 2:
            delete_action = context_menu.addAction("Delete")
            delete_action.triggered.connect(self.delete_item)

            delete_all_action = context_menu.addAction("Delete All")
            delete_all_action.triggered.connect(self.delete_all_item)

        if self.context_menu_type != 1:
            self.setSortingEnabled(True)
            # 递增排序
            sort_asc_action = QAction("Sort Ascending")
            sort_asc_action.triggered.connect(self.sort_ascending)
            context_menu.addAction(sort_asc_action)

            # 递减排序
            sort_desc_action = QAction("Sort Descending")
            sort_desc_action.triggered.connect(self.sort_descending)
            context_menu.addAction(sort_desc_action)
        # 显示菜单
        context_menu.exec_(self.mapToGlobal(pos))

    def delete_item(self):
        selectedItems = self.selectedItems()
        for item in selectedItems:
            self.removeItem(item)

        self.listChanged.emit("delete")

    def removeItem(self, item):
        self.takeItem(self.row(item))

    def delete_all_item(self):
        self.clear()

        self.listChanged.emit("delete")

    def sort_ascending(self):
        # 递增排序
        self.sortItems(Qt.AscendingOrder)

    def sort_descending(self):
        # 递减排序
        self.sortItems(Qt.DescendingOrder)
