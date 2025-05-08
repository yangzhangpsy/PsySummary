from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QDockWidget


class SizeContainerWidget(QWidget):
    """
    we can set initial size of dock widget by using this.
    """

    def __init__(self, parent=None):
        super(SizeContainerWidget, self).__init__(parent)

    def setWidget(self, widget: QWidget):
        """
        设置包含的widget
        :param widget:
        :return:
        """
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        self.setLayout(layout)

    def sizeHint(self):
        return QSize(200, 10)


class DockWidget(QDockWidget):
    """
    my dock widget, it can resize and set initial size.
    """

    def __init__(self):
        super(DockWidget, self).__init__()
        self.size_container = SizeContainerWidget()
        super(DockWidget, self).setWidget(self.size_container)
        self.setFeatures(QDockWidget.NoDockWidgetFeatures | QDockWidget.DockWidgetClosable)

    def setWidget(self, QWidget):
        """
        we can't set initial size of dock widget,
        but we can achieve this goal by using a widget embed in dock widget.
        :param QWidget:
        :return:
        """
        self.size_container.setWidget(QWidget)
