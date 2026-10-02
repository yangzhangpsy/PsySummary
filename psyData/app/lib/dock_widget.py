from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QDockWidget


class SizeContainerWidget(QWidget):
    """
    Helper widget for dock size hints.
    """

    def __init__(self, parent=None):
        super(SizeContainerWidget, self).__init__(parent)

    def setWidget(self, widget: QWidget):
        """
        Wrap a child widget with zero margins.
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
    Dock widget with a custom size hint.
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
