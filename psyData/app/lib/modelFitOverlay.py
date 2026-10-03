"""Non-blocking visual overlay shown during PsySummary model fitting."""

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPalette
from PyQt5.QtWidgets import QWidget

from app.lib.dotted_spinner import FRAME_COUNT, FRAME_INTERVAL_MS, paint_dotted_spinner


class ModelFitOverlay(QWidget):
    """Block content interaction while painting a small animated fitting spinner."""

    OVERLAY_COLOR = (245, 245, 245, 64)
    SPINNER_COLOR = (70, 70, 70)
    DARK_THEME_SPINNER_COLOR = (255, 255, 255)
    MESSAGE_COLOR = (25, 25, 25)
    DETAIL_COLOR = (47, 111, 176)  # #2F6FB0
    DARK_THEME_DETAIL_COLOR = (138, 199, 255)  # #8AC7FF

    def __init__(self, parent=None):
        """Initialize a hidden overlay whose animation repaints only this widget."""
        super().__init__(parent)
        self._frame = 0
        self._message = 'Preparing model fitting…'
        self._detail = ''
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(FRAME_INTERVAL_MS)
        self.animation_timer.timeout.connect(self._advanceFrame)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.hide()

    def start(self, message='Preparing model fitting…', detail=''):
        """Show the blocking overlay and start its localized spinner animation."""
        self._message = message
        self._detail = detail
        self._frame = 0
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.setFocus(Qt.OtherFocusReason)
        self.animation_timer.start()
        self.update()

    def setProgress(self, current, total, label):
        """Display the current model queue position and model label."""
        self._message = f'Fitting model: {current} of {total}…'
        self._detail = str(label)
        self.update()

    def stop(self):
        """Stop the spinner and uncover the PsySummary interface."""
        self.animation_timer.stop()
        self.hide()
        self._frame = 0

    def syncGeometry(self):
        """Keep the overlay aligned with its parent after a window resize."""
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        if not self.isHidden():
            self.raise_()

    def _advanceFrame(self):
        """Advance one pulse frame and repaint only the spinner region."""
        self._frame = (self._frame + 1) % FRAME_COUNT
        self.update(self._spinnerUpdateRect())

    def _spinnerCenter(self):
        """Return the spinner center above the two centered status lines."""
        return QPointF(self.width() / 2.0, self.height() / 2.0 - 54.0)

    def _spinnerUpdateRect(self):
        """Return the small dirty region needed for one spinner frame."""
        center = self._spinnerCenter()
        return QRectF(center.x() - 32.0, center.y() - 32.0, 64.0, 64.0).toAlignedRect()

    def paintEvent(self, _event):
        """Paint a light translucent surface, spinner, and status message."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(*self.OVERLAY_COLOR))

        spinner_center = self._spinnerCenter()
        background = (self.parentWidget() or self).palette().color(QPalette.Window)
        spinner_color = QColor(*(
            self.DARK_THEME_SPINNER_COLOR if background.lightness() < 128
            else self.SPINNER_COLOR))
        paint_dotted_spinner(painter, spinner_center, spinner_color, self._frame)

        message_font = QFont(self.font())
        message_font.setBold(True)
        message_font.setPointSizeF(max(message_font.pointSizeF(), 12.0))
        painter.setFont(message_font)
        painter.setPen(QColor(*self.MESSAGE_COLOR))
        message_rect = QRectF(24.0, spinner_center.y() + 40.0,
                              max(0.0, self.width() - 48.0), 30.0)
        painter.drawText(message_rect, Qt.AlignCenter, self._message)

        detail_font = QFont(self.font())
        detail_font.setPointSizeF(max(detail_font.pointSizeF(), 9.0))
        painter.setFont(detail_font)
        painter.setPen(QColor(*(
            self.DARK_THEME_DETAIL_COLOR if background.lightness() < 128
            else self.DETAIL_COLOR)))
        detail_rect = QRectF(24.0, spinner_center.y() + 70.0,
                             max(0.0, self.width() - 48.0), 26.0)
        painter.drawText(
            detail_rect, Qt.AlignCenter | Qt.TextSingleLine,
            painter.fontMetrics().elidedText(
                self._detail, Qt.ElideMiddle, int(detail_rect.width())),
        )
