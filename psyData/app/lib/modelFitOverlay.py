"""Shared visual waiting overlay for PsySummary background operations."""

from PyQt5.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPalette
from PyQt5.QtWidgets import QWidget

from app.lib.dotted_spinner import FRAME_INTERVAL_MS, elapsed_spinner_frame, paint_dotted_spinner


class ModelFitOverlay(QWidget):
    """Block content interaction while painting a small animated waiting spinner."""

    OVERLAY_COLOR = (245, 245, 245, 64)
    SPINNER_COLOR = (70, 70, 70)
    DARK_THEME_SPINNER_COLOR = (255, 255, 255)
    MESSAGE_COLOR = (25, 25, 25)
    DETAIL_COLOR = (47, 111, 176)  # #2F6FB0
    DARK_THEME_DETAIL_COLOR = (138, 199, 255)  # #8AC7FF
    SHOW_DELAY_MS = 300

    def __init__(self, parent=None):
        """Initialize a hidden overlay whose animation repaints only this widget."""
        super().__init__(parent)
        self._frame = 0
        self._animation_clock = QElapsedTimer()
        self._message = 'Preparing model fitting…'
        self._detail = ''
        self._running = False
        self._feedback_visible = False
        self._reveal_timer = QTimer(self)
        self._reveal_timer.setSingleShot(True)
        self._reveal_timer.setTimerType(Qt.PreciseTimer)
        self._reveal_timer.timeout.connect(self._revealFeedback)
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(FRAME_INTERVAL_MS)
        self.animation_timer.timeout.connect(self._advanceFrame)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.hide()

    def start(self, message='Preparing model fitting…', detail=''):
        """Block immediately, revealing visual feedback only after 300 ms of waiting."""
        self._message = message
        self._detail = detail
        if not self._running:
            self._running = True
            self._feedback_visible = False
            self._frame = 0
            self._animation_clock.start()
            self._reveal_timer.start(self.SHOW_DELAY_MS)
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.setFocus(Qt.OtherFocusReason)
        self.update()

    def _revealFeedback(self):
        """Ignore stopped jobs and premature callbacks from a superseded delay."""
        if not self._running or self._feedback_visible:
            return
        remaining = self.SHOW_DELAY_MS - self._animation_clock.elapsed()
        if remaining > 0:
            self._reveal_timer.start(remaining)
            return
        self._feedback_visible = True
        self._frame = elapsed_spinner_frame(self._animation_clock.elapsed())
        self.animation_timer.start()
        self.update()

    def setProgress(self, current, total, label):
        """Display the current model queue position and model label."""
        self._message = f'Fitting model: {current} of {total}…'
        self._detail = str(label)
        self.update()

    def setStage(self, message):
        """Change the phase text while retaining the current model and condition detail."""
        self._message = str(message)
        self.update()

    def stop(self):
        """Cancel pending feedback and uncover immediately, without a minimum display time."""
        self._running = False
        self._feedback_visible = False
        self._reveal_timer.stop()
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
        if not self._running or not self._feedback_visible:
            return
        self._frame = elapsed_spinner_frame(self._animation_clock.elapsed())
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
        if not self._feedback_visible:
            return
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
