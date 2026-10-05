"""Shared pulsing-dot animation for project loading and model fitting."""

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QPainter


DOT_COUNT = 9
FRAME_COUNT = 24
FRAME_INTERVAL_MS = 50


def elapsed_spinner_frame(elapsed_ms, interval_ms=FRAME_INTERVAL_MS, origin=0, direction=1):
    """Keep the animation phase tied to elapsed time rather than delivered timer ticks."""
    return (origin + direction * elapsed_ms / interval_ms) % FRAME_COUNT


def paint_dotted_spinner(painter, center, color, frame):
    """Draw fixed dots with a clockwise wave of shrinking size and opacity."""
    phase = (frame % FRAME_COUNT) / FRAME_COUNT
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.translate(QPointF(center))
    painter.setPen(Qt.NoPen)
    for index in range(DOT_COUNT):
        # Age since the leading edge passed this dot, expressed in revolutions.
        age = (phase - index / DOT_COUNT) % 1.0
        strength = 1.0 - age
        dot_color = QColor(color)
        dot_color.setAlpha(round(color.alpha() * strength))
        radius = 2.0 + 3.0 * strength
        painter.setBrush(dot_color)
        painter.drawEllipse(QPointF(0.0, -21.0), radius, radius)
        painter.rotate(360.0 / DOT_COUNT)
    painter.restore()
