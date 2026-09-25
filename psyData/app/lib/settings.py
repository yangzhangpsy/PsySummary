"""QSettings wrapper with stable default-value behavior."""

from PyQt5.QtCore import QSettings


class Settings(QSettings):
    def value(self, key, defaultValue=None, type=None):
        value = super().value(key, defaultValue)
        return defaultValue if value is None else value
