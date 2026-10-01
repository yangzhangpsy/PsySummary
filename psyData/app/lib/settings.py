from PyQt5.QtCore import QSettings


class Settings(QSettings):
    def value(self, p_str, defaultValue=None, type=None):
        value = super(Settings, self).value(p_str, defaultValue)
        if value is None:
            value = defaultValue
        return value
