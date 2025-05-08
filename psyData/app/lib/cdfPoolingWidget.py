import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QDialog, QTextEdit, QFrame
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from app.psyDataFunc import PsyDataFunc


def outlier_mode_lnlike(params, cdf_values):
    po, omega = params

    if not (0 <= po <= 1 and 0 <= omega < 1):
        return np.inf

    valid_density = (1 - po)

    outlier_density = np.zeros_like(cdf_values)
    mask = cdf_values >= omega
    outlier_range = 1 - omega
    if outlier_range == 0:
        return np.inf

    outlier_density[mask] = 2 * (cdf_values[mask] - omega) / (outlier_range ** 2)

    mixture_density = valid_density + po * outlier_density
    mixture_density = np.maximum(mixture_density, 1e-12)

    return -np.sum(np.log(mixture_density))


def fit_outlier_model(cdf_values):
    init_params = [0.01, 0.98]
    bounds = [(0.0001, 0.2), (0.7, 0.999)]

    result = minimize(outlier_mode_lnlike, np.array(init_params), args=(cdf_values,), bounds=bounds)
    if result.success:
        return result.x
    else:
        return [None, None]
        # raise RuntimeError("Model fitting failed")


def generate_simulated_cdf_data(n=5000, true_po=0.03, omega=0.98):
    n_outliers = int(n * true_po)
    n_valid = n - n_outliers

    valid_cdfs = np.random.uniform(0, 1, n_valid)
    outlier_cdfs = 1 - np.sqrt(np.random.uniform(0, 1, n_outliers) * (1 - omega) ** 2)

    return np.concatenate([valid_cdfs, outlier_cdfs])


class CdfPoolingWidget(QDialog):
    def __init__(self, cdf_data, po_hat, omega_hat):
        super().__init__()
        self.po_hat = po_hat
        self.omega_hat = omega_hat
        self.original_omega = omega_hat
        self.setWindowTitle('CDF Pooling Model Fit')
        self.setWindowIcon(PsyDataFunc.getImageObject("icon.png", type=1))
        self.setWindowModality(Qt.ApplicationModal)

        self.canvas = MplCanvas(cdf_data, po_hat, omega_hat)

        self.warning_Info = QTextEdit()
        self.warning_Info.setReadOnly(True)
        self.warning_Info.setTextInteractionFlags(Qt.NoTextInteraction)
        self.warning_Info.setFrameShape(QFrame.NoFrame)
        self.warning_Info.setObjectName("Warning Info")
        self.warning_Info.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.warning_Info.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.warning_Info.setFixedHeight(60)
        font = QFont("Arial", 12, QFont.Normal)
        self.warning_Info.setFont(font)
        self.warning_Info.setStyleSheet("QTextEdit { background: transparent; border: none; }")

        self.warning_Info.setHtml("<div style='text-align: center;'><p style='line-height: 1.5;'>Carefully examine "
                                  "the value of ω. If it seems unreasonable or the dataset contains fewer<br>than "
                                  "5000 points, select \"Abort CDF Pooling\" to skip this step.</p></div>")

        content_height = self.warning_Info.fontMetrics().height() * 2
        self.warning_Info.setFixedHeight(int(content_height * 1.5))

        self.abort_button = QPushButton('Abort CDF Pooling')
        self.cancel_button = QPushButton('Estimated ω')
        self.ok_button = QPushButton('OK')

        self.abort_button.clicked.connect(self.abort_button_event)
        self.cancel_button.clicked.connect(self.reset_omega)
        self.ok_button.clicked.connect(self.accept_omega)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.abort_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)

        layout = QVBoxLayout()
        layout.addWidget(self.warning_Info)
        layout.addWidget(self.canvas)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def abort_button_event(self):
        self.omega_hat = -1
        self.close()

    def reset_omega(self):
        self.canvas.omega_hat = self.original_omega
        self.canvas.plot(self.canvas.cdf_data, self.canvas.po_hat, self.original_omega)

    def accept_omega(self):
        self.omega_hat = self.canvas.omega_hat
        self.close()
        # print(f"Accepted omega: {accepted_omega:.4f}")


class MplCanvas(FigureCanvas):
    def __init__(self, cdf_values, po_value, omega_value):
        self.text_annotation = None
        self.cross_hair_h = None
        self.cross_hair_v = None
        fig, self.ax = plt.subplots(figsize=(6, 4))
        super().__init__(fig)
        self.cdf_data = cdf_values
        self.po_hat = po_value
        self.omega_hat = omega_value
        self.plot(cdf_values, po_value, omega_value)
        self.mouse_click = self.mpl_connect('button_press_event', self.onclick)
        self.mouse_move = self.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.plot_cross_hair()

    def plot_cross_hair(self):
        self.cross_hair_v, = self.ax.plot([0, 0], [0, 1], transform=self.ax.get_xaxis_transform(), color='black', lw=0.8,
                                          ls='-')
        self.cross_hair_h, = self.ax.plot([0, 1], [0, 0], transform=self.ax.get_yaxis_transform(), color='black', lw=0.8,
                                          ls='-')
        self.text_annotation = self.ax.text(self.omega_hat, 1-self.po_hat, '', transform=self.ax.transAxes,
                                            horizontalalignment='right', verticalalignment='top', fontsize=10)

    def plot(self, cdf_data, po_hat, omega_hat):
        self.ax.clear()
        self.plot_cross_hair()
        self.ax.hist(cdf_data, bins=100, density=True, alpha=0.7, label=f"Empirical CDFs ({len(cdf_data)})")
        self.ax.axvline(omega_hat, color="red", linestyle=":", linewidth=2, label=f"Estimated ω = {omega_hat:.4f}")

        x = np.linspace(0, 1, 1000)
        fitted_density = np.full_like(x, 1 - po_hat)
        mask_fit = x >= omega_hat
        fitted_density[mask_fit] += po_hat * (2 * (x[mask_fit] - omega_hat) / ((1 - omega_hat) ** 2))

        self.ax.plot(x, fitted_density, color='blue', linestyle="--", linewidth=2, label='Fitted Mixture Density')

        self.ax.set_xlabel("CDF value")
        self.ax.set_ylabel("Density")
        self.ax.set_title("To adjust ω, click the left mouse button", fontsize=10)
        # self.ax.legend()
        self.ax.legend(loc='upper left')
        self.draw()

    def onclick(self, event):
        if event.inaxes == self.ax:
            self.omega_hat = event.xdata
            self.plot(self.cdf_data, self.po_hat, self.omega_hat)

    def on_mouse_move(self, event):
        if event.inaxes == self.ax and 0 <= event.xdata <= 1:
            # 鼠标在范围内时显示十字线
            self.cross_hair_v.set_visible(True)
            self.cross_hair_h.set_visible(True)
            self.text_annotation.set_visible(True)

            self.cross_hair_v.set_xdata([event.xdata, event.xdata])
            self.cross_hair_h.set_ydata([event.ydata, event.ydata])
            # 使用axes坐标 (相对坐标系)，避免注释超出图像范围
            inv = self.ax.transAxes.inverted()
            ax_coord = inv.transform(self.ax.transData.transform((event.xdata, np.maximum(event.ydata, 1 - self.po_hat))))
            ax_x, ax_y = ax_coord

            # 限制注释在图中显示完整
            ax_x = np.clip(ax_x + 0.02, 0.01, 0.85)
            ax_y = np.clip(ax_y + 0.02, 0.01, 0.95)

            self.text_annotation.set_position((ax_x, ax_y+0.1))

            # print(f"{event.xdata} {event.ydata}")
            self.text_annotation.set_text(f"ω: {event.xdata:.4f}")

        else:
            # 鼠标超出范围隐藏十字线
            self.cross_hair_v.set_visible(False)
            self.cross_hair_h.set_visible(False)
            self.text_annotation.set_visible(False)
        self.draw_idle()


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys

    np.random.seed(42)
    cdf_data = generate_simulated_cdf_data()
    po_hat, omega_hat = fit_outlier_model(cdf_data)

    app = QApplication(sys.argv)
    main_window = CdfPoolingWidget(cdf_data, po_hat, omega_hat)
    main_window.show()
    sys.exit(app.exec_())
