import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QMainWindow
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.optimize import minimize
from scipy.stats import norm


def pexGAUSbnd(x, mu=5, sigma=1, tau=1):
    """
    Compute the CDF of the ex-Gaussian distribution, ensuring values are within [0,1].
    """
    # x = np.clip(x, -1e10, 1e10)
    cdf_values = norm.cdf(x, mu, sigma) - np.exp((mu - x) / tau + (sigma ** 2) / (2 * tau ** 2)) * norm.cdf(
        (x - mu) / sigma - sigma / tau)
    cdf_values = np.clip(cdf_values, 0, 1)
    return cdf_values


def exG_lnlike(params, rts, rt_bounds=None):
    """
    Compute the log-likelihood of response times under an ex-Gaussian model.
    """
    mu, sigma, tau = np.abs(params)
    # pdf_values = (1 / tau) * np.exp((mu - rts) / tau + (sigma ** 2) / (2 * tau ** 2)) * norm.cdf(
    #     (rts - mu) / sigma - sigma / tau)

    # Use log-space calculations to prevent overflow
    log_pdf_values = np.log(1 / tau) + (mu - rts) / tau + (sigma ** 2) / (2 * tau ** 2) + \
                     norm.logcdf((rts - mu) / sigma - sigma / tau)
    pdf_values = np.exp(log_pdf_values)

    pdf_values = np.clip(pdf_values, 1e-10, np.inf)  # Avoid log(0)

    if rt_bounds is not None:
        lower_bound, upper_bound = rt_bounds
        if (np.min(rts) < lower_bound) or (np.max(rts) > upper_bound):
            raise ValueError("Likelihood cannot be computed if any RTs are outside the bounds")
        lost_prob = pexGAUSbnd(lower_bound, mu, sigma, tau) + (1 - pexGAUSbnd(upper_bound, mu, sigma, tau))
        pdf_values /= (1 - lost_prob)

    return -np.sum(np.log(pdf_values))


def exG_estimate_x(rts, start_prop_var_in_tau=None, rt_bounds=None, method="BFGS"):
    exG_estimated = exG_estimate(rts, start_prop_var_in_tau, rt_bounds, method)
    return exG_estimated['x']


def exG_estimate(rts, start_prop_var_in_tau=None, rt_bounds=None, method="BFGS"):
    """
    Estimate the parameters mu, sigma, and tau using maximum likelihood estimation.
    """
    if start_prop_var_in_tau is None:
        start_prop_var_in_tau = [0.2, 0.4, 0.6, 0.8]

    rts_mean = np.mean(rts)
    rts_variance = np.var(rts)
    best_result = None

    for prop in start_prop_var_in_tau:
        start_tau_variance = rts_variance * prop
        start_normal_variance = rts_variance - start_tau_variance
        start_tau = np.sqrt(start_tau_variance)
        start_sigma = np.sqrt(start_normal_variance)
        start_mu = rts_mean - start_tau
        start_params = [start_mu, start_sigma, start_tau]

        result = minimize(exG_lnlike, np.array(start_params), args=(rts, rt_bounds), method=method)
        # print(result['x'])
        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_result.start_prop_tau_variance = prop

    best_result.x = np.abs(best_result.x)  # Ensure parameters are positive
    return best_result


def CDFpooling_main(rt_df, sub_vars_list, cond_vars_list, rt_var_name,
                    min_trials_required=50,
                    start_prop_var_in_tau=None,
                    rt_bounds=None,
                    method="BFGS"):
    """
    Main function for CDF pooling analysis using multi-column composite grouping,
    without modifying original DataFrame columns.

    Parameters:
    - rt_df: DataFrame with RT data.
    - sub_vars_list: List of column names that define subject identity.
    - cond_vars_list: List of column names that define condition identity.
    - rt_var_name: Name of the RT column.
    - min_trials_required: Minimum number of trials required for estimation.
    - start_prop_var_in_tau: Initial values for optimization.
    - rt_bounds: Optional bounds on RT.
    - method: Optimization method.
    """

    if start_prop_var_in_tau is None:
        start_prop_var_in_tau = [0.1, 0.3, 0.5, 0.7]

    est_results = []

    rt_cdf_var_name = f"{rt_var_name}_cdf"

    rt_df[rt_cdf_var_name] = np.nan

    group_keys = sub_vars_list + cond_vars_list
    # unique_combinations = rt_df.loc[~rt_df['excluded'], group_keys].drop_duplicates()
    unique_combinations = rt_df.loc[:, group_keys].drop_duplicates()

    for _, combo in unique_combinations.iterrows():
        # 构建行筛选条件
        condition = np.ones(len(rt_df), dtype=bool)
        for col in group_keys:
            condition &= (rt_df[col] == combo[col])
        # condition &= ~rt_df['excluded']

        selected_rts = rt_df.loc[condition, rt_var_name].values

        if len(selected_rts) < min_trials_required:
            print(f"Failed estimation for {dict(combo)}, insufficient trials: {len(selected_rts)}")
            return None
        else:
            est_result = exG_estimate(selected_rts, start_prop_var_in_tau, rt_bounds, method)

            mu, sigma, tau = est_result.x

            rt_df.loc[condition, rt_cdf_var_name] = pexGAUSbnd(selected_rts, mu, sigma, tau)

    return rt_df


def CDFpooling_main_old(rt_df, min_trials_required=50, start_prop_var_in_tau=[0.1, 0.3, 0.5, 0.7], rt_bounds=None,
                    method="BFGS"):
    """
    Main function for CDF pooling analysis.
    """
    uniq_subs = rt_df['sub'].unique()
    uniq_conds = np.sort(rt_df['cond'].unique())
    est_results = []
    rt_df['cdf'] = np.nan

    if 'excluded' not in rt_df.columns:
        rt_df['excluded'] = False
        print("Analysis includes all trials because no variable 'excluded' is present.")

    for sub in uniq_subs:
        for cond in uniq_conds:
            selected_rows = (rt_df['sub'] == sub) & (rt_df['cond'] == cond) & (~rt_df['excluded'])
            selected_rts = rt_df.loc[selected_rows, 'rt'].values

            if len(selected_rts) < min_trials_required:
                print(
                    f"Skipping estimation for subject {sub}, condition {cond}, insufficient trials: {len(selected_rts)}")
                est_results.append({"sub": sub, "cond": cond, "estimated": False})
            else:
                est_result = exG_estimate(selected_rts, start_prop_var_in_tau, rt_bounds, method)
                est_result.estimated = True
                est_result.sub = sub
                est_result.cond = cond
                est_result.n_trials = len(selected_rts)

                mu, sigma, tau = est_result.x
                rt_df.loc[selected_rows, 'cdf'] = pexGAUSbnd(selected_rts, mu, sigma, tau)

                est_results.append({
                    "sub": sub, "cond": cond, "n_trials": len(selected_rts), "estimated": True,
                    "mu": mu, "sigma": sigma, "tau": tau, "value": est_result.fun
                })

    est_results_df = pd.DataFrame(est_results)
    return {"cdf": rt_df['cdf'], "est_results_df": est_results_df}


def generate_ex_gaussian_data(n=100, mu=100, sigma=100, nu=25):
    """生成ex-gaussian分布的随机数据"""
    gaussian_part = np.random.normal(mu, sigma, n)
    exponential_part = np.random.exponential(nu, n)
    return gaussian_part + exponential_part


# demo code:
def plot_exgaussian_fit(data, estimated_params):
    """Plot the histogram of the data and the fitted ex-Gaussian distribution."""
    mu, sigma, tau = estimated_params
    x = np.linspace(min(data), max(data), 100)
    pdf_fitted = (1 / tau) * np.exp((mu - x) / tau + (sigma ** 2) / (2 * tau ** 2)) * norm.cdf(
        (x - mu) / sigma - sigma / tau)

    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Histogram')
    plt.plot(x, pdf_fitted, 'r-', lw=2, label='Fitted Ex-Gaussian')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Ex-Gaussian Fit\nMu: {mu:.4f}, Sigma: {sigma:.4f}, Tau: {tau:.4f}')
    plt.legend()
    plt.show()


def exGaussianRunDemo():
    # rts = generate_ex_gaussian_data(1000, 900, 150, 300)
    # np.savetxt("rts_data.txt", rts, fmt="%.6f")
    rts = np.loadtxt('exGaussian_data.txt')
    # To read the data in R:
    # rts <- read.table("rts_data.txt", header=FALSE)[,1]
    fitResults = exG_estimate(rts)
    print(fitResults['x'] - [900, 150, 110])

    # Plot the fitted distribution
    plot_exgaussian_fit(rts, fitResults['x'])


# exGaussianRunDemo()
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Matplotlib in PyQt5 Example')

        # 创建一个MplCanvas实例
        self.canvas = MplCanvas()

        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        # 创建中心窗口小部件
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


class MplCanvas(FigureCanvas):
    def __init__(self):
        fig, self.ax = plt.subplots(figsize=(5, 4))
        super().__init__(fig)
        self.plot()

    def plot(self):
        # 这里可以放置你需要绘制的图形
        self.ax.plot([0, 1, 2, 3], [10, 20, 25, 30], label='Sample Line')
        self.ax.set_title('Sample Matplotlib Plot')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.legend()
