"""Fit time series of EdU-positive nucleoids with kinetic models.

Fits the measured time series of EdU-positive nucleoids with different kinetic
models and computes important model constants and their errors. Reproduces
values given in Supplementary Table 1 and Fig. 1f in the manuscript.

Time internally is in hours [0-120], displayed also in days.

Part of "The TFAM to mtDNA ratio defines inner-cellular nucleoid
populations with distinct activity levels"
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def draw_pseudo_data(y, e):
    """Draw random pseudo data for bootstrapping.

    Assumes measured data follows a normal distribution with the given
    mean and standard deviation.
    """
    return y + np.random.randn(len(e)) * e


def main():
    # Parameters
    np.random.seed(0)
    Nbootstrp = 400

    # Load experimental data
    m = np.loadtxt('data/edu_positive_nucleoids_timeseries.csv', delimiter=',')
    data_x = m[:, 0]
    data_y = m[:, 1]
    data_err = m[:, 2]

    x = np.arange(0, 121, dtype=float)  # fine grid for display (0-120 hours)
    x2 = np.arange(0, 30.01, 0.01) * 24  # 30 days for 95% positive EdU time

    ci_level = 0.05  # 95% confidence interval

    # Average effective growth rate
    d = np.log(2) / (7 * 24)  # division rate: 7 days
    print(f'division rate: {d * 24:.2f} days')

    # ========================================================================
    # Model A - One subpopulation (fit only first 18h)
    # ========================================================================
    print('Model A - one subpopulation')

    model_a = lambda p, t: 1 - np.exp(-2 * p[0] * t)
    ix = data_x <= 18

    def minimizer_a(p):
        return np.sum((model_a(p, data_x[ix]) - data_y[ix])**2)

    from scipy.optimize import minimize as _minimize
    res = _minimize(minimizer_a, [0.05], bounds=[(d, None)])
    p_a = res.x
    yA = model_a(p_a, x)

    # ========================================================================
    # Model B - Independency model
    # ========================================================================
    print('Model B - Independency model\n')

    # Parameters: gamma (fraction slow), alpha_s (slow rate), alpha_f (fast rate)
    model_b = lambda p, t: 1 - p[0] * np.exp(-2 * p[1] * t) - (1 - p[0]) * np.exp(-2 * p[2] * t)

    def minimizer_b(p):
        return np.sum((model_b(p, data_x) - data_y)**2)

    p0_b = [0.5, 0.005, 0.05]
    bounds_b = [(0, 1), (d, None), (d, None)]
    res = minimize(minimizer_b, p0_b, bounds=bounds_b)
    p_b = res.x
    yB = model_b(p_b, x)

    # Bootstrapping
    ex_b = np.zeros((Nbootstrp, len(p_b)))
    ey_b = np.zeros((Nbootstrp, len(data_y)))
    ey2_b = np.zeros((Nbootstrp, len(x2)))

    for i in range(Nbootstrp):
        ye = draw_pseudo_data(data_y, data_err)
        minimizer_i = lambda p: np.sum((model_b(p, data_x) - ye)**2)
        res_i = minimize(minimizer_i, p0_b, bounds=bounds_b)
        ex_b[i, :] = res_i.x
        ey_b[i, :] = model_b(res_i.x, data_x)
        ey2_b[i, :] = model_b(res_i.x, x2)

    ci_b = np.quantile(ex_b, [ci_level / 2, 1 - ci_level / 2], axis=0)
    ytt_b = np.quantile(ey2_b, [ci_level / 2, 0.5, 1 - ci_level / 2], axis=0)

    print('Fast replication (active) population')
    print(f' Size {(1 - p_b[0]) * 100:.0f}% ({(1 - ci_b[1, 0]) * 100:.0f}% - {(1 - ci_b[0, 0]) * 100:.0f}%)')
    print(f' Replication rate (per nucleoid, per day) {p_b[2] * 24:.2f} ({ci_b[0, 2] * 24:.2f} - {ci_b[1, 2] * 24:.2f})')
    print(f' Degradation rate (per nucleoid, per day) {(p_b[2] - d) * 24:.2f} ({(ci_b[0, 2] - d) * 24:.2f} - {(ci_b[1, 2] - d) * 24:.2f})')

    print('\nSlow replication (inactive) population')
    print(f' Size {p_b[0] * 100:.0f}% ({ci_b[0, 0] * 100:.0f}% - {ci_b[1, 0] * 100:.0f}%)')
    print(f' Replication rate (per nucleoid, per day) {p_b[1] * 24:.2f} ({ci_b[0, 1] * 24:.2f} - {ci_b[1, 1] * 24:.2f})')
    print(f' Degradation rate (per nucleoid, per day) {(p_b[1] - d) * 24:.2f} ({(ci_b[0, 1] - d) * 24:.2f} - {(ci_b[1, 1] - d) * 24:.2f})')
    print()

    # Time until 50%/90% EdU positive
    for ti in [0.5, 0.9]:
        ti1 = np.log(1 - ti) / (-2 * p_b[2] * 24)
        ti2 = np.log(1 - ti) / (-2 * ci_b[1, 2] * 24)
        ti3 = np.log(1 - ti) / (-2 * ci_b[0, 2] * 24)
        print(f'Time until {ti * 100:.1f}% of the fast replicating (active) population is EdU-labeled: {ti1:.2f} days ({ti2:.2f} - {ti3:.2f} days)')

        ti1 = np.log(1 - ti) / (-2 * p_b[1] * 24)
        ti2 = np.log(1 - ti) / (-2 * ci_b[1, 1] * 24)
        ti3 = np.log(1 - ti) / (-2 * ci_b[0, 1] * 24)
        print(f'Time until {ti * 100:.1f}% of the slow replicating (inactive) population is EdU-labeled: {ti1:.2f} days ({ti2:.2f} - {ti3:.2f} days)')

        idx1 = np.argmin(np.abs(ytt_b[1, :] - ti)); t1 = x2[idx1] / 24
        idx2 = np.argmin(np.abs(ytt_b[2, :] - ti)); t2 = x2[idx2] / 24
        idx3 = np.argmin(np.abs(ytt_b[0, :] - ti)); t3 = x2[idx3] / 24
        print(f'Time until {ti * 100:.1f}% of all nucleoids are EdU-labeled: {t1:.2f} days  ({t2:.2f} - {t3:.2f} days)')

    # ========================================================================
    # Model C - Inactivation model
    # ========================================================================
    print('\n\nModel C - Inactivation model\n')

    # Two free parameters: Delta_f, Alpha_f
    def model_c(p, t):
        return 1 - ((p[0] - d) * 2 * p[1] * np.exp(-d * t) +
                     d * (2 * p[1] - p[0]) * np.exp(-2 * p[1] * t)) / \
                    (p[0] * (2 * p[1] - d))

    def minimizer_c(p):
        return np.sum((model_c(p, data_x) - data_y)**2)

    p0_c = [0.005, 0.03]
    bounds_c = [(d, None), (d, None)]
    res = minimize(minimizer_c, p0_c, bounds=bounds_c)
    p_c = res.x
    yC = model_c(p_c, x)

    # Derived parameter values
    alpha_f = p_c[1]
    beta_f = p_c[1] - p_c[0]
    tau = p_c[0] - d  # transfer rate fast to slow
    gamma = tau / p_c[0]  # fast growing fraction

    # Model for slow replicating population only
    def model_sp(xp, t):
        af, bf, tau_p, gam = xp
        return 1 - ((gam * (af + bf) + tau_p) * np.exp(-d * t) -
                     tau_p * (1 - gam) * np.exp(-(af + bf + tau_p + d) * t)) / \
                    gam / (af + bf + tau_p)

    # Bootstrapping
    ex_c = np.zeros((Nbootstrp, len(p_c)))
    exf_c = np.zeros((Nbootstrp, 4))
    ey_c = np.zeros((Nbootstrp, len(data_y)))
    ey2_c = np.zeros((Nbootstrp, len(x2)))
    eeysp_c = np.zeros((Nbootstrp, len(x2)))

    for i in range(Nbootstrp):
        ye = draw_pseudo_data(data_y, data_err)
        minimizer_i = lambda p: np.sum((model_c(p, data_x) - ye)**2)
        res_i = minimize(minimizer_i, p0_c, bounds=bounds_c)
        pi = res_i.x
        ex_c[i, :] = pi
        xif = [pi[1], pi[1] - pi[0], pi[0] - d, (pi[0] - d) / pi[0]]
        exf_c[i, :] = xif
        ey_c[i, :] = model_c(pi, data_x)
        ey2_c[i, :] = model_c(pi, x2)
        eeysp_c[i, :] = model_sp(xif, x2)

    cif = np.quantile(exf_c, [ci_level / 2, 1 - ci_level / 2], axis=0)
    ytt_c = np.quantile(ey2_c, [ci_level / 2, 0.5, 1 - ci_level / 2], axis=0)
    yttsp = np.quantile(eeysp_c, [ci_level / 2, 0.5, 1 - ci_level / 2], axis=0)

    print('Fast replication (active) population')
    print(f' Size {gamma * 100:.0f}% ({cif[0, 3] * 100:.0f}% - {cif[1, 3] * 100:.0f}%)')
    print(f' Replication rate (per nucleoid, per day) {alpha_f * 24:.2f} ({cif[0, 0] * 24:.2f} - {cif[1, 0] * 24:.2f})')
    print(f' Degradation rate (per nucleoid, per day) {beta_f * 24:.2f} ({cif[0, 1] * 24:.2f} - {cif[1, 1] * 24:.2f})')

    print('\nSlow replication (inactive) population')
    print(f' Size {(1 - gamma) * 100:.0f}% ({(1 - cif[1, 3]) * 100:.0f}% - {(1 - cif[0, 3]) * 100:.0f}%)')

    print('\nTransfer from fast to slow')
    print(f' (per nucleoid per day) {tau * 24:.2f} ({cif[0, 2] * 24:.2f} - {cif[1, 2] * 24:.2f})')
    print()

    # Time until 50%/90% labeled
    for ti in [0.5, 0.9]:
        ti1 = np.log(1 - ti) / (-2 * alpha_f * 24)
        ti2 = np.log(1 - ti) / (-2 * cif[1, 0] * 24)
        ti3 = np.log(1 - ti) / (-2 * cif[0, 0] * 24)
        print(f'Time until {ti * 100:.1f}% of the fast replicating (active) population is EdU-labeled: {ti1:.2f} days ({ti2:.2f} - {ti3:.2f} days)')

        idx1 = np.argmin(np.abs(yttsp[1, :] - ti)); t1 = x2[idx1] / 24
        idx2 = np.argmin(np.abs(yttsp[2, :] - ti)); t2 = x2[idx2] / 24
        idx3 = np.argmin(np.abs(yttsp[0, :] - ti)); t3 = x2[idx3] / 24
        print(f'Time until {ti * 100:.1f}% of the slow replicating (inactive) population is EdU-labeled: {t1:.2f} days ({t2:.2f} - {t3:.2f} days)')

        idx1 = np.argmin(np.abs(ytt_c[1, :] - ti)); t1 = x2[idx1] / 24
        idx2 = np.argmin(np.abs(ytt_c[2, :] - ti)); t2 = x2[idx2] / 24
        idx3 = np.argmin(np.abs(ytt_c[0, :] - ti)); t3 = x2[idx3] / 24
        print(f'Time until {ti * 100:.1f}% of all nucleoids are EdU-labeled: {t1:.2f} days  ({t2:.2f} - {t3:.2f} days)')

    # ========================================================================
    # Recreate Figure 1f from the manuscript
    # ========================================================================
    lw = 1.5
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, yA * 100, 'k--', linewidth=lw, label='Model A')
    ax.plot(x, yB * 100, 'r-', linewidth=lw, label='Model B')
    ax.plot(x, yC * 100, 'b-', linewidth=lw, label='Model C')
    ax.errorbar(data_x, data_y * 100, yerr=data_err * 100, fmt='ko', label='Data')
    ax.set_xlabel('EdU-Incubation [h]')
    ax.set_xlim([0, 120])
    ax.set_xticks(range(0, 121, 20))
    ax.set_ylabel('EdU-positive nucleoids (%)')
    ax.set_ylim([0, 100])
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True)
    ax.legend()
    ax.set_title('reproduction of Fig. 1f')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
