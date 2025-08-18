import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from numpy.linalg import inv
from astropy.io import fits
from matplotlib.ticker import FuncFormatter

# === CONFIG ===
cluster_list = [4682]
output_dir = "Figures"
os.makedirs(output_dir, exist_ok=True)

# === LOAD (CSV + FITS) ===
mf_jacobi = pd.read_csv("FINAL_with_hunt_ages.csv")
with fits.open("asu_all.fit") as hdul:
    pd.DataFrame(hdul[1].data).to_csv("output.csv", index=False)
df_all = pd.read_csv("output.csv").dropna(subset=["Mass50"])

# === MODEL: broken power law with one break ===
def new_imf(masses, m_break, normal, low_slope, high_slope):
    imf = np.zeros_like(masses, dtype=float)
    mask_low = masses < m_break
    mask_high = ~mask_low
    imf[mask_low]  = normal * (masses[mask_low]  / m_break) ** low_slope
    imf[mask_high] = normal * (masses[mask_high] / m_break) ** high_slope
    return imf

# === FIT UTIL ===
def _cov_from_least_squares(result, n_params):
    try:
        J = result.jac
        dof = max(1, result.fun.size - n_params)
        s_sq = np.sum(result.fun**2) / dof
        JTJ_inv = inv(J.T @ J)
        return s_sq * JTJ_inv
    except Exception:
        return None

def fit_broken_powerlaw(x, y, yerr, color, title, eps=0.05):
    msk = (x > 0) & np.isfinite(x) & (y > 0) & np.isfinite(y) & (yerr > 0) & np.isfinite(yerr)
    x, y, yerr = x[msk], y[msk], yerr[msk]
    if x.size < 4:
        return None, None, f"{title}: not enough points"

    x_min, x_max = np.min(x), np.max(x)
    lower_break = x_min * (1.0 + eps)
    upper_break = x_max * (1.0 - eps)
    if upper_break <= lower_break:
        lower_break = x_min * 1.001
        upper_break = x_max * 0.999

    bounds = ([lower_break, 1e-10, -5.0, -5.0],
              [upper_break, 1e+6,  5.0,  5.0])

    geo = np.sqrt(x_min * x_max)
    m0s = [geo, np.percentile(x, 30), np.percentile(x, 70)]
    p0s = []
    for m0 in m0s:
        idx = np.argmin(np.abs(x - m0))
        norm0 = max(1e-10, y[idx])
        p0s.append([float(np.clip(m0, lower_break, upper_break)), norm0, -1.0, -2.0])
        p0s.append([float(np.clip(m0, lower_break, upper_break)), norm0, -0.8, -2.3])

    best = None
    best_popt, best_pcov = None, None
    for p0 in p0s:
        try:
            popt, pcov = curve_fit(
                new_imf, x, y, sigma=yerr,
                p0=p0, bounds=bounds, maxfev=20000, absolute_sigma=True
            )
            resid = (y - new_imf(x, *popt)) / yerr
            chi2 = float(np.sum(resid**2))
            if (best is None) or (chi2 < best):
                best, best_popt, best_pcov = chi2, popt, pcov
        except Exception:
            continue

    if best_popt is None:
        def res(theta):
            return (y - new_imf(x, *theta)) / yerr
        idx = np.argmin(np.abs(x - geo))
        norm0 = max(1e-10, y[idx])
        theta0 = np.array([np.clip(geo, lower_break, upper_break), norm0, -1.0, -2.0], dtype=float)
        try:
            lsq = least_squares(res, theta0, bounds=bounds, loss="soft_l1", f_scale=1.0, max_nfev=50000)
            best_popt = lsq.x
            best_pcov = _cov_from_least_squares(lsq, n_params=4)
        except Exception as e:
            return None, None, f"{title}: fit failed ({e})"

    return best_popt, best_pcov, None

def plot_with_fit(ax, x, y, yerr, color, label, title):
    popt, pcov, note = fit_broken_powerlaw(x, y, yerr, color, title)
    ax.errorbar(x, y, yerr=yerr, fmt='o', ms=9, capsize=5, elinewidth=2, lw=2,
                color=color, label=label)
    if popt is not None:
        xs = np.sort(x)
        ax.plot(xs, new_imf(xs, *popt), color=color, lw=3)
        m_break = popt[0]
        m_break_err = float(np.sqrt(pcov[0, 0])) if (pcov is not None and np.isfinite(pcov[0, 0])) else np.nan
        ax.axvline(m_break, color="gray", linestyle="--", alpha=0.7, lw=2)
        ax.text(0.05, 0.05,
                f"Break = {m_break:.2f}" + (f" ± {m_break_err:.2f}" if np.isfinite(m_break_err) else "") + " $M_{{\odot}}$",
                transform=ax.transAxes, fontsize=30, color="gray",
                ha="left", va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"))
    else:
        print(note or f"{title}: fit unavailable")

    # Log scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Integer ticks on log x-axis
    max_x = x.max() * 1.1
    ticks = np.arange(1, int(np.ceil(max_x)) + 1)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}" if v.is_integer() else ""))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.minorticks_on()
    ax.tick_params(axis="x", which="major", labelsize=18)
    ax.tick_params(axis="y", which="major", labelsize=18)


# === MAIN ===
def plot_4panel_imf(cluster_id):
    cluster_data = mf_jacobi[mf_jacobi["id"] == cluster_id]
    if cluster_data.empty:
        print(f"No data for cluster {cluster_id}")
        return
    cluster_name = cluster_data["name"].iloc[0]

    mf = pd.read_parquet("mass_functions_jacobi.parquet")
    sf = pd.read_parquet("selection_functions.parquet")
    mf = mf[mf["name"] == cluster_name].reset_index(drop=True)
    sf = sf[sf["name"] == cluster_name].reset_index(drop=True)

    sel_corr = mf["mass_raw"] / mf["selection_function"]
    frac_raw = mf["mass_raw_error"] / mf["mass_raw"]
    frac_sf  = mf["selection_function_error"] / mf["selection_function"]
    sel_corr_err = np.sqrt(frac_raw**2 + frac_sf**2) * sel_corr

    # 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    axes = axes.flatten()
    fig.suptitle(
        f"{cluster_name}",
        fontsize=40, fontweight="bold"
    )

    all_x, all_y = [], []

    panels = [
        (mf["bin_center"].values, mf["mass_raw"].values, mf["mass_raw_error"].values, "black", None),
        ( mf["bin_center"].values, sel_corr.values, sel_corr_err.values, "tab:orange", None),
        ( mf["bin_center"].values, mf["mass_corrected"].values, mf["mass_corrected_error"].values, "tab:green", None),
        ( 
         pd.read_parquet("NGC_6067_age_varied/mass_functions_jacobi.parquet")
           .loc[lambda df: df["log_age"]==8.6]
           .sort_values("bin_center")["bin_center"].values,
         pd.read_parquet("NGC_6067_age_varied/mass_functions_jacobi.parquet")
           .loc[lambda df: df["log_age"]==8.6]
           .sort_values("bin_center")["mass_corrected"].values,
         pd.read_parquet("NGC_6067_age_varied/mass_functions_jacobi.parquet")
           .loc[lambda df: df["log_age"]==8.6]
           .sort_values("bin_center")["mass_corrected_error"].values,
         "tab:blue", None)
    ]

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (ax, ( x, y, yerr, color, label)) in enumerate(zip(axes, panels)):
        all_x.append(x); all_y.append(y)
        plot_with_fit(ax, x, y, yerr, color, label, None)

        # Panel labels in top-left
        ax.text(0.02, 0.95, panel_labels[i], transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="top", ha="left")

        # Remove panel titles
        # Axis labels only on left column and bottom row
        if i % 2 == 1:  # right column
            ax.set_ylabel("")
        else:
            ax.set_ylabel("dN / dM", fontsize=40)

        if i < 2:  # top row
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Mass [$M_{\odot}$]", fontsize=40)

    # Unified axes limits
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    x_min, x_max = all_x.min()*0.9, all_x.max()*1.1
    y_min, y_max = all_y.min()*0.9, all_y.max()*1.1
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.savefig(os.path.join(output_dir, f"IMF_4panel_{cluster_id}.png"), dpi=300)
    plt.show()

# === RUN ===
for cid in cluster_list:
    plot_4panel_imf(cid)
