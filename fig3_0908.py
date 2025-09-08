import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

# ===============================
# Setup
# ===============================
OUTPUT_DIR = "Final_Figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "Filtered_08_18_with_slope_errs.csv"

# ===============================
# Helpers
# ===============================
def ensure_columns(df: pd.DataFrame, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def impose_min_age_error(df):
    """Replace zero age errors with a typical small nonzero value."""
    for col in ["age_err_low", "age_err_high"]:
        nonzero = df.loc[df[col] > 0, col]
        if not nonzero.empty:
            floor = np.median(nonzero.nsmallest(10))
            df.loc[df[col] <= 0, col] = floor
    return df

def rolling_stats_fixed_width(x, y, width=0.75, min_points=20, step=0.025):
    """
    Compute rolling median + IQR of y as a function of x.
    Returns also index ranges for robust and edge regions.
    """
    order = np.argsort(x)
    x, y = x[order], y[order]

    x_mid, y_median, y_low, y_high, npts = [], [], [], [], []
    x_min, x_max = x.min(), x.max()

    centers = np.arange(x_min, x_max, step)
    for xc in centers:
        mask = (x >= xc - width/2) & (x < xc + width/2)
        if mask.sum() > 0:
            x_mid.append(np.median(x[mask]))
            y_median.append(np.median(y[mask]))
            y_low.append(np.percentile(y[mask], 25))
            y_high.append(np.percentile(y[mask], 75))
            npts.append(mask.sum())

    x_mid, y_median, y_low, y_high, npts = map(np.array, [x_mid, y_median, y_low, y_high, npts])

    robust_mask = npts >= min_points
    return x_mid, y_median, y_low, y_high, robust_mask

# ===============================
# Plotting functions
# ===============================
def plot_running_median(ax, x_med, y_med, y_low, y_high, robust_mask):
    """Plot solid median inside robust region, dashed outside."""
    if robust_mask.any():
        i_first = np.argmax(robust_mask)
        i_last = len(robust_mask) - np.argmax(robust_mask[::-1]) - 1

        # Solid central portion
        ax.plot(x_med[i_first:i_last+1], y_med[i_first:i_last+1],
                color="darkred", lw=2.5, label="Running median")

        # Dashed extensions
        if i_first > 0:
            ax.plot(x_med[:i_first+1], y_med[:i_first+1],
                    color="darkred", lw=2.5, ls="--")
        if i_last < len(x_med)-1:
            ax.plot(x_med[i_last:], y_med[i_last:],
                    color="darkred", lw=2.5, ls="--")

    ax.fill_between(x_med, y_low, y_high, color="darkred", alpha=0.2, label="IQR (25–75%)")

def plot_break_mass_vs_age(df, ax=None, show=True):
    required = ["age", "age_err_low", "age_err_high",
                "break_mass", "break_mass_err"]
    ensure_columns(df, required)
    dfv = df[required].dropna()
    if dfv.empty:
        print("No valid rows for break_mass vs age; skipping.")
        return

    r_break, p_break = pearsonr(dfv["age"], dfv["break_mass"])
    rho_break, rho_pval = spearmanr(dfv["age"], dfv["break_mass"])

    x_med, y_med, y_low, y_high, robust_mask = rolling_stats_fixed_width(
        dfv["age"].to_numpy(),
        dfv["break_mass"].to_numpy(),
        width=0.75, min_points=20, step=0.025
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    xerr = [dfv["age_err_low"].to_numpy(), dfv["age_err_high"].to_numpy()]
    yerr = dfv["break_mass_err"].to_numpy()

    ax.errorbar(
        dfv["age"], dfv["break_mass"],
        xerr=xerr, yerr=yerr,
        fmt='o', color='steelblue',
        ecolor='gray', elinewidth=1, capsize=2,
        alpha=0.7, markeredgecolor='k'
    )

    plot_running_median(ax, x_med, y_med, y_low, y_high, robust_mask)

    ax.axhline(0.5, color="black", lw=1.5, ls=":", label=r"Kroupa ($0.5\,M_\odot$)")
    ax.set_xlabel(r'$\log_{10}$ Age [yr]', fontsize=12)
    ax.set_ylabel(r'Break Mass [M$_\odot$]', fontsize=12)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title("Break Mass Increases for Older Clusters", fontsize=16, weight="bold")

    caption_text = f"Pearson r = {r_break:.2f} (p={p_break:.1e})"
    ax.text(
        0.98, 0.02, caption_text, transform=ax.transAxes,
        fontsize=10, ha="right", va="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3")
    )
    ax.legend()

    if ax is None or show:
        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, "break_mass_vs_age.png")
        plt.savefig(outpath, dpi=300)
        if show:
            plt.show()
        print("\nBreak Mass vs Age")
        print(f"Saved plot to: {outpath}")
        print(f"Pearson r = {r_break:.3f}, P-value = {p_break:.3e}")
        print(f"Spearman rho = {rho_break:.3f}, P-value = {rho_pval:.3e}")

def analyze_slope_vs_age(df, y_col, y_label, yerr_col, outfile=None, y_limits=None, ax=None, show=True):
    required = ["age", "age_err_low", "age_err_high", y_col, yerr_col]
    ensure_columns(df, required)
    dfv = df[required].dropna()
    if dfv.empty:
        print(f"No valid rows for {y_col} vs age; skipping.")
        return

    r, p = pearsonr(df["age"].dropna(), df[y_col].dropna())
    rho, rho_p = spearmanr(df["age"].dropna(), df[y_col].dropna())

    x_med, y_med, y_low, y_high, robust_mask = rolling_stats_fixed_width(
        dfv["age"].to_numpy(),
        dfv[y_col].to_numpy(),
        width=0.75, min_points=20, step=0.025
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    xerr = [dfv["age_err_low"].to_numpy(), dfv["age_err_high"].to_numpy()]
    yerr = dfv[yerr_col].to_numpy()

    ax.errorbar(
        dfv["age"], dfv[y_col],
        xerr=xerr, yerr=yerr,
        fmt='o', color='steelblue',
        ecolor='gray', elinewidth=1, capsize=2,
        alpha=0.7, markeredgecolor='k'
    )

    plot_running_median(ax, x_med, y_med, y_low, y_high, robust_mask)

    ax.set_xlabel(r'$\log_{10}$ Age [yr]', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_title(f"Correlation of {y_label} and Age", fontsize=16, weight="bold")
    caption_text = f"Pearson r = {r:.2f} (p={p:.1e})"
    ax.text(
        0.98, 0.02, caption_text, transform=ax.transAxes,
        fontsize=10, ha="right", va="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3")
    )
    ax.legend()

    if ax is None or show:
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile, dpi=300)
        if show:
            plt.show()
        print(f"\n{y_col} vs Age")
        if outfile:
            print(f"Saved plot to: {outfile}")
        print(f"Pearson r = {r:.3f}, P-value = {p:.3e}")
        print(f"Spearman rho = {rho:.3f}, P-value = {rho_p:.3e}")

# ------------ NEW: scatter-only helper for mini-panels ------------
def _scatter_only_zoom(df, y_col, y_label, yerr_col, ax):
    """
    Scatter + error bars only (no running median) and zoom y using point values only.
    Used exclusively by the three-panel figure mini-panels.
    """
    required = ["age", "age_err_low", "age_err_high", y_col, yerr_col]
    ensure_columns(df, required)
    dfv = df[required].dropna()
    if dfv.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    x = dfv["age"].to_numpy()
    y = dfv[y_col].to_numpy()
    xerr = [dfv["age_err_low"].to_numpy(), dfv["age_err_high"].to_numpy()]
    yerr = dfv[yerr_col].to_numpy()

    ax.errorbar(
        x, y,
        xerr=xerr, yerr=yerr,
        fmt='o', color='steelblue',
        ecolor='gray', elinewidth=1, capsize=2,
        alpha=0.7, markeredgecolor='k'
    )

    ax.set_xlabel(r'$\log_{10}$ Age [yr]', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)

    # Zoom using point values only (ignore error bars)
    ymin, ymax = np.min(y), np.max(y)
    if np.isfinite(ymin) and np.isfinite(ymax):
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(y_label, fontsize=12, weight="bold")

# ===============================
# Three-panel figure
# ===============================
def plot_three_panel(df):
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios=[1,1], figure=fig)

    ax_big = fig.add_subplot(gs[:, 0])   # big left panel
    ax_top = fig.add_subplot(gs[0, 1])   # small top-right
    ax_bot = fig.add_subplot(gs[1, 1])   # small bottom-right

    # Main panel: unchanged (includes running median + dashed edges + IQR)
    plot_break_mass_vs_age(df, ax=ax_big, show=False)

    # Mini-panels: scatter-only + zoomed y (no medians)
    _scatter_only_zoom(df, "high_slope", r"High-mass Slope", "high_slope_err", ax=ax_top)
    _scatter_only_zoom(df, "low_slope",  r"Intermediate-mass Slope",  "low_slope_err",  ax=ax_bot)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "three_panel.png")
    plt.savefig(outpath, dpi=300)
    #plt.show()
    print(f"Three-panel figure saved to {outpath}")

# ===============================
# Main
# ===============================
def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if "id" in df.columns:
        before = len(df)
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df.drop_duplicates(subset="id", keep="first")
        after = len(df)
        print(f"Dropped {before - after} duplicate rows based on 'id'")
    else:
        print("⚠️ Warning: 'id' column not found, skipping duplicate removal.")
    print(f"Dataset after dropping duplicates: {len(df)} clusters")

    df = impose_min_age_error(df)

    # Single-panel figures (unchanged)
    plot_break_mass_vs_age(df)
    analyze_slope_vs_age(df, "high_slope", r"High-mass Slope",
                         "high_slope_err",
                         outfile=os.path.join(OUTPUT_DIR, "high_slope_vs_age.png"))
    analyze_slope_vs_age(df, "low_slope", r"Low-mass Slope",
                         "low_slope_err",
                         outfile=os.path.join(OUTPUT_DIR, "low_slope_vs_age.png"))

    # Three-panel figure (mini-panels: scatter-only + zoom)
    plot_three_panel(df)

if __name__ == "__main__":
    main()
