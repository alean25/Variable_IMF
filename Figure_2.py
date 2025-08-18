import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
import seaborn as sns

# --- Setup Output Directory ---
output_dir = "Figures"
os.makedirs(output_dir, exist_ok=True)

# --- Load Jacobi Mass Function Data ---
try:
    mf_jacobi = pd.read_parquet("mass_functions_jacobi.parquet")
except FileNotFoundError:
    print("Error: 'mass_functions_jacobi.parquet' not found. Please ensure the file is in the correct directory.")
    exit()

clusters_df = pd.read_csv("clusters.csv")
mf_jacobi = pd.merge(mf_jacobi, clusters_df, on='id', how='left', suffixes=('_mf', '_cluster'))
fit_results = []

# --- IMF Function ---
def new_imf(masses, m_break, normal, low_slope, high_slope):
    imf = np.zeros_like(masses, dtype=float)
    mask_low = masses < m_break
    if m_break > 0:
        imf[mask_low] = normal * (masses[mask_low] / m_break) ** low_slope
    else:
        imf[mask_low] = normal * (masses[mask_low] ** low_slope)
    imf[~mask_low] = normal * (masses[~mask_low] / m_break) ** high_slope if m_break > 0 else normal * (masses[~mask_low] ** high_slope)
    return imf

# --- Individual Cluster Fit and Plot ---
def plot_and_fit_imf(cluster_id, data):
    cluster_data = data[data['id'] == cluster_id]
    if cluster_data.empty:
        print(f"No data found for cluster ID: {cluster_id}. Skipping.")
        return

    masses = cluster_data['bin_center'].values
    ydata = cluster_data['mass_corrected'].values
    yerr = cluster_data['mass_corrected_error'].values

    valid = (ydata > 0) & (yerr > 0) & np.isfinite(ydata) & np.isfinite(yerr) & np.isfinite(masses) & (masses > 0)
    masses, ydata, yerr = masses[valid], ydata[valid], yerr[valid]

    if len(masses) < 5:
        print(f"Too few valid data points ({len(masses)}) for cluster {cluster_id}. Skipping.")
        return

    try:
        bounds = ([0.05, 1e-6, -5.0, -5.0], [5.0, 1e6, 5.0, 5.0])
        p0 = [0.5, 1.0, -1.0, -2.0]
        popt, pcov = curve_fit(new_imf, masses, ydata, sigma=yerr, p0=p0,
                               bounds=bounds, maxfev=10000,
                               absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"Fit failed for cluster {cluster_id}: {e}. Skipping.")
        return

    m_break_fit, normal_fit, l_slope_fit, h_slope_fit = popt
    m_break_fit_err, normal_fit_err, l_slope_fit_err, h_slope_fit_err = perr

    y_model = new_imf(masses, *popt)
    cluster_name = cluster_data['cluster_name'].iloc[0] if 'cluster_name' in cluster_data.columns else np.nan

    fit_results.append({
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "m_break": m_break_fit,
        "m_break_err": m_break_fit_err,
        "masses": masses.tolist(),
        "ydata": ydata.tolist(),
        "yerr": yerr.tolist(),
        "popt": popt.tolist(),
    })

# --- Multi-panel Plot (2x2) ---
def plot_multiple_clusters_2x2(cluster_ids, data):
    assert len(cluster_ids) == 4, "Please provide exactly 4 cluster IDs."

    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'axes.edgecolor': '0.2'})
    plt.rcParams.update({
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "grid.alpha": 0.3
    })

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=False)
    axs = axs.flatten()

    for i, cluster_id in enumerate(cluster_ids):
        cluster_data_df = data[data['id'] == cluster_id]
        ax = axs[i]
        if cluster_data_df.empty:
            ax.set_title(f"ID {cluster_id}: No data", fontsize=10)
            continue

        current_fit_result = next((item for item in fit_results if item["cluster_id"] == cluster_id), None)
        if current_fit_result is None:
            ax.set_title(f"ID {cluster_id}: Fit data N/A", fontsize=10)
            continue

        masses = np.array(current_fit_result['masses'])
        ydata = np.array(current_fit_result['ydata'])
        yerr = np.array(current_fit_result['yerr'])
        popt = np.array(current_fit_result['popt'])
        m_break_fit = current_fit_result['m_break']
        m_break_fit_err = current_fit_result['m_break_err']
        cluster_name = current_fit_result['cluster_name']

        y_model = new_imf(masses, *popt)

        ax.errorbar(masses, ydata, yerr=yerr, fmt='o', markersize=5,
                    color='black', ecolor='gray', elinewidth=1, capsize=2,
                    label='Corrected MF')
        ax.plot(masses, y_model, color=sns.color_palette("Set1")[0], lw=2, label='Best-fit IMF')
        ax.axvline(x=m_break_fit, color='gray', linestyle=':', linewidth=2, label='Break Mass')  # Vertical break mass line
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"{cluster_name}", fontsize=17)
        ax.set_xlim([0.1, 5])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        textstr = f"$m_\\mathrm{{break}}$ = {m_break_fit:.2f} ± {m_break_fit_err:.2f}"
        ax.text(0.05, 0.30, textstr, transform=ax.transAxes,
                fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

        if i == 0:
            ax.legend(loc='lower left', fontsize=9, frameon=True, framealpha=0.8)

    fig.text(0.5, 0.03, 'Mass [M$_\odot$]', ha='center', fontsize=20)
    fig.text(0.02, 0.5, r'N(M) / ΔM', va='center', rotation='vertical', fontsize=20)

    plt.tight_layout(rect=[0.04, 0.05, 1, 0.96])
    output_path = os.path.join(output_dir, f'figure_2.png')
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

# --- Run Fitting and Plotting ---
clusters_to_plot = [4423, 4591, 14, 4571]

print("Performing individual fits and populating fit_results for required clusters...")
unique_clusters_for_fit = list(set(clusters_to_plot))
for ID in unique_clusters_for_fit:
    plot_and_fit_imf(ID, mf_jacobi)
print(f"Fit results collected for {len(fit_results)} clusters.")

plot_multiple_clusters_2x2(clusters_to_plot, mf_jacobi)
