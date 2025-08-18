import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os


output_dir = "Figures"
os.makedirs(output_dir, exist_ok=True)
output_path = output_path = os.path.join(output_dir, "figure_3")
# ===============================
# Load data
# ===============================
df = pd.read_csv("FILTERED_FINAL_LIST.csv")
df.columns = df.columns.str.strip()
print(f"Original dataset: {len(df)} clusters")

# ===============================
# Compute asymmetric age errors
# ===============================
df['age_err_low'] = np.where(
    df['age_16'].notna(),
    df['age'] - df['age_16'],
    np.nan
)
df['age_err_high'] = np.where(
    df['age_84'].notna(),
    df['age_84'] - df['age'],
    np.nan
)

# ===============================
# Fill missing age errors using well-measured clusters
# ===============================
def fill_missing_age_errors_fixed(df, err_low_col='age_err_low', err_high_col='age_err_high', 
                                  max_typical_error=0.05):
    """
    Fill missing age errors by using typical small errors from well-measured clusters.
    """
    df = df.copy()
    
    # Select well-measured clusters
    well_measured = df[
        df[err_low_col].notna() & df[err_high_col].notna() &
        (df[err_low_col] <= max_typical_error) &
        (df[err_high_col] <= max_typical_error)
    ]
    
    # Take representative typical errors
    typical_low = well_measured[err_low_col].median()
    typical_high = well_measured[err_high_col].median()
    
    # Fill missing or zero errors
    df[err_low_col] = df[err_low_col].fillna(typical_low)
    df[err_low_col] = df[err_low_col].replace(0, typical_low)
    
    df[err_high_col] = df[err_high_col].fillna(typical_high)
    df[err_high_col] = df[err_high_col].replace(0, typical_high)
    
    return df

df = fill_missing_age_errors_fixed(df, max_typical_error=0.05)

# ===============================
# Apply filters
# ===============================
age_error = 0.3

filtered_df = df[
    (df['kind'].isin(['o', 'm'])) &
    (df['n_bins_before_break'] > 2) &
    (df['n_bins_after_break'] > 2) &
    (df['cst'] > 15) &
    (df['class_50'] > 0.75) &
    (df['a_v_50'] < 1.0) &
    (df['high_slope'] > -4.0) &
    (df['low_slope'] < 4.0) &
    (df['break_mass_err'] < 0.15) &
    (df['age_err_low'] < age_error) &
    (df['age_err_high'] < age_error)
]
print(f"Filtered dataset: {len(filtered_df)} clusters")

filtered_df.to_csv("FILTERED_FINAL_LIST.csv", index=False)


# ===============================
# Helpers for correlation/plotting
# ===============================
age_column = 'age'

def compute_valid_data(x_col):
    required_cols = [x_col, age_column, 'age_16', 'age_84',
                     'break_mass_err', 'age_err_low', 'age_err_high']
    available_cols = [c for c in required_cols if c in filtered_df.columns]
    df_valid = filtered_df[available_cols].dropna(subset=[x_col, age_column])
    return df_valid

def compute_pearson(df_valid, x_col):
    if df_valid[x_col].nunique() < 2 or df_valid[age_column].nunique() < 2:
        return np.nan, np.nan
    else:
        return pearsonr(df_valid[x_col], df_valid[age_column])

def compute_error_bars(df_valid, y_col='break_mass_err'):
    xerr = [df_valid['age_err_low'], df_valid['age_err_high']]
    yerr = df_valid[y_col] if y_col in df_valid.columns else None
    return xerr, yerr

# ===============================
# Break Mass vs Age plot
# ===============================
valid_break = compute_valid_data('break_mass')
r_break, p_break = compute_pearson(valid_break, 'break_mass')
xerr_break, yerr_break = compute_error_bars(valid_break)

plt.figure(figsize=(8,6))
plt.errorbar(
    valid_break[age_column],
    valid_break['break_mass'],
    xerr=xerr_break,
    yerr=yerr_break,
    fmt='o',
    color='steelblue',
    ecolor='gray',
    elinewidth=1,
    capsize=2,
    alpha=0.7,
    markeredgecolor='k'
)
plt.xlabel(r'$\log_{10}$ Age [yr]', fontsize=12)
plt.ylabel(r'Break Mass [M$_\odot$]', fontsize=12)
plt.title(f'Pearson r = {r_break:.1f} P-value = {p_break:.1e}', fontsize=14)
plt.tight_layout()
#plt.show()
plt.savefig(output_path)

print("\nBreak Mass vs Age")
print(f"Pearson r = {r_break:.3f}")
print(f"P-value = {p_break:.3e}")
