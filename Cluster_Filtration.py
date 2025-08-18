import pandas as pd
import numpy as np
import re

# --- Load the datasets ---
try:
    df = pd.read_csv('filtered_results_age.csv')
    print("Loaded filtered_results_age.csv successfully.")
except FileNotFoundError:
    print("Error: 'filtered_results_age.csv' not found. Please ensure the file is in the same directory.")
    exit()

try:
    dg = pd.read_csv('table2.csv')
    print("Loaded table2.csv successfully.")
except FileNotFoundError:
    print("Error: 'table2.csv' not found. Please ensure the file is in the same directory.")
    exit()

try:
    de = pd.read_csv('open_clusters_parameters.csv')
    print("Loaded open_clusters_parameters.csv successfully.")
except FileNotFoundError:
    print("Error: 'open_clusters_parameters.csv' not found. Please ensure the file is in the same directory.")
    exit()

# --- Step 1: Filter the primary DataFrame ---
print("Applying filters to the main DataFrame...")
filtered_df = df[
    (df['kind'].isin(['o', 'm'])) &
    (df['n_bins_before_break'] > 2) &
    (df['n_bins_after_break'] > 2) &
    (df['cst'] > 15) &
    (df['class_50'] > 0.75) &
    (df['a_v_50'] < 1.0) &
    (df['high_slope'] > -4.0) &
    (df['low_slope'] < 4.0) &
    (df['m_break_err'] < 0.15) &
    (df['distance_50'] < 2000)
].copy()

print(f"Filtered down to {len(filtered_df)} clusters.")

# --- Step 2: Prepare secondary dataframes ---
def format_cluster_name_de(name):
    match = re.search(r'\d', name)
    if match:
        index = match.start()
        return name[:index] + '_' + name[index:]
    return name

print("Standardizing cluster names for 'open_clusters_parameters'...")
de['Cluster'] = de['Cluster'].astype(str).apply(format_cluster_name_de)

# Normalize naming columns for merging
dg = dg.rename(columns={'Cluster': 'name'})
de = de.rename(columns={'Cluster': 'name'})

# --- Step 3: Start with default/fallback age from the original filtered df (age_50, age_16, age_84) ---
# FIX: Correctly calculate errors from percentiles
filtered_df['logAge'] = filtered_df['age_50']
filtered_df['e_logAge_upper'] = filtered_df['age_84'] - filtered_df['age_50']
filtered_df['e_logAge_lower'] = filtered_df['age_50'] - filtered_df['age_16']

# --- Step 4: Merge higher priority sources in order ---
# First, merge open_clusters_parameters (highest priority)
de_subset = de[['name', 'logA', 'e_logA', 'E_logA']].copy()
de_subset = de_subset.rename(columns={
    'logA': 'logAge_de',
    'e_logA': 'e_logAge_upper_de',
    'E_logA': 'e_logAge_lower_de'
})

# --- NEWLY ADDED CODE BLOCK STARTS HERE ---
# This block checks for and invalidates entries from 'de' with large relative errors.

# Define a threshold for what constitutes a 'too large' error in percent.
# For example, 25.0 means we discard values where the error is more than 25% of the age.
error_threshold_percent = 20.0

# Calculate the percentage error. We use np.abs to be safe.
# Replace potential zero in age with NaN to avoid division by zero errors.
logAge_for_calc = de_subset['logAge_de'].replace(0, np.nan)
percent_error_upper = np.abs((de_subset['e_logAge_upper_de'] / logAge_for_calc) * 100)
percent_error_lower = np.abs((de_subset['e_logAge_lower_de'] / logAge_for_calc) * 100)

# Identify rows where either error percentage is too high or the age is invalid.
invalid_error_condition = (
    (percent_error_upper > error_threshold_percent) |
    (percent_error_lower > error_threshold_percent) |
    (de_subset['logAge_de'].isna())
)

# For rows with invalid errors, set their age to NaN.
# This will cause them to be skipped in the merge update, allowing a fallback to other catalogs.
print(f"Checking for large errors in 'de' catalog. Found and will skip {invalid_error_condition.sum()} rows.")
de_subset.loc[invalid_error_condition, 'logAge_de'] = np.nan

# --- NEWLY ADDED CODE BLOCK ENDS HERE ---


filtered_df = filtered_df.merge(
    de_subset,
    left_on='cluster_name',
    right_on='name',
    how='left',
    suffixes=('', '_de')
)

# Use de values where present (and not invalidated by our check)
has_de = ~filtered_df['logAge_de'].isna()
filtered_df.loc[has_de, 'logAge'] = filtered_df.loc[has_de, 'logAge_de']
filtered_df.loc[has_de, 'e_logAge_upper'] = filtered_df.loc[has_de, 'e_logAge_upper_de']
filtered_df.loc[has_de, 'e_logAge_lower'] = filtered_df.loc[has_de, 'e_logAge_lower_de']

# Then, for those still missing, use table2 (second priority)
dg_subset = dg[['name', 'logage', 'e_logage']].copy().rename(columns={
    'logage': 'logAge_dg',
    'e_logage': 'e_logAge_dg'
})

filtered_df = filtered_df.merge(
    dg_subset,
    left_on='cluster_name',
    right_on='name',
    how='left',
    suffixes=('', '_dg')
)

# Apply table2 where de was missing (or invalidated) but dg exists
need_dg = filtered_df['logAge_de'].isna() & ~filtered_df['logAge_dg'].isna()
filtered_df.loc[need_dg, 'logAge'] = filtered_df.loc[need_dg, 'logAge_dg']
filtered_df.loc[need_dg, 'e_logAge_upper'] = filtered_df.loc[need_dg, 'e_logAge_dg']
filtered_df.loc[need_dg, 'e_logAge_lower'] = filtered_df.loc[need_dg, 'e_logAge_dg']

# Clean up intermediate columns
cols_to_drop = [c for c in filtered_df.columns if c.endswith('_de') or c.endswith('_dg') or c == 'name']
filtered_df = filtered_df.drop(columns=cols_to_drop, errors='ignore')

print("Finished adding age data to the filtered clusters.")

# --- Step 5: Save result ---
output_filename = 'break_mass_filtered_clusters.csv'
filtered_df.to_csv(output_filename, index=False) # Changed to index=False, which is more common
print(f"Successfully saved the final filtered data to '{output_filename}'.")

print("\nFinal DataFrame head:")
print(filtered_df.head())
print("\nFinal DataFrame columns:")
print(filtered_df.columns)