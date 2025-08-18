import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import minimize
import os

output_dir = "Figures"
os.makedirs(output_dir, exist_ok=True)
output_path = output_path = os.path.join(output_dir, "figure_1")
# Set plotting style
sns.set_style("white")

def kroupa_imf(masses, temp, alpha1=0.3, alpha2=1.3, alpha3=2.3, total_mass=1000):
    """Calculate Kroupa IMF with temperature-dependent mass breaks."""
    m_break1 = 0.08 * (temp / 20)**2
    m_break2 = 0.5 * (temp / 20)**2
    
    # Initialize IMF values
    imf_vals = np.zeros_like(masses, dtype=float)
    
    # Define mass regime masks
    mask1 = masses < m_break1
    mask2 = (masses >= m_break1) & (masses < m_break2)
    mask3 = masses >= m_break2
    
    # Calculate normalization constants
    C2 = m_break2**(alpha2 - alpha3) if m_break2 > 0 else 0
    C1 = C2 * (m_break1**(alpha1 - alpha2)) if m_break1 > 0 else 0
    
    # Apply power laws in each regime
    imf_vals[mask3] = masses[mask3]**(-alpha3)
    imf_vals[mask2] = C2 * masses[mask2]**(-alpha2)
    imf_vals[mask1] = C1 * masses[mask1]**(-alpha1)
    
    # Scale to desired total mass
    current_total_mass = np.trapz(masses * imf_vals, x=masses)
    scaling_factor = total_mass / current_total_mass if current_total_mass > 0 else 0
    
    return imf_vals * scaling_factor

def kroupa_imf_scaled(masses, temp, alpha1=0.3, alpha2=1.3, alpha3=2.3, total_mass=1000):
    """Alternative normalization approach for fitting."""
    m_break1 = 0.08 * (temp / 20)**2
    m_break2 = 0.5 * (temp / 20)**2
    
    imf_vals = np.zeros_like(masses, dtype=float)
    mask1 = masses < m_break1
    mask2 = (masses >= m_break1) & (masses < m_break2)
    mask3 = masses >= m_break2
    
    # Different normalization approach
    imf_vals[mask3] = masses[mask3]**(-alpha3)
    norm_const_23 = m_break2**(-alpha3) / (m_break2**(-alpha2))
    
    if norm_const_23 > 0:
        imf_vals[mask2] = norm_const_23 * masses[mask2]**(-alpha2)
        if len(masses[mask1]) > 0:
            val_at_break1 = norm_const_23 * m_break1**(-alpha2)
            norm_const_12 = val_at_break1 / (m_break1**(-alpha1))
            imf_vals[mask1] = norm_const_12 * masses[mask1]**(-alpha1)
    
    # Scale to total mass
    current_total_mass = np.trapz(masses * imf_vals, x=masses)
    scaling_factor = total_mass / current_total_mass if current_total_mass > 0 else 0
    
    return imf_vals * scaling_factor

def calculate_turnoff_mass(age_gyr):
    """Calculate main sequence turnoff mass for given age."""
    return (age_gyr / 10.0)**(-0.4) if age_gyr > 0 else np.inf

def calculate_dmf_final(masses, mu, m_depl, delta_depl, mu_depl, stripping_aggressiveness=1.5):
    """Calculate final mass function changes due to dynamical evolution."""
    if mu >= mu_depl:
        uniform_delta = delta_depl * (1.0 - mu) / (1.0 - mu_depl)
        return np.full_like(masses, uniform_delta)
    else:
        delta_uniform = delta_depl * (np.log10(mu) / np.log10(mu_depl))
        a1 = stripping_aggressiveness * (1 - mu / mu_depl)
        a2 = 0.356 * a1 + 0.019 * a1**2
        
        delta_vals = np.zeros_like(masses)
        poly_mask = masses < m_depl
        delta_vals[~poly_mask] = delta_uniform
        
        if np.any(poly_mask):
            l = np.log10(masses[poly_mask] / m_depl)
            delta_vals[poly_mask] = delta_uniform + a1 * l - a2 * l**2
            
        return delta_vals

def calculate_t_depl(t_rh0_myr, rh_rt_initial):
    """Calculate depletion timescale."""
    log_t_depl_myr = -0.210 + 0.873 * np.log10(t_rh0_myr) - 1.084 * np.log10(rh_rt_initial)
    return 10**log_t_depl_myr / 1000  # Convert to Gyr

def calculate_m_depl(t_depl_gyr):
    """Calculate depletion mass."""
    m_turnoff_at_t_depl = calculate_turnoff_mass(t_depl_gyr)
    x = 5.0
    return (1.14**x + (0.60 * m_turnoff_at_t_depl)**x)**(1/x)

def calculate_delta_depl(cluster_type, t_1_percent_gyr=None):
    """Calculate depletion parameter based on cluster type."""
    if cluster_type == "tidal-filling":
        if t_1_percent_gyr is None:
            raise ValueError("t_1_percent_gyr must be provided for tidal-filling clusters.")
        return 0.35 - 0.12 * np.log10(t_1_percent_gyr * 1000)
    elif cluster_type == "under-filling":
        return -0.06
    elif cluster_type == "mass-segregated":
        return 0.0
    else:
        raise ValueError("cluster_type must be 'tidal-filling', 'under-filling', or 'mass-segregated'")

def dndlogm(masses, temp, norm):
    """Convert dN/dm to dN/dlogM."""
    dndm = kroupa_imf_scaled(masses, temp, total_mass=norm)
    return dndm * masses * np.log(10)

# --- Main Analysis ---

# Set up figure
plt.rcParams['figure.figsize'] = [16, 8]  # Set default figure size
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 6))

# Cluster parameters
t_rh0_myr = 4050
r_h0 = 11.43
r_j = 61.1
rh_rt_initial = r_h0 / r_j
cluster_type = "under-filling"
mu_depl = 0.66

# Calculate derived parameters
t_depl = calculate_t_depl(t_rh0_myr, rh_rt_initial)
m_depl = calculate_m_depl(t_depl)
delta_depl = calculate_delta_depl(cluster_type)

print(f"--- Calculated Cluster Parameters ---")
print(f"Depletion Time: {t_depl:.2f} Gyr")
print(f"Depletion Mass: {m_depl:.2f} M_sun")
print(f"Depletion Loss: {delta_depl:.2f}")
print(f"-------------------------------------\n")

# Mass grid and evolution parameters
m_min, m_max = 0.01, 100.0
masses = np.logspace(np.log10(m_min), np.log10(m_max), 500)
temp = 20
total_cloud_mass_for_imf = 10
assigned_ages_gyr = [0.0, 10, 20, 24]
mu_values = [1.0, 0.50, 0.2, 0.1]

# Plot 1: IMF Evolution
initial_imf = kroupa_imf(masses, temp, total_mass=total_cloud_mass_for_imf)
sns.lineplot(x=masses, y=initial_imf, label=f'Initial IMF (T={temp} K)',
             lw=2.5, linestyle='-', color='darkred', ax=ax1)

turnoff_label_plotted = True
for i, age in enumerate(assigned_ages_gyr[1:]):
    mu = mu_values[i+1]
    # compute evolution before truncation
    delta_vals = calculate_dmf_final(masses, mu, m_depl, delta_depl, mu_depl)
    evolved_mf = initial_imf * (10**delta_vals)
    m_turnoff = calculate_turnoff_mass(age)

    # get y value at turnoff from the unmasked curve
    idx_turnoff = np.argmin(np.abs(masses - m_turnoff))
    y_turnoff = evolved_mf[idx_turnoff]

    # apply turnoff truncation
    evolved_mf[masses > m_turnoff] = np.nan

    label = (f'T={temp} K, μ={mu:.2f} (t≈{age:.1f} Gyr)')
    sns.lineplot(x=masses, y=evolved_mf, label=label, lw=1.5,
                 linestyle='--', color='darkred', ax=ax1)

    # draw vertical line if valid
    if np.isfinite(y_turnoff) and y_turnoff > 0:
        bottom = ax1.get_ylim()[0]
        ax1.vlines(m_turnoff, ymin=bottom, ymax=y_turnoff,
                       linestyles='--', linewidth=1.2, color='darkred')


# Configure Plot 1
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel("Stellar Mass (M$_{\\odot}$)", fontsize=14)
ax1.set_ylabel("dN/dM (Number of Stars per Mass Interval)", fontsize=14)
ax1.set_xlim(0.08, 20)
ax1.set_ylim(1e-2, 1000)
ax1.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 1.0))

# Plot 2: Data Fitting
# Load observational data
df = pd.read_csv("Default_Dataset.csv", header=None)
df.columns = ['log_mass', 'log_density']
df['mass'] = 10 ** df['log_mass']
df['density'] = 10 ** df['log_density']

# Define objective functions for fitting
def objective_blue(params):
    temp, norm = params
    if temp < 5 or temp > 100 or norm <= 0:
        return np.inf
    model = dndlogm(df['mass'].values, temp, norm)
    return np.sum((np.log10(model) - np.log10(df['density'].values))**2)

def objective_red(params):
    norm = params[0]
    if norm <= 0:
        return np.inf
    model = dndlogm(df['mass'].values, 20, norm)
    return np.sum((np.log10(model) - np.log10(df['density'].values))**2)

# Perform fits
result_blue = minimize(objective_blue, x0=[30, 1000], bounds=[(5, 100), (10, 10000)])
result_red = minimize(objective_red, x0=[1000], bounds=[(10, 10000)])

best_temp, best_norm_blue = result_blue.x
best_norm_red = result_red.x[0]

# Generate best-fit models
masses_fit = np.logspace(np.log10(0.01), np.log10(100), 1300)
best_fit_blue = dndlogm(masses_fit, best_temp, best_norm_blue)
best_fit_red = dndlogm(masses_fit, 20, best_norm_red)

# Plot data and fits
sns.set_style("ticks")
ax2.plot(masses_fit, best_fit_blue, lw=2, label=f'Best-fit (T = {best_temp:.1f} K)', color='blue')
ax2.plot(masses_fit, best_fit_red, lw=2, label='Best-fit at 20 K', color='darkred')
ax2.plot(df['mass'], df['density'], linestyle='--', lw=2, label='Guszejnov 2022', color='blue')

# Configure Plot 2
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Stellar Mass (M$_{\\odot}$)', fontsize=14)
ax2.set_ylabel('dN/dM', fontsize=14)
ax2.grid(False)
ax2.set_xlim(0.01, 100)

ax2.legend()

plt.tight_layout()
#plt.show()
plt.savefig(output_path)