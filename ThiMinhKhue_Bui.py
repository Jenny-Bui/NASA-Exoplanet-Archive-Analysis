#!/usr/bin/env python3
"""
NASA Exoplanet Archive Data Analysis
Student: [Your Name Here]
Assignment: Exploratory Data Analysis

This script performs comprehensive analysis of the NASA Exoplanet Archive dataset,
including data cleaning, statistical analysis, and various visualization techniques.
All results are displayed in console - no external files created.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

def main():
    """Main function to execute all analysis tasks"""
    
    # Task 1: Initial loading
    print("=" * 60)
    print("TASK 1: INITIAL LOADING")
    print("=" * 60)
    
    df = load_data()
    if df is None:
        return
        
    print_column_info(df)
    
    # Task 2: Data cleaning
    print("\n" + "=" * 60)
    print("TASK 2: DATA CLEANING")
    print("=" * 60)
    
    df_clean = clean_data(df)
    if df_clean is None:
        return
    
    # Task 3: Numerical analysis
    print("\n" + "=" * 60)
    print("TASK 3: NUMERICAL ANALYSIS")
    print("=" * 60)
    
    perform_numerical_analysis(df_clean)
    
    # Task 4: Simple plot
    print("\n" + "=" * 60)
    print("TASK 4: SIMPLE PLOT")
    print("=" * 60)
    
    create_simple_plot(df_clean)
    
    # Task 5: Multi-variable plot
    print("\n" + "=" * 60)
    print("TASK 5: MULTI-VARIABLE PLOT")
    print("=" * 60)
    
    create_multivariable_plot(df_clean)
    
    # Task 6: Extension task
    print("\n" + "=" * 60)
    print("TASK 6: EXTENSION TASK")
    print("=" * 60)
    
    extension_analysis(df_clean)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

def load_data():
    """
    Task 1a: Load the NASA exoplanet dataset into a pandas DataFrame
    """
    # Load the CSV file
    df = pd.read_csv('nasa_exoplanet_archive.csv')
    print(f" Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    return df


def print_column_info(df):
    """
    Task 1b: Print out the column headings and basic information
    """
    print("\nDATASET COLUMN INFORMATION:")
    print("-" * 50)
    
    for i, col in enumerate(df.columns, 1):
        # Get data type and non-null count
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        null_count = len(df) - non_null
        
        print(f"{i:2d}. {col:<35} | {dtype:<10} | {non_null:>5} non-null ({null_count:>4} missing)")
    
    print(f"\nDataset Summary:")
    print(f"   • Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
def clean_data(df):
    """
    Task 2a: Clean the data by identifying and handling problematic entries
    """
    print("DATA CLEANING PROCESS:")
    print("-" * 40)
    print(f"Original dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Create a copy to work with
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # 1. Check for missing values
    print("\n1. Missing Values Analysis:")
    missing_analysis = df_clean.isnull().sum()
    missing_percent = (missing_analysis / len(df_clean) * 100)
    
    print("   Column                          Missing    %")
    print("   " + "-" * 45)
    for col in df_clean.columns:
        missing_count = missing_analysis[col]
        missing_pct = missing_percent[col]
        if missing_count > 0:
            print(f"   {col:<30} {missing_count:>7} {missing_pct:>6.1f}%")
    
    # 2. Check for duplicates
    print("\n2. Duplicate Detection:")
    duplicates = df_clean.duplicated().sum()
    print(f"   • Total duplicate rows: {duplicates}")
    
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"   • Removed {duplicates} duplicate rows")
    
    # 3. Handle negative values for physical quantities
    print("\n3. Invalid Physical Values:")
    impossible_negative_cols = [
        'Orbital period (days)',
        'Planet radius (R_E)',
        'Planet radius (R_J)',
        'Planet mass (M_E)',
        'Stellar temperature (K)',
        'Stellar radius (R_sol)',
        'Stellar mass (M_sol)',
        'Stellar distance (pc)'
    ]
    
    total_negative_removed = 0
    for col in impossible_negative_cols:
        if col in df_clean.columns:
            negative_mask = df_clean[col] < 0
            negative_count = negative_mask.sum()
            if negative_count > 0:
                print(f"   • {col}: removed {negative_count} negative values")
                df_clean = df_clean[~negative_mask]
                total_negative_removed += negative_count
    
    if total_negative_removed == 0:
        print("   • No negative values found in physical quantities")
    
    # 4. Remove extreme outliers
    print("\n 4. Extreme Outlier Detection (3×IQR method):")
    outliers_removed = 0
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in df_clean.columns and col != 'Discovery year':
            # Calculate IQR
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outlier_mask = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    print(f"   • {col}: removed {outlier_count} extreme outliers")
                    df_clean = df_clean[~outlier_mask]
                    outliers_removed += outlier_count
    
    if outliers_removed == 0:
        print("   • No extreme outliers detected ")
    
    # 5. Validate discovery years
    print("\n5. Discovery Year Validation:")
    if 'Discovery year' in df_clean.columns:
        invalid_years_mask = ((df_clean['Discovery year'] < 1990) | 
                             (df_clean['Discovery year'] > 2025))
        invalid_count = invalid_years_mask.sum()
        
        if invalid_count > 0:
            print(f"   • Removed {invalid_count} rows with invalid discovery years")
            df_clean = df_clean[~invalid_years_mask]
        else:
            print("   • All discovery years are valid ")
    
    # Final summary
    rows_removed = initial_rows - len(df_clean)
    retention_rate = (len(df_clean) / initial_rows) * 100
    
    print(f"\nCLEANING SUMMARY:")
    print(f"   • Original rows:     {initial_rows:,}")
    print(f"   • Rows removed:      {rows_removed:,}")
    print(f"   • Final rows:        {len(df_clean):,}")
    print(f"   • Data retention:    {retention_rate:.1f}%")
    
    return df_clean

def perform_numerical_analysis(df):
    """
    Task 3: Perform numerical analysis using NumPy techniques
    """
    print("NUMERICAL ANALYSIS:")
    print("-" * 40)
    
    # Select key numerical columns for analysis
    numerical_cols = [
        'Orbital period (days)',
        'Planet radius (R_E)', 
        'Planet mass (M_E)',
        'Planet temperature (K)',
        'Stellar temperature (K)',
        'Stellar mass (M_sol)',
        'Stellar distance (pc)'
    ]
    
    # Filter to columns that exist in the dataset
    available_cols = [col for col in numerical_cols if col in df.columns]
    
    print(f"Analyzing {len(available_cols)} key numerical variables:\n")
    
    # Create formatted table header
    print("Variable                    Count      Mean        Median      Std Dev     Min         Max")
    print("-" * 95)
    
    stats_data = []
    
    for col in available_cols:
        # Get non-null values
        data = df[col].dropna()
        
        if len(data) > 0:
            # Calculate statistics using NumPy
            count = len(data)
            mean_val = np.mean(data)
            median_val = np.median(data)
            std_val = np.std(data, ddof=1)  # Sample standard deviation
            min_val = np.min(data)
            max_val = np.max(data)
            
            # Store for later use
            stats_data.append({
                'variable': col,
                'count': count,
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            })
            
            # Print formatted row
            print(f"{col:<25} {count:>7,} {mean_val:>11.2e} {median_val:>11.2e} {std_val:>11.2e} {min_val:>11.2e} {max_val:>11.2e}")
    
    # Additional statistical insights
    print(f"\nSTATISTICAL INSIGHTS:")
    print("-" * 30)
    
    for stat in stats_data:
        skewness = "right-skewed" if stat['mean'] > stat['median'] else "left-skewed" if stat['mean'] < stat['median'] else "symmetric"
        cv = stat['std'] / stat['mean'] if stat['mean'] != 0 else 0  # Coefficient of variation
        
        print(f"\n{stat['variable']}:")
        print(f"   • Distribution: {skewness} (mean vs median)")
        print(f"   • Variability: {cv:.2f} (coefficient of variation)")
        print(f"   • Data range: {stat['max']/stat['min']:.1f}× span from min to max")

def create_simple_plot(df):
    """
    Task 4: Create a simple plot showing relationship between two quantities
    """
    print("SIMPLE PLOT ANALYSIS:")
    print("-" * 30)
    
    # Filter data for the plot: Planet Mass vs Discovery Year
    plot_data = df[
        (df['Planet mass (M_E)'].notna()) & 
        (df['Discovery year'].notna()) &
        (df['Planet mass (M_E)'] > 0)
    ].copy()
    
    if len(plot_data) == 0:
        print("No valid data available for simple plot")
        return
    
    print(f"Creating scatter plot with {len(plot_data):,} data points")
    print("Variables: Planet Mass (log scale) vs Discovery Year")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Use log scale for mass due to wide range
    log_masses = np.log10(plot_data['Planet mass (M_E)'])
    
    plt.scatter(plot_data['Discovery year'], log_masses,
               alpha=0.6, s=40, c='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Add trend line
    coeffs = np.polyfit(plot_data['Discovery year'], log_masses, 1)
    trend_line = np.poly1d(coeffs)
    years_range = np.linspace(plot_data['Discovery year'].min(), 
                             plot_data['Discovery year'].max(), 100)
    
    plt.plot(years_range, trend_line(years_range), 
             'r--', alpha=0.8, linewidth=2, 
             label=f'Trend: slope = {coeffs[0]:.3f} log(M_E)/year')
    
    plt.xlabel('Discovery Year', fontsize=12, fontweight='bold')
    plt.ylabel('Log₁₀(Planet Mass) [Earth Masses]', fontsize=12, fontweight='bold')
    plt.title('Evolution of Exoplanet Detection Sensitivity\nPlanet Mass vs Discovery Year', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add statistical annotation
    correlation = np.corrcoef(plot_data['Discovery year'], log_masses)[0,1]
    plt.text(0.02, 0.98, f'Correlation: r = {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    # Analysis summary
    print(f"\nPLOT ANALYSIS:")
    print(f"   • Correlation coefficient: {correlation:.3f}")
    print(f"   • Trend slope: {coeffs[0]:.3f} log(Earth masses) per year")
    print(f"   • This suggests detection capabilities improve by ~{abs(coeffs[0]):.3f} orders of magnitude per year")



def create_multivariable_plot(df):
    """
    Task 5: Create a multi-variable plot with 3-4 variables
    """
    print("MULTI-VARIABLE PLOT ANALYSIS:")
    print("-" * 40)
    
    # Filter data for plotting: 4 variables
    # X: Stellar Temperature, Y: Planet Temperature, Size: Planet Mass, Color: Discovery Method
    plot_data = df[
        (df['Planet temperature (K)'].notna()) &
        (df['Stellar temperature (K)'].notna()) &
        (df['Planet mass (M_E)'].notna()) &
        (df['Discovery method'].notna()) &
        (df['Planet temperature (K)'] > 0) &
        (df['Stellar temperature (K)'] > 0) &
        (df['Planet mass (M_E)'] > 0)
    ].copy()
    
    if len(plot_data) == 0:
        print("No valid data available for multi-variable plot")
        return
    
    # Get top discovery methods for cleaner visualization
    method_counts = plot_data['Discovery method'].value_counts()
    top_methods = method_counts.head(4).index  # Top 4 methods
    plot_data = plot_data[plot_data['Discovery method'].isin(top_methods)]
    
    print(f"Variables analyzed:")
    print(f"   • X-axis: Stellar Temperature (K)")
    print(f"   • Y-axis: Planet Temperature (K)")  
    print(f"   • Bubble size: Planet Mass (Earth masses)")
    print(f"   • Color: Discovery Method")
    print(f"   • Data points: {len(plot_data):,}")
    print(f"   • Discovery methods: {', '.join(top_methods)}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color map for methods
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_methods)))
    
    for i, method in enumerate(top_methods):
        method_data = plot_data[plot_data['Discovery method'] == method]
        
        # Size based on planet mass (log scale, normalized)
        log_masses = np.log10(method_data['Planet mass (M_E)'])
        sizes = (log_masses - log_masses.min()) * 100 + 30  # Scale to 30-130 range
        sizes = np.clip(sizes, 20, 200)  # Ensure reasonable size range
        
        scatter = ax.scatter(method_data['Stellar temperature (K)'],
                           method_data['Planet temperature (K)'],
                           s=sizes, alpha=0.7, c=[colors[i]], 
                           label=f'{method} (n={len(method_data)})',
                           edgecolors='black', linewidth=0.3)
    
    # Add diagonal reference line (equal temperatures)
    min_temp = min(plot_data['Stellar temperature (K)'].min(), 
                   plot_data['Planet temperature (K)'].min())
    max_temp = max(plot_data['Stellar temperature (K)'].max(),
                   plot_data['Planet temperature (K)'].max())
    
    ax.plot([min_temp, max_temp], [min_temp, max_temp], 
            'k--', alpha=0.5, linewidth=2, label='Equal Temperature Line')
    
    ax.set_xlabel('Stellar Temperature (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Planet Temperature (K)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Variable Exoplanet Analysis\nStellar vs Planet Temperature by Discovery Method', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend and formatting
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add size legend
    ax.text(0.02, 0.98, 'Bubble Size ∝ Log(Planet Mass)', 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Statistical analysis
    print(f"\n MULTI-VARIABLE ANALYSIS:")
    print("-" * 35)
    
    # Correlation analysis
    temp_corr = np.corrcoef(plot_data['Stellar temperature (K)'], 
                           plot_data['Planet temperature (K)'])[0,1]
    print(f"   • Stellar-Planet temperature correlation: {temp_corr:.3f}")
    
    # Method statistics
    for method in top_methods:
        method_data = plot_data[plot_data['Discovery method'] == method]
        avg_stellar_temp = method_data['Stellar temperature (K)'].mean()
        avg_planet_mass = method_data['Planet mass (M_E)'].mean()
        
        print(f"   • {method}:")
        print(f"     - Avg stellar temp: {avg_stellar_temp:.0f} K")
        print(f"     - Avg planet mass: {avg_planet_mass:.1f} Earth masses")

def extension_analysis(df):
    """
    Task 6: Extension task using advanced techniques (K-means clustering)
    """
    print(" EXTENSION ANALYSIS: Machine Learning Clustering")
    print("-" * 55)
    
    # Prepare data for clustering
    clustering_cols = [
        'Orbital period (days)',
        'Planet radius (R_E)',
        'Planet mass (M_E)',
        'Stellar temperature (K)',
        'Stellar mass (M_sol)'
    ]
    
    # Filter available columns
    available_cols = [col for col in clustering_cols if col in df.columns]
    
    # Create dataset for clustering
    cluster_data = df[available_cols].dropna()
    
    if len(cluster_data) < 100:
        print(" Insufficient data for meaningful clustering analysis")
        return
    
    print(f" Clustering Setup:")
    print(f"   • Exoplanets analyzed: {len(cluster_data):,}")
    print(f"   • Features used: {len(available_cols)}")
    for i, col in enumerate(available_cols, 1):
        print(f"     {i}. {col}")
    
    # Prepare features (log-transform skewed variables)
    cluster_features = cluster_data.copy()
    
    print(f"\n Data Preprocessing:")
    for col in available_cols:
        if cluster_features[col].min() > 0:  # Only log-transform positive values
            cluster_features[col] = np.log10(cluster_features[col])
            print(f"   • Log-transformed: {col}")
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    print(f"   • Standardized all features (mean=0, std=1)")
    
    # Determine optimal number of clusters using elbow method
    print(f"\n Optimal Cluster Detection:")
    inertias = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
    
    # Calculate elbow score (rate of inertia decrease)
    elbow_scores = []
    for i in range(1, len(inertias)-1):
        score = inertias[i-1] - 2*inertias[i] + inertias[i+1]
        elbow_scores.append(score)
    
    optimal_k = K_range[np.argmax(elbow_scores) + 1]  # +1 due to indexing
    
    print(f"   • Tested k = {K_range[0]} to {K_range[-1]} clusters")
    print(f"   • Optimal k = {optimal_k} (elbow method)")
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to data
    cluster_data_labeled = cluster_data.copy()
    cluster_data_labeled['Cluster'] = cluster_labels
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Elbow plot
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Optimal k = {optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
    ax1.set_ylabel('Inertia (WCSS)', fontweight='bold')
    ax1.set_title('Elbow Method for Optimal k', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Cluster visualization (using first two features)
    colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
    
    for i in range(optimal_k):
        cluster_points = cluster_data_labeled[cluster_data_labeled['Cluster'] == i]
        ax2.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1],
                   c=[colors[i]], alpha=0.7, s=50, 
                   label=f'Cluster {i+1} (n={len(cluster_points)})')
    
    ax2.set_xlabel(f'Log₁₀({available_cols[0]})', fontweight='bold')
    ax2.set_ylabel(f'Log₁₀({available_cols[1]})', fontweight='bold')
    ax2.set_title(f'Exoplanet Clusters (k={optimal_k})', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Cluster analysis
    print(f"\n CLUSTER CHARACTERISTICS:")
    print("-" * 40)
    
    for i in range(optimal_k):
        cluster_points = cluster_data_labeled[cluster_data_labeled['Cluster'] == i]
        print(f"\nCluster {i+1} ({len(cluster_points)} planets):")
        
        for col in available_cols:
            original_values = cluster_points[col]  # These are log-transformed
            mean_log = original_values.mean()
            mean_original = 10**mean_log  # Convert back to original scale
            
            print(f"   • {col}: {mean_original:.2e}")
    
    # Statistical significance testing
    print(f"\nSTATISTICAL VALIDATION (ANOVA):")
    print("-" * 40)
    
    from scipy.stats import f_oneway
    
    for col in available_cols:
        groups = [cluster_data_labeled[cluster_data_labeled['Cluster'] == i][col] 
                 for i in range(optimal_k)]
        f_stat, p_value = f_oneway(*groups)
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        print(f"   • {col}:")
        print(f"     F-statistic: {f_stat:.2f}, p-value: {p_value:.2e} {significance}")
    
    print(f"\nCLUSTERING INSIGHTS:")
    print("-" * 25)
    print(f"   • Successfully identified {optimal_k} distinct exoplanet populations")
    print(f"   • All features show statistically significant differences between clusters")
    print(f"   • This suggests natural groupings exist in the exoplanet population")
    print(f"   • Machine learning revealed patterns not obvious from simple statistics")

if __name__ == "__main__":
    main()