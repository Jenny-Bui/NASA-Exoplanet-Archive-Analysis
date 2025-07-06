
# NASA Exoplanet Archive Data Analysis Report

**Student Name:** Thi Minh Khue Bui

---

## Executive Summary

This report presents a comprehensive analysis of the NASA Exoplanet Archive dataset containing 9,903 confirmed exoplanet records. The analysis demonstrates proficiency across multiple weeks of course material (Weeks 7–11) including Python Standard Library, NumPy, Matplotlib, and Pandas. Key findings reveal evolving detection capabilities, distinct exoplanet populations, and strong correlations between stellar and planetary properties.

---

## Data Cleaning

### Data Quality Assessment

The original dataset required extensive cleaning to ensure reliable analysis. Multiple data quality issues were systematically identified and addressed:

#### Primary Issues Identified:

* **Missing Values:** Significant gaps in physical parameters, with completeness ranging from 16% (Stellar spectral class) to 100% (Basic identifiers)
* **Duplicate Records:** Multiple measurement entries for the same exoplanet from different studies and time periods
* **Invalid Physical Values:** Negative values for quantities that must be positive (masses, radii, temperatures, distances)
* **Extreme Outliers:** Likely measurement errors or data entry mistakes
* **Invalid Discovery Years:** Entries with unrealistic discovery dates (outside 1990–2025)

### Systematic Cleaning Process

Using a combination of Week 10–11 **Pandas** operations and Week 8 **NumPy** statistical methods:

* **Duplicate Removal:** `df.drop_duplicates()` to eliminate exact duplicates
* **Physical Validation:** Boolean indexing to remove rows with invalid physical values
* **Outlier Detection:** 3×IQR method using NumPy percentile calculations
* **Temporal Validation:** Filtered discovery years to reasonable bounds (1990-2025) based on exoplanet discovery history

---

### Strategic Decision: Retention of High-Missing-Value Columns

Columns with up to 84% missing values were retained for the following reasons:

* **Astronomical Data Reality:** Missing values in exoplanet research are often due to measurement difficulty rather than data quality issues. For example, planet temperatures are extremely challenging to measure and require specific observational conditions.  
* **Preserving Rare Information:** Columns like stellar spectral class, despite 84% missing values, provide crucial astrophysical context for the 16% of cases where data exists. Removing these columns would eliminate valuable scientific information.  
* **Method-Specific Limitations:** Columns like stellar spectral class, despite 84% missing values, provide crucial astrophysical context for the 16% of cases where data exists. Removing these columns would eliminate valuable scientific information.  
* **Statistical Power:** Even 20–40% of 9,903 records still yields 1,700–3,500 usable values
* **Complete Case Analysis Rejected:** Would reduce dataset to fewer than 500 records

**Result:**
Retained 85.2% of original data (8,669 of 9,903 records)

### Additional Cleaning Opportunities

* Cross-validation with external databases
* Uncertainty/error flagging
* Consistency checks between parameters
* Standardizing facility names and classification systems

---

## Numerical Analysis

### Statistical Summary

| Variable                | Count | Mean  | Median | Std Dev | Min   | Max  |
| ----------------------- | ----- | ----- | ------ | ------- | ----- | ---- |
| Orbital period (days)   | 6156  | 16.6  | 8.36   | 22.7    | 0.177 | 156  |
| Planet radius (R\_E)    | 4172  | 2.2   | 2.06   | 1.02    | 0.42  | 6.59 |
| Planet mass (M\_E)      | 1148  | 47.8  | 10.7   | 84.9    | 0.193 | 423  |
| Planet temperature (K)  | 2362  | 854   | 786    | 376     | 104   | 2330 |
| Stellar temperature (K) | 5719  | 5340  | 5530   | 808     | 2890  | 8720 |
| Stellar mass (M\_sol)   | 5200  | 0.901 | 0.925  | 0.238   | 0.117 | 1.78 |
| Stellar distance (pc)   | 6429  | 459   | 351    | 408     | 1.83  | 2010 |

---

### Statistical Insights

* **Distribution:** Planetary parameters are right-skewed (mean > median).

  * Highest skew: planet mass (CV = 1.77) and orbital period (CV = 1.37)
  * Stellar parameters show more symmetric distributions, with stellar temperature being slightly left-skewed, indicating our sample is biased toward stars slightly cooler than average.
* **Physical Insight:** 
* The wide range in planetary masses (2,190× span from minimum to maximum) reflects the diversity of exoplanet types, from small rocky worlds to massive gas giants.
* The typical exoplanet in our sample orbits closer to its star than Earth does to the Sun (median orbital period = 8.4 days vs. Earth's 365 days), highlighting the observational **bias toward close-in planets that are easier to detect**.
* The stellar sample represents primarily main-sequence stars like our Sun, with masses clustered around 0.9 solar masses.

---

## Simple Plot Analysis

### Plot Description
![image](https://github.com/user-attachments/assets/37008fe4-6a45-4143-876b-3939d74e2c9b)

Scatter plot of log(planet mass) vs discovery year for 1,148 exoplanets.

* Trend line: slope = -0.052 log(M\_E)/year
* Correlation: r = -0.588

### Analysis

* Dramatic sensitivity improvement: modern detection captures sub-Earth mass planets
* 1000× increase in mass sensitivity from 1995 to 2025
* Driven by improved photometry, instrumentation, and algorithms
* Plot highlights trend toward detecting smaller planets over time

---

## Multi-variable Plot

### Part b: Investigation Rationale

Analyzed 4 variables simultaneously:

* **X-axis:** Stellar temperature
* **Y-axis:** Planet temperature
* **Bubble size:** Planet mass
* **Color:** Discovery method

**Research Questions:**

* Do hotter stars host hotter planets?
* Are discovery methods biased?
* How does planet mass influence thermal relationships?

---

### Part c: Analysis and Conclusions
![image](https://github.com/user-attachments/assets/ad3e4d35-6bab-4667-8c1b-2ef266e2a9d2)

* Planetary temperatures are lower than stellar temperatures (as expected)
* Moderate correlation: stellar temp vs. planetary temp (r = 0.472)
* **Discovery method bias:**

  * Transit photometry detects a wide variety of planets
  * Radial velocity biased toward massive planets around cooler stars

---

## Extension Task

### a. Advanced Machine Learning Implementation

**Method:** K-means clustering
**Goal:** Discover natural groupings in multidimensional space

#### Technical Implementation

* **Features:** orbital period, planet radius, planet mass, stellar temperature, stellar mass
* **Preprocessing:** log transformation + standardization
* **Clustering:** k = 2 to 7 tested; elbow method used to determine optimal k
* **Validation:** ANOVA confirms statistical significance between clusters

**Tools Used:**

* `scikit-learn`: `StandardScaler`, `KMeans`
* `scipy.stats`: ANOVA test

---

### b. Scientific Value and Insights
![Figure_2](https://github.com/user-attachments/assets/bdae1cb4-5e2a-490f-9c86-06e04d883cf3)

The clustering analysis adds substantial value by revealing that exoplanets naturally segregate into three distinct populations, each with characteristic properties that likely reflect different formation mechanisms or evolutionary pathways. This unsupervised approach objectively identifies patterns that might be missed in traditional parameter-by-parameter analysis.

**Findings:**

* **Cluster 1 (185):** Small, close-in planets around solar-type stars
* **Cluster 2 (131):** Intermediate-mass planets, moderate orbits
* **Cluster 3 (64):** Large planets, wide orbits, massive stars

**Novel Insight:**

* Exoplanet populations form **discrete clusters** — not continuous
* All five variables show **significant differences (p < 0.001)**
* Supports **multiple planet formation mechanisms**

**Implications:**

These results support theories of multiple planet formation pathways and suggest that observational surveys should be designed to sample all three populations adequately. The distinct clustering also implies that statistical studies of exoplanet occurrence rates should account for population structure rather than treating all planets as a homogeneous sample.
