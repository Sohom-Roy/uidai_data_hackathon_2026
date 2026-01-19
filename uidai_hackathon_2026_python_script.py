#!/usr/bin/env python
# coding: utf-8

# # UIDAI Data Hackathon 2026  
# ## Identifying Aadhaar Exclusion Risk Using Enrolment & Update Patterns
# 
# ---
# 
# ### 📍 Problem Context
# Aadhaar is a foundational digital identity system in India. However, delays or failures in
# enrolment updates, demographic corrections, and biometric revalidation can lead to **service exclusion**,
# especially for children transitioning into adulthood and vulnerable populations.
# 
# ---
# 
# ### 🎯 Objective
# This project aims to build a **data‑driven Aadhaar Exclusion Risk Index** using:
# - Enrolment volumes
# - Demographic update activity
# - Biometric update activity
# 
# The final output is a **state‑level risk ranking** to help policymakers identify
# regions requiring targeted Aadhaar service interventions.
# 
# 

# In[1]:


# Core data libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# File handling
import glob

# Display & style settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# In[2]:


# ---------- Enrolment Data ----------
enrolment_files = glob.glob("api_data_aadhar_enrolment/*.csv")
print("Enrolment files found:", len(enrolment_files))

enrolment_df = pd.concat(
    [pd.read_csv(f) for f in enrolment_files],
    ignore_index=True
)

# ---------- Demographic Data ----------
demographic_files = glob.glob("api_data_aadhar_demographic/*.csv")
print("Demographic files found:", len(demographic_files))

demographic_df = pd.concat(
    [pd.read_csv(f) for f in demographic_files],
    ignore_index=True
)

# ---------- Biometric Data ----------
biometric_files = glob.glob("api_data_aadhar_biometric/*.csv")
print("Biometric files found:", len(biometric_files))

biometric_df = pd.concat(
    [pd.read_csv(f) for f in biometric_files],
    ignore_index=True
)
assert len(enrolment_files) > 0, "No enrolment files found. Check folder path."
assert len(demographic_files) > 0, "No demographic files found. Check folder path."
assert len(biometric_files) > 0, "No biometric files found. Check folder path."


# In[3]:


print("Enrolment Columns:\n", enrolment_df.columns)
print("\nDemographic Columns:\n", demographic_df.columns)
print("\nBiometric Columns:\n", biometric_df.columns)




# In[4]:


enrolment_df.head()


# In[5]:


demographic_df.head()


# In[6]:


biometric_df.head()


# # Date Parsing & Initial Standardization
# 
# ## Why this step is important
# All three datasets come from different API pulls.
# The UIDAI datasets are time‑series based.
# To perform year‑wise aggregation and trend analysis, we must:
# - Convert date columns into proper datetime objects
# - Remove invalid / corrupt date entries
# - Extract the year field for downstream grouping
# - Standardize column names
# - Ensure consistent temporal alignment (year)
#   
# This ensures temporal consistency across enrolment, demographic, and biometric datasets.

# In[7]:


# -----------------------------------
# Standardize column names
# -----------------------------------
enrolment_df.columns = enrolment_df.columns.str.lower().str.strip()
demographic_df.columns = demographic_df.columns.str.lower().str.strip()
biometric_df.columns = biometric_df.columns.str.lower().str.strip()

# -----------------------------------
# Parse date columns safely
# -----------------------------------
enrolment_df['date'] = pd.to_datetime(enrolment_df['date'], errors='coerce')
demographic_df['date'] = pd.to_datetime(demographic_df['date'], errors='coerce')
biometric_df['date'] = pd.to_datetime(biometric_df['date'], errors='coerce')

# Drop invalid dates
enrolment_df = enrolment_df.dropna(subset=['date'])
demographic_df = demographic_df.dropna(subset=['date'])
biometric_df = biometric_df.dropna(subset=['date'])

# Extract year
enrolment_df['year'] = enrolment_df['date'].dt.year
demographic_df['year'] = demographic_df['date'].dt.year
biometric_df['year'] = biometric_df['date'].dt.year

print("Date parsing completed successfully.")


# # Data Cleaning & Preprocessing
# 
# ## 🧠 Why this step?
# UIDAI datasets often suffer from:
# - Inconsistent state spellings
# - Extra spaces
# - Legacy UT names
# - Multiple representations of merged UTs
# 
# We clean once, correctly, and globally.

# In[8]:


def clean_state_names(df):
    df['state'] = (
        df['state']
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace('&', 'and')
    )

    state_fixes = {
        'westbengal': 'west bengal',
        'west  bengal': 'west bengal',
        'west bangal': 'west bengal',
        'orissa': 'odisha',
        'pondicherry': 'puducherry',
        'andaman & nicobar islands': 'andaman and nicobar islands',
        'the dadra and nagar haveli and daman and diu':
            'dadra and nagar haveli and daman and diu',
        'dadra and nagar haveli': 'dadra and nagar haveli and daman and diu',
        'daman and diu': 'dadra and nagar haveli and daman and diu'
    }

    df['state'] = df['state'].replace(state_fixes)

    # Remove numeric garbage states
    df = df[~df['state'].str.isnumeric()]

    return df


# In[9]:


# Apply cleaning to all datasets
enrolment_df = clean_state_names(enrolment_df)
demographic_df = clean_state_names(demographic_df)
biometric_df = clean_state_names(biometric_df)

print("State name standardization complete.")


# # State‑Year Level Aggregation (Baseline Metrics)
# 
# The raw UIDAI datasets are recorded at a daily / transactional level.
# However, Aadhaar service stress and exclusion risk are structural, regional issues.
# To make datasets comparable and merge‑ready, we aggregate all indicators to a:
# 
# State × Year granularity
# 
# This ensures:
# - Consistent time resolution
# - Correct state‑level comparison
# - Clean downstream risk calculations

# ### 🧮 Aggregation Logic
# 
# | Dataset      | What We Aggregate                  | Why It Matters                                              |
# |--------------|------------------------------------|-------------------------------------------------------------|
# | Enrolment    | Age‑wise enrolment counts           | Captures enrolment volume and baseline population coverage  |
# | Demographic  | Age‑wise demographic updates        | Measures demand for demographic corrections and updates     |
# | Biometric    | Age‑wise biometric updates          | Indicates biometric refresh stress and re‑authentication   |
# 

# In[10]:


# -------------------------------
# Enrolment: State-Year Aggregation
# -------------------------------
enrolment_state_year = (
    enrolment_df
    .groupby(['state', 'year'], as_index=False)
    .agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    })
)

print("Enrolment (state-year) shape:", enrolment_state_year.shape)
enrolment_state_year.head()


# In[11]:


# ---------------------------------
# Demographic Updates: State-Year
# ---------------------------------
demographic_state_year = (
    demographic_df
    .groupby(['state', 'year'], as_index=False)
    .agg({
        'demo_age_5_17': 'sum',
        'demo_age_17_': 'sum'
    })
)

print("Demographic (state-year) shape:", demographic_state_year.shape)
demographic_state_year.head()


# In[12]:


# --------------------------------
# Biometric Updates: State-Year
# --------------------------------
biometric_state_year = (
    biometric_df
    .groupby(['state', 'year'], as_index=False)
    .agg({
        'bio_age_5_17': 'sum',
        'bio_age_17_': 'sum'
    })
)

print("Biometric (state-year) shape:", biometric_state_year.shape)
biometric_state_year.head()


# # Dataset Merging & Feature Construction
# 
# ## Why this step is critical
# So far, we have three clean, state‑year aggregated datasets:
# - Enrolment volume (baseline population)
# - Demographic update activity
# - Biometric update activity
# 
# However, Aadhaar exclusion risk is not driven by a single dataset.
# It emerges from the interaction between enrolment scale and update stress.
# 
# This step ensures:
# - One row = one State × Year
# - No duplicate states
# - No missing values
# 
# All metrics aligned and comparable

# In[13]:


# -----------------------------------
# Merge Enrolment + Demographic Updates
# -----------------------------------
master_df = pd.merge(
    enrolment_state_year,
    demographic_state_year,
    on=['state', 'year'],
    how='left'
)

# -----------------------------------
# Merge Biometric Updates
# -----------------------------------
master_df = pd.merge(
    master_df,
    biometric_state_year,
    on=['state', 'year'],
    how='left'
)

print("Master dataset shape:", master_df.shape)
master_df.head()


# In[14]:


# Columns related to update activity
update_cols = [
    'demo_age_5_17', 'demo_age_17_',
    'bio_age_5_17', 'bio_age_17_'
]

# Replace NaNs with 0 (no updates recorded)
master_df[update_cols] = master_df[update_cols].fillna(0)

master_df.isna().sum()


# In[15]:


master_df['total_enrolment'] = (
    master_df['age_0_5'] +
    master_df['age_5_17'] +
    master_df['age_18_greater']
)


# ### Total Enrolment (Baseline Population)
# 
# This metric captures the **scale of Aadhaar coverage** in each state-year.
# It represents the total population interacting with Aadhaar services.
# 
# Why this matters:
# - Larger enrolment bases naturally generate higher update demand
# - Serves as the **denominator** for all rate-based risk indicators
# - Normalizes update volumes across states of different sizes
# 

# In[16]:


# -----------------------------------
# Total Enrolment (Baseline Population)
# -----------------------------------
master_df['total_enrolment'] = (
    master_df['age_0_5'] +
    master_df['age_5_17'] +
    master_df['age_18_greater']
)


# In[17]:


# -----------------------------------
# Demographic Update Rate
# -----------------------------------
master_df['demographic_update_rate'] = (
    (master_df['demo_age_5_17'] + master_df['demo_age_17_']) /
    master_df['total_enrolment']
)


# In[18]:


# -----------------------------------
# Biometric Update Rate
# -----------------------------------
master_df['biometric_update_rate'] = (
    (master_df['bio_age_5_17'] + master_df['bio_age_17_']) /
    master_df['total_enrolment']
)


# In[19]:


# -----------------------------------
# Child Transition Risk
# -----------------------------------
master_df['child_transition_risk'] = (
    master_df['bio_age_5_17'] /
    master_df['age_5_17']
)


# In[20]:


# -----------------------------------
# Defensive Cleaning of Risk Metrics
# -----------------------------------
risk_cols = [
    'demographic_update_rate',
    'biometric_update_rate',
    'child_transition_risk'
]

master_df[risk_cols] = (
    master_df[risk_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

# Sanity check
master_df[risk_cols].describe()


# Aadhaar exclusion risk is not evenly distributed — it is driven by extreme biometric stress and child transition failures concentrated in specific states, necessitating a composite, normalized risk index.
# At this point, we have transformed raw UIDAI counts into
# **interpretable, defensively-cleaned risk indicators**.

# ## Normalization & Aadhaar Exclusion Risk Index Construction
# 
# ## 🧠 Why Normalization Is Critical
# 
# The three risk indicators we constructed in **Step 7** are not directly comparable:
# 
# | Metric | Typical Range | Interpretation |
# |------|--------------|----------------|
# | **Demographic Update Rate** | ~0–80 | Administrative correction stress |
# | **Biometric Update Rate** | ~0–95 | Biometric refresh demand |
# | **Child Transition Risk** | Can exceed 100 | High-risk exclusion pathway |
# 
# If we combine these indicators **without normalization**, metrics with larger numeric ranges would dominate the composite index unfairly.
# 
# ### 👉 Normalization ensures:
# 
# - **Equal comparability** across all indicators  
# - **Stable and meaningful weighting**  
# - **Transparent interpretation** for policymakers and judges  
# 
# 

# ## Min-Max Normalization (0–1 Scale)
# 
# We use **Min-Max Scaling**, the most interpretable normalization method for policy indices:
# 
# $$
# X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
# $$
# 
# ### Why Min-Max?
# 
# - **Preserves relative differences** between observations  
# - **Easy to explain** to non-technical stakeholders  
# - **Keeps the final index bounded** between 0 and 1  
# 
# 

# In[21]:


# ---------------------------------------
# Risk metrics to normalize
# ---------------------------------------
risk_cols = [
    'demographic_update_rate',
    'biometric_update_rate',
    'child_transition_risk'
]

# ---------------------------------------
# Min-Max Normalization
# ---------------------------------------
for col in risk_cols:
    master_df[col + '_norm'] = (
        (master_df[col] - master_df[col].min()) /
        (master_df[col].max() - master_df[col].min())
    )

# ---------------------------------------
# Invert Child Transition Risk
# Higher child biometric failure = higher exclusion risk
# ---------------------------------------
master_df['child_transition_risk_norm'] = (
    1 - master_df['child_transition_risk_norm']
)

# Preview normalized values
master_df[
    ['state', 'year'] +
    [c for c in master_df.columns if c.endswith('_norm')]
].head()


# ### Interpretation After Normalization
# 
# | Normalized Metric Value | Meaning |
# |------------------------|---------|
# | **Value close to 1** | High stress / high exclusion risk |
# | **Value close to 0** | Low stress / stable Aadhaar services |
# 
# ---
# 
# ### ⚠️ Important Design Choice
# 
# We **invert `child_transition_risk_norm`** because:
# 
# - Higher biometric failures among children indicate **worse outcomes**
# - The **final composite index should increase** with higher exclusion risk
# - This ensures **directional consistency** across all indicators  
# 
# 

# ### Weighted Aadhaar Exclusion Risk Index
# 
# We now combine the **normalized indicators** into a single **composite index**.
# 
# #### 🎯 Weighting Rationale
# 
# | Component | Weight | Reason |
# |---------|--------|--------|
# | **Demographic Update Stress** | 0.3 | Administrative correction burden |
# | **Biometric Update Stress** | 0.3 | Biometric infrastructure strain |
# | **Child Transition Risk** | 0.4 | Direct exclusion pathway |
# 
# 

# In[22]:


# ---------------------------------------
# Define weights
# ---------------------------------------
w_demo = 0.3
w_bio = 0.3
w_child = 0.4

# ---------------------------------------
# Aadhaar Exclusion Risk Index
# ---------------------------------------
master_df['aadhaar_exclusion_risk_index'] = (
    w_demo * master_df['demographic_update_rate_norm'] +
    w_bio * master_df['biometric_update_rate_norm'] +
    w_child * master_df['child_transition_risk_norm']
)

# ---------------------------------------
# Final normalization (0–1)
# ---------------------------------------
master_df['aadhaar_exclusion_risk_index'] = (
    (master_df['aadhaar_exclusion_risk_index'] -
     master_df['aadhaar_exclusion_risk_index'].min()) /
    (master_df['aadhaar_exclusion_risk_index'].max() -
     master_df['aadhaar_exclusion_risk_index'].min())
)

master_df[['state', 'year', 'aadhaar_exclusion_risk_index']].head()


# In[23]:


master_df['aadhaar_exclusion_risk_index'].describe()


# ## State-Level Risk Ranking & Interpretation
# 
# ### 🎯 Purpose
# 
# At this stage, we convert the **composite Aadhaar Exclusion Risk Index** into a **policy-ready ranking** that:
# 
# - Identifies **high-risk states**
# - Enables **comparative analysis**
# - Supports **targeted interventions**

# In[24]:


# --------------------------------------
# State-Year Risk Ranking
# --------------------------------------
risk_ranking = (
    master_df
    .sort_values('aadhaar_exclusion_risk_index', ascending=False)
    .reset_index(drop=True)
)

risk_ranking[['state', 'year', 'aadhaar_exclusion_risk_index']].head(10)


# In[25]:


# Top 10 highest-risk state-year observations
top_high_risk = risk_ranking.head(10)
top_high_risk


# In[26]:


# Bottom 10 lowest-risk state-year observations
lowest_risk = risk_ranking.tail(10)
lowest_risk


# ## Distribution of Aadhaar Exclusion Risk Index

# In[27]:


plt.figure(figsize=(10, 6))
sns.histplot(
    master_df['aadhaar_exclusion_risk_index'],
    bins=12,
    kde=True,
    color='steelblue'
)

plt.title("Distribution of Aadhaar Exclusion Risk Index", fontsize=14)
plt.xlabel("Exclusion Risk Index (0 = Low, 1 = High)")
plt.ylabel("Number of State-Year Observations")
plt.show()


# ### 📊 Interpretation: Distribution of Aadhaar Exclusion Risk Index
# 
# - The risk index is **concentrated around mid to high values (≈ 0.5–0.7)**, indicating that most states face **moderate exclusion risk**.
# - A **small number of states lie at the extreme high end (≥ 0.8)**, representing **critical risk hotspots** that need urgent policy attention.
# - Very **few states show low risk (≤ 0.2)**, suggesting that Aadhaar exclusion vulnerability is **widespread rather than isolated**.
# - The smooth, near‑bell‑shaped distribution implies **systemic structural factors**, not random outliers, are driving exclusion risk.
# 
# **Conclusion:** Aadhaar exclusion risk is a **nationwide concern**, with clear prioritization needed for high‑risk states rather than isolated interventions.
# 

#  ## Top High‑Risk States (Policy‑Critical View)

# In[28]:


top_risk_states = (
    master_df
    .groupby('state', as_index=False)['aadhaar_exclusion_risk_index']
    .mean()
    .sort_values('aadhaar_exclusion_risk_index', ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_risk_states,
    x='aadhaar_exclusion_risk_index',
    y='state',
    hue='state',
    palette='Reds_r',
    legend=False
)

plt.title("Top 10 States by Aadhaar Exclusion Risk", fontsize=14)
plt.xlabel("Average Exclusion Risk Index")
plt.ylabel("State")
plt.tight_layout()
plt.show()


# ### 📊 Insights from Top 10 States by Aadhaar Exclusion Risk
# 
# - The chart highlights **states with the highest average Aadhaar Exclusion Risk Index**, indicating regions where citizens are more vulnerable to service exclusion.
# - **Chhattisgarh and Chandigarh** emerge as the highest‑risk regions, suggesting sustained pressure from enrolment updates, biometric refresh, or child transition gaps.
# - Several **administratively strong states** (e.g., Kerala, Maharashtra, Tamil Nadu) still appear in the top‑risk list, showing that exclusion risk is **not only a function of capacity**, but also of update intensity and population dynamics.
# - The relatively **narrow spread of risk values** among the top 10 suggests systemic, nationwide challenges rather than isolated state failures.
# - These states should be **priority targets for policy intervention**, focused biometric update drives, and transition‑age Aadhaar awareness programs.
# 
# 🔍 *Overall, the ranking validates the Exclusion Risk Index as a meaningful tool for identifying high‑impact regions requiring targeted Aadhaar service strengthening.*
# 

# ## Component Contribution Analysis

# In[29]:


risk_components = master_df[[
    'demographic_update_rate_norm',
    'biometric_update_rate_norm',
    'child_transition_risk_norm'
]]

plt.figure(figsize=(10, 6))
sns.boxplot(data=risk_components)

plt.title("Normalized Risk Component Distribution", fontsize=14)
plt.ylabel("Normalized Value (0–1)")
plt.xticks(
    ticks=[0, 1, 2],
    labels=[
        'Demographic Update Risk',
        'Biometric Update Risk',
        'Child Transition Risk'
    ]
)
plt.show()


# ### 📦 Interpretation: Normalized Risk Component Distributions
# 
# This boxplot compares the **normalized distributions (0–1 scale)** of the three core **Aadhaar exclusion risk components** across states.
# 
# - **Child Transition Risk** shows consistently high values, indicating that biometric update failures among children transitioning to adulthood are the **dominant exclusion driver** in most states.
# 
# - **Biometric Update Risk** exhibits a **moderate spread**, suggesting uneven biometric re-enrolment pressure across regions.
# 
# - **Demographic Update Risk** remains **comparatively low and stable**, implying demographic corrections are **less volatile contributors** to exclusion risk.
# 
# ### 🔑 Key Insight
# 
# Aadhaar exclusion risk is **primarily driven by child biometric transition stress**, reinforcing the need for **targeted interventions during age-linked biometric updates**.
# 
# 

# ## Risk Index vs Enrolment Scale (Sanity Check)

# In[30]:


plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=master_df,
    x='total_enrolment',
    y='aadhaar_exclusion_risk_index',
    alpha=0.7
)

plt.title("Exclusion Risk vs Total Enrolment", fontsize=14)
plt.xlabel("Total Enrolment")
plt.ylabel("Exclusion Risk Index")
plt.show()


# ### 📊 Exclusion Risk vs Total Enrolment — Key Insight
# 
# This scatter plot analyzes the relationship between the **Aadhaar Exclusion Risk Index** and **total enrolment** at the state-year level.
# 
# #### Key Observations
# 
# - No strong linear relationship exists between enrolment size and exclusion risk.
# - Several **low-enrolment states show high exclusion risk**, driven by update stress and transition bottlenecks.
# - **High-enrolment states exhibit moderate to varied risk**, reflecting stronger administrative capacity.
# 
# #### 🏛️ Policy Implication
# 
# Aadhaar exclusion is a **systemic service issue**, not merely a function of population size.  
# Targeted interventions should prioritize **process efficiency and update infrastructure**, particularly in **smaller or high-risk states**.
# 
# 

# ## 🏆 Final Aadhaar Exclusion Risk Ranking (State‑Level)
# 
# The table below presents the **final state‑level ranking** based on the average Aadhaar Exclusion Risk Index.
# Higher values indicate **greater vulnerability to Aadhaar service exclusion**.
# 
# > This ranking aggregates exclusion risk across enrolment scale, demographic update stress,
# > biometric update pressure, and child transition failures.
# 
# 

# In[31]:


final_ranking = (
    master_df
    .groupby('state', as_index=False)['aadhaar_exclusion_risk_index']
    .mean()
    .sort_values('aadhaar_exclusion_risk_index', ascending=False)
)

final_ranking.style.hide(axis="index")


# **Interpretation:**
# - States at the top of this ranking require **immediate policy attention**.
# - The index enables **prioritization of resources**, not one‑size‑fits‑all interventions.
# 

# ## 🧠 Policy‑Ready Summary
# 
# This analysis reveals that Aadhaar exclusion risk is **structural, persistent, and unevenly distributed**
# across Indian states.
# 
# ### Key Findings
# - Aadhaar exclusion risk is **widespread**, with most states exhibiting **moderate to high risk**.
# - **Child biometric transition failures** emerge as the **single most dominant risk driver**.
# - Exclusion risk is **not strongly correlated with enrolment volume**, indicating that
#   administrative capacity and update infrastructure matter more than population size.
# - Even administratively strong states can experience high risk due to **update intensity and scale effects**.
# 
# ### Policy Implications
# - Shift from reactive correction to **preventive Aadhaar service design**.
# - Prioritize **age‑linked biometric update campaigns**, especially for children turning 18.
# - Deploy **mobile biometric units** in high‑risk, low‑capacity states.
# - Use the Exclusion Risk Index as a **monitoring dashboard**, updated annually.
# 
# This index enables **data‑driven governance**, allowing UIDAI and policymakers to
# identify, rank, and intervene in high‑risk regions proactively.
# 

# # 🎯 Conclusion & Hackathon Abstract
# 
# This project introduces a **novel, data‑driven Aadhaar Exclusion Risk Index** that quantifies
# state‑level vulnerability to Aadhaar service exclusion using enrolment, demographic updates,
# biometric refresh activity, and child transition risk.
# 
# By harmonizing multiple UIDAI datasets and engineering interpretable risk indicators,
# the index moves beyond raw counts to capture **systemic service stress**.
# Our findings show that Aadhaar exclusion is not merely a population‑scale issue,
# but a function of **update capacity, biometric refresh pressure, and lifecycle transitions**.
# 
# The proposed index provides a **scalable, policy‑ready framework** for:
# - Identifying high‑risk states
# - Prioritizing Aadhaar service interventions
# - Monitoring exclusion risk over time
# 
# This approach supports UIDAI’s mission of **inclusive digital identity** by enabling
# targeted, preventive, and evidence‑based governance.
# 
