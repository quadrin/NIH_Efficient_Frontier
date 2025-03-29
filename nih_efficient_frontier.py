import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data

def load_data():
    funding = pd.read_csv("https://raw.githubusercontent.com/quadrin/NIH_Efficient_Frontier/refs/heads/main/NIH_Funding_Since_1999.csv")
    yll = pd.read_csv("https://raw.githubusercontent.com/quadrin/NIH_Efficient_Frontier/refs/heads/main/CDC%20WONDER%20-%20Underlying%20Cause%20of%20Death.csv")
    return funding, yll

funding_df, yll_df = load_data()

# --- Melt and clean funding data ---
funding_long = funding_df.melt(id_vars="Fiscal Years", var_name="Year", value_name="Funding")
funding_long.rename(columns={"Fiscal Years": "Institute"}, inplace=True)
funding_long["Year"] = funding_long["Year"].astype(int)
funding_long["Funding"] = pd.to_numeric(funding_long["Funding"], errors="coerce")
funding_long.dropna(subset=["Funding"], inplace=True)

# --- Clean YLL data and map to NIH institutes ---
icd_map = {
    "Neoplasms": "NCI",
    "Diseases of the circulatory system": "NHLBI",
    "Diseases of the respiratory system": "NHLBI",
    "Endocrine, nutritional and metabolic diseases": "NIDDK",
    "Diseases of the nervous system": "NINDS",
    "Mental and behavioural disorders": "NIMH",
    "Diseases of the genitourinary system": "NIDDK",
    "Diseases of the digestive system": "NIDDK",
    "Diseases of the musculoskeletal system and connective tissue": "NIAMS",
    "Diseases of the skin and subcutaneous tissue": "NIAMS",
    "Certain infectious and parasitic diseases": "NIAID",
    "Diseases of the blood and blood-forming organs": "NHLBI",
    "Diseases of the eye and adnexa": "NEI",
    "Diseases of the ear and mastoid process": "NIDCD",
    "Congenital malformations, deformations and chromosomal abnormalities": "NICHD",
    "Certain conditions originating in the perinatal period": "NICHD",
    "Pregnancy, childbirth and the puerperium": "NICHD",
    "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified": "NINDS",
}

yll_df["Population"] = pd.to_numeric(yll_df["Population"], errors="coerce")
pop_2005 = yll_df[yll_df["Year"] == 2005].groupby("ICD Chapter")["Population"].mean().dropna().to_dict()
yll_df["Population_2005"] = yll_df["ICD Chapter"].map(pop_2005)
yll_df["Norm_Burden"] = yll_df["Deaths"] / yll_df["Population_2005"]
yll_df.dropna(subset=["Norm_Burden"], inplace=True)
yll_df["Institute"] = yll_df["ICD Chapter"].map(icd_map)
yll_df.dropna(subset=["Institute"], inplace=True)
burden_normalized = yll_df.groupby(["Year", "Institute"])["Norm_Burden"].sum().reset_index().rename(columns={"Norm_Burden": "Burden"})

# --- Calculate return using lag window ---
results = []
for lag in range(9, 17):
    for window in range(1, 6):
        for year in range(1999, 2021 - lag - window):
            current_funding = funding_long[funding_long["Year"] == year].copy()
            impact_years = list(range(year + lag, year + lag + window + 1))
            burden_window = burden_normalized[burden_normalized["Year"].isin(impact_years)]
            burden_avg = burden_window.groupby("Institute")["Burden"].mean().reset_index()
            burden_prev = burden_normalized[burden_normalized["Year"] == (year + lag - 1)]
            delta = burden_prev.merge(burden_avg, on="Institute", suffixes=("_prev", "_future"))
            delta["Burden_Change"] = delta["Burden_prev"] - delta["Burden_future"]
            merged = delta.merge(current_funding, on="Institute")
            merged["Return"] = merged["Burden_Change"] / merged["Funding"]
            merged["Lag"] = lag
            merged["Start_Year"] = year
            results.append(merged[["Institute", "Start_Year", "Lag", "Return"]])

roi_df = pd.concat(results, ignore_index=True)
roi_stats = roi_df.groupby(["Institute", "Lag"])["Return"].agg(Mean_Return="mean", Std_Dev="std").reset_index()
roi_stats["Stability_Score"] = roi_stats["Mean_Return"] / roi_stats["Std_Dev"]

# --- Streamlit App ---
st.title("NIH Efficient Frontier Explorer")
st.markdown("""
Explore the lag-adjusted return on NIH investments using CDC burden data. 
""")

# Sidebar Controls
st.sidebar.header("Filters")
institutes = sorted(roi_stats["Institute"].unique())
selected_institutes = st.sidebar.multiselect("Select Institutes", options=institutes, default=institutes)
lag_range = st.sidebar.slider("Lag Range", min_value=9, max_value=16, value=(9, 16))

# Filtered view
filtered = roi_stats[(roi_stats["Institute"].isin(selected_institutes)) & (roi_stats["Lag"].between(*lag_range))]

# Plot Mean Return
st.subheader("Mean Return vs Lag")
fig1, ax1 = plt.subplots(figsize=(10, 5))
for inst in filtered["Institute"].unique():
    df_inst = filtered[filtered["Institute"] == inst]
    ax1.plot(df_inst["Lag"], df_inst["Mean_Return"], marker='o', label=inst)
ax1.axhline(0, color='gray', linestyle='--')
ax1.set_xlabel("Lag (Years)")
ax1.set_ylabel("Mean Return")
ax1.set_title("Mean Return by Lag")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True)
st.pyplot(fig1)

# Plot Risk
st.subheader("Risk (Standard Deviation) vs Lag")
fig2, ax2 = plt.subplots(figsize=(10, 5))
for inst in filtered["Institute"].unique():
    df_inst = filtered[filtered["Institute"] == inst]
    ax2.plot(df_inst["Lag"], df_inst["Std_Dev"], marker='s', label=inst)
ax2.set_xlabel("Lag (Years)")
ax2.set_ylabel("Standard Deviation")
ax2.set_title("Risk by Lag")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True)
st.pyplot(fig2)

# Download button
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered.to_csv(index=False),
    file_name="filtered_roi_stats.csv",
    mime="text/csv"
)  
