import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data

def load_data():
    base = "https://raw.githubusercontent.com/quadrin/NIH_Efficient_Frontier/main/"
    funding = pd.read_csv(base + "NIH_Funding_Since_1999.csv")
    yll = pd.read_csv(base + "CDC%20WONDER%20-%20Underlying%20Cause%20of%20Death.csv")
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

yll_agg = yll_df.groupby(["ICD Chapter", "Year"])['Deaths'].sum().reset_index()
yll_agg["Institute"] = yll_agg["ICD Chapter"].map(icd_map)
yll_agg.dropna(subset=["Institute"], inplace=True)
burden = yll_agg.groupby(["Institute", "Year"])['Deaths'].sum().reset_index().rename(columns={"Deaths": "Burden"})

# --- Calculate return with lag ---
funding_long["Impact_Year"] = funding_long["Year"] + 10
df = funding_long.merge(burden, left_on=["Institute", "Impact_Year"], right_on=["Institute", "Year"], suffixes=("_Funding", "_Burden"))
df.sort_values(by=["Institute", "Impact_Year"], inplace=True)
df["Prev_Burden"] = df.groupby("Institute")["Burden"].shift(1)
df["Burden_Change"] = df["Prev_Burden"] - df["Burden"]
df["Return"] = df["Burden_Change"] / df["Funding"]
df.dropna(subset=["Return"], inplace=True)

# --- Portfolio: Risk vs Return ---
returns_matrix = df.pivot(index="Impact_Year", columns="Institute", values="Return").dropna(axis=1)
mean_returns = returns_matrix.mean()
risk = returns_matrix.std()
summary = pd.DataFrame({"Mean Return": mean_returns, "Risk": risk}).sort_values("Mean Return", ascending=False)

# --- Streamlit App ---
st.title("NIH Efficient Frontier Explorer")

st.markdown("""
This app calculates and visualizes the risk-return profile of NIH institutes based on historical funding and disease burden
(from CDC WONDER). A 10-year lag is applied between investment and outcome.
""")

st.subheader("NIH Institute Risk vs Return")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(summary["Risk"], summary["Mean Return"], s=100)
for label in summary.index:
    ax.annotate(label, (summary.loc[label, "Risk"], summary.loc[label, "Mean Return"]))
ax.set_xlabel("Risk (Standard Deviation of Return)")
ax.set_ylabel("Mean Return (Burden Reduction per $)")
ax.set_title("NIH Portfolio Risk vs Return")
ax.grid(True)
st.pyplot(fig)

st.subheader("Raw Portfolio Table")
st.dataframe(summary.reset_index())
