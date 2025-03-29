# Streamlit App to Analyze NIH Efficiency Using Bisias Framework
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

@st.cache_data

def load_data():
    funding = pd.read_csv("https://raw.githubusercontent.com/quadrin/NIH_Efficient_Frontier/refs/heads/main/NIH_Funding_Since_1999.csv")
    yll = pd.read_csv("https://raw.githubusercontent.com/quadrin/NIH_Efficient_Frontier/refs/heads/main/CDC%20WONDER%20-%20Underlying%20Cause%20of%20Death.csv")
    return funding, yll

funding_df, yll_df = load_data()

# ICD Chapter -> NIH Institute
icd_to_nih = {
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

# Normalize and map YLL data
yll_df["Population"] = pd.to_numeric(yll_df["Population"], errors="coerce")
pop_2005 = yll_df[yll_df["Year"] == 2005].groupby("ICD Chapter")["Population"].mean().dropna().to_dict()
yll_df["Population_2005"] = yll_df["ICD Chapter"].map(pop_2005)
yll_df["Norm_Burden"] = yll_df["Deaths"] / yll_df["Population_2005"]
yll_df.dropna(subset=["Norm_Burden"], inplace=True)
yll_df["Institute"] = yll_df["ICD Chapter"].map(icd_to_nih)
yll_df.dropna(subset=["Institute"], inplace=True)
burden_grouped = yll_df.groupby(["Year", "Institute"])["Norm_Burden"].sum().reset_index().rename(columns={"Norm_Burden": "Burden"})

# Reshape and clean funding data
funding_long = funding_df.melt(id_vars="Fiscal Years", var_name="Year", value_name="Funding")
funding_long.rename(columns={"Fiscal Years": "Institute"}, inplace=True)
funding_long["Year"] = funding_long["Year"].astype(int)
funding_long["Funding"] = pd.to_numeric(funding_long["Funding"], errors="coerce")
funding_long.dropna(subset=["Funding"], inplace=True)
funding_grouped = funding_long.groupby(["Year", "Institute"])["Funding"].sum().reset_index()

# Calculate ROI matrix for a specific lag and window
def compute_roi_matrix(lag=12, window=3):
    records = []
    for year in range(1999, 2021 - lag - window):
        current_funding = funding_grouped[funding_grouped["Year"] == year].copy()
        impact_years = list(range(year + lag, year + lag + window + 1))
        burden_window = burden_grouped[burden_grouped["Year"].isin(impact_years)]
        burden_avg = burden_window.groupby("Institute")["Burden"].mean().reset_index()
        burden_prev = burden_grouped[burden_grouped["Year"] == (year + lag - 1)]
        delta = burden_prev.merge(burden_avg, on="Institute", suffixes=("_prev", "_future"))
        delta["Burden_Change"] = delta["Burden_prev"] - delta["Burden_future"]
        merged = delta.merge(current_funding, on="Institute")
        merged["Return"] = merged["Burden_Change"] / merged["Funding"]
        merged["Start_Year"] = year
        records.append(merged[["Institute", "Start_Year", "Return"]])
    df = pd.concat(records)
    pivot = df.pivot(index="Start_Year", columns="Institute", values="Return").dropna(axis=1, how="any")
    return pivot

# Efficient Frontier Optimization
def efficient_frontier(roi_matrix):
    mu = roi_matrix.mean().values
    Sigma = roi_matrix.cov().values
    n = len(mu)

    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    ret = mu @ w

    prob = cp.Problem(cp.Maximize(ret - 0.1 * risk),
                      [cp.sum(w) == 1, w >= 0])
    prob.solve()

    return roi_matrix.columns, w.value, mu, Sigma

# Streamlit UI
st.title("NIH Efficient Frontier Explorer - Institute Level")
st.markdown("Explore return on NIH investments by Institute using efficient frontier optimization.")

roi_matrix = compute_roi_matrix(lag=12, window=3)
labels, weights, mean_returns, cov_matrix = efficient_frontier(roi_matrix)

st.subheader("Efficient Portfolio Weights")
portfolio_df = pd.DataFrame({"Institute": labels, "Weight": weights})
st.dataframe(portfolio_df.sort_values(by="Weight", ascending=False))

st.subheader("Efficient Frontier Scatter Plot")
fig, ax = plt.subplots()
risks = np.sqrt(np.diag(cov_matrix))
ax.scatter(risks, mean_returns)
for i, label in enumerate(labels):
    ax.annotate(label, (risks[i], mean_returns[i]))
ax.set_xlabel("Risk (Std Dev)")
ax.set_ylabel("Mean Return")
ax.set_title("Risk vs Return by NIH Institute")
st.pyplot(fig)

st.download_button(
    label="Download Efficient Portfolio as CSV",
    data=portfolio_df.to_csv(index=False),
    file_name="efficient_portfolio_by_institute.csv",
    mime="text/csv"
)
