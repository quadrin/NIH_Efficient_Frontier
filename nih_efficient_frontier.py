import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from GitHub raw URLs
base_url = "https://raw.githubusercontent.com/quadrin/NIH_Efficient_Frontier/main/"
dfs = {
    "total_deaths": pd.read_csv(base_url + "CDC%20WONDER%20-%20Total%20Deaths.csv"),
    "underlying_cause": pd.read_csv(base_url + "CDC%20WONDER%20-%20Underlying%20Cause%20of%20Death.csv"),
    "nih_funding": pd.read_csv(base_url + "NIH_Funding_Since_1999.csv"),
    "gdp_growth": pd.read_csv(base_url + "US%20GDP%20Growth%20(%25)%20YoY.csv")
}

# Placeholder sensitivity dataset: replace with actual logic from joined/processed data
# For demo, use underlying_cause
sensitivity_df = dfs["underlying_cause"]

# Sidebar controls
st.sidebar.header("Controls")
selected_institutes = st.sidebar.multiselect(
    "Select NIH Institutes to Display",
    options=sensitivity_df["Institute"].unique() if "Institute" in sensitivity_df else [],
    default=sensitivity_df["Institute"].unique() if "Institute" in sensitivity_df else []
)

lag_min, lag_max = st.sidebar.slider("Select Lag Range (Years)", min_value=5, max_value=15, value=(5, 15))

# Filter data
if "Institute" in sensitivity_df:
    filtered_df = sensitivity_df[
        (sensitivity_df["Institute"].isin(selected_institutes)) &
        (sensitivity_df["Lag"].between(lag_min, lag_max))
    ]

    # Pivot for heatmap
    heatmap_data = filtered_df.pivot(index="Institute", columns="Lag", values="Correlation_with_Burden")

    # Display heatmap
    st.title("NIH Funding vs Healthcare Burden Sensitivity Analysis")
    st.markdown("""
    Use the sliders and multiselect to explore how the correlation between NIH funding and downstream healthcare burden 
    changes based on lag assumptions.
    """)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, center=0, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax, 
                cbar_kws={'label': 'Correlation'})
    plt.title("Correlation Heatmap: NIH Funding vs Burden by Lag")
    st.pyplot(fig)

    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_sensitivity_data.csv",
        mime="text/csv"
    )
else:
    st.warning("The uploaded dataset does not include 'Institute' and 'Lag' columns. Please preprocess accordingly.")
