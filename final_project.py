import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Tuberculosis Dashboard", page_icon="🩺", layout="wide")
st.title("🩺 Tuberculosis Chest X-ray Dataset Analysis")

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("tb_xray_dataset.csv")
    return df

df = load_data()
st.success("Dataset loaded successfully from tb_xray_dataset.csv")

# --- Download Raw Data Button ---
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Raw Data",
    data=csv_data,
    file_name="tb_xray_dataset.csv",
    mime="text/csv"
)

# --- Dataset Summary Metrics ---
temp_df = df.copy()
if "Patient_ID" in temp_df.columns:
    temp_df = temp_df.drop(columns=["Patient_ID"])

total_patients = temp_df.shape[0]
n_factors = temp_df.shape[1]
n_male = temp_df[temp_df["Gender"] == "Male"].shape[0] if "Gender" in temp_df.columns else 0
n_female = temp_df[temp_df["Gender"] == "Female"].shape[0] if "Gender" in temp_df.columns else 0
n_normal = temp_df[temp_df["Class"] == "Normal"].shape[0] if "Class" in temp_df.columns else 0
n_tb = temp_df[temp_df["Class"] == "Tuberculosis"].shape[0] if "Class" in temp_df.columns else 0

st.header("Basic Dataset Information")
st.subheader("Sample of Raw Data")
st.dataframe(df.head(10))

col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Total Patients", value=f"{total_patients:,}")
col2.metric(label="Number of Factors", value=f"{n_factors:,}")
col3.metric(label="Male / Female", value=f"{n_male:,} / {n_female:,}")
col4.metric(label="Normal / TB", value=f"{n_normal:,} / {n_tb:,}")

# --- Data Preparation for Chart Section ---
if "Patient_ID" in df.columns:
    df = df.drop(columns=["Patient_ID"])

df["Chest_Pain_Num"] = df["Chest_Pain"].map({"Yes": 1, "No": 0})
df["Night_Sweats_Num"] = df["Night_Sweats"].map({"Yes": 1, "No": 0})
df["Blood_in_Sputum_Num"] = df["Blood_in_Sputum"].map({"Yes": 1, "No": 0})
df["Smoking_History_Num"] = df["Smoking_History"].map({"Never": 0, "Former": 1, "Current": 2})
df["Previous_TB_History_Num"] = df["Previous_TB_History"].map({"No": 0, "Yes": 1})
df["Fever_Num"] = df["Fever"].map({"Mild": 1, "Moderate": 2, "High": 3})

# --- Summary Chart Dropdowns ---
VISUALS = {
    "Age Distribution": lambda df: px.histogram(df, x="Age", title="Age Distribution in Dataset"),
    "Gender Ratio": lambda df: px.pie(df, names="Gender", title="Gender Distribution"),
    "Mean Prevalence/Severity of Symptoms & Risk Factors": lambda df: px.pie(
        pd.DataFrame({
            "Symptom": [
                "Chest Pain", "Cough Severity", "Breathlessness", "Fatigue",
                "Weight Loss", "Fever", "Night Sweats", "Blood in Sputum",
                "Smoking History", "Previous TB History"
            ],
            "Mean Value": [
                df["Chest_Pain_Num"].mean(), df["Cough_Severity"].mean(),
                df["Breathlessness"].mean(), df["Fatigue"].mean(),
                df["Weight_Loss"].mean(), df["Fever_Num"].mean(),
                df["Night_Sweats_Num"].mean(), df["Blood_in_Sputum_Num"].mean(),
                df["Smoking_History_Num"].mean(), df["Previous_TB_History_Num"].mean()
            ]
        }),
        names="Symptom", values="Mean Value", title="Mean Symptom Prevalence/Severity"
    ),
    "TB Class Distribution": lambda df: px.histogram(
        df, x="Class", color="Class",
        category_orders={"Class": ["Normal", "Tuberculosis"]},
        title="TB vs Normal Case Distribution"
    ),
    "Age Distribution by TB Class": lambda df: px.histogram(
        df, x="Age", color="Class", marginal="rug", nbins=20, barmode="overlay",
        opacity=0.6, title="Age Distribution by TB Class"
    ),
}

st.sidebar.header("Summary Chart Selection")
selected = st.sidebar.multiselect(
    "Choose up to 2 charts to display",
    options=list(VISUALS.keys()),
    default=[list(VISUALS.keys())[0]],
    max_selections=2
)

if len(selected) == 1:
    chart = VISUALS[selected[0]](df)
    st.plotly_chart(chart, use_container_width=True)
elif len(selected) == 2:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(VISUALS[selected[0]](df), use_container_width=True)
    with col2:
        st.plotly_chart(VISUALS[selected[1]](df), use_container_width=True)

# --- Structured Q&A for Symptom & Risk Analysis and Top Symptom Combinations ---

# Prepare additional data for Q&A visualizations
df["Sputum_Production_Num"] = df["Sputum_Production"].map({"Low": 1, "Medium": 2, "High": 3})
df["Class_Num"] = df["Class"].map({"Normal": 0, "Tuberculosis": 1})
df["Symptom_Combination"] = (
    "F" + df["Fatigue"].astype(str) +
    "_S" + df["Sputum_Production_Num"].astype(str) +
    "_B" + df["Blood_in_Sputum_Num"].astype(str)
)
combo_tb_rate = (
    df.groupby("Symptom_Combination")["Class_Num"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)
df["Cough_Bin"] = pd.cut(df["Cough_Severity"], bins=[0, 2, 5, 8], labels=["Low", "Moderate", "High"])
df["Weight_Bin"] = pd.cut(df["Weight_Loss"], bins=[0, 5, 10, 20], labels=["Low", "Medium", "High"])
heat_data = df.pivot_table(
    values="Class_Num",
    index="Cough_Bin",
    columns="Weight_Bin",
    aggfunc="mean"
)
df["Smoking_Num"] = df["Smoking_History"].map({"No": 0, "Yes": 1})
df["Previously_Treated_for_TB_Num"] = df["Previous_TB_History"].map({"No": 0, "Yes": 1})
df["RiskCombo"] = (
    "S" + df["Smoking_Num"].astype(str) +
    "_P" + df["Previously_Treated_for_TB_Num"].astype(str) +
    "_F" + df["Fatigue"].astype(str) +
    "_C" + df["Cough_Severity"].astype(str) +
    "_Fev" + df["Fever"].astype(str)
)
tb_risk_combo = (
    df.groupby("RiskCombo")["Class_Num"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)

# For "Symptom Group Score" in Q7
symptom_components = [
    "Chest_Pain_Num", "Cough_Severity", "Breathlessness", "Fatigue",
    "Weight_Loss", "Fever_Num", "Night_Sweats_Num", "Blood_in_Sputum_Num"
]
df["Symptom_Score"] = df[symptom_components].sum(axis=1)

question_dict = {
    "Can we determine whether one has Tuberculosis by fever levels?": {
        "plot_code": lambda df: px.histogram(df, x='Fever', color='Class', barmode='group', title="Fever Level vs TB Class"),
        "answer": "Among the three fever levels, the Mild category has the highest number of TB cases, followed by High and then Moderate. This might be counterintuitive, as one might expect higher TB cases at high fever levels. It suggests fever intensity alone may not predict TB reliably."
    },
    "Does Smoking effect the chances of getting diagnosed with Tuberculosis?": {
        "plot_code": lambda df: px.histogram(df, x='Smoking_History', color='Class', barmode='group', title="Smoking History vs TB Class"),
        "answer": "Former and Current smokers show a slightly higher count of TB cases than Never smokers. While the Normal class is higher in all groups, the proportion of TB cases is marginally greater in smokers."
    },
    "Does the presence of blood in sputum increase TB risk?": {
        "plot_code": lambda df: px.bar(combo_tb_rate, x="Symptom_Combination", y="Class_Num", title="Top Symptom Combinations (Fatigue + Sputum + Blood) with High TB Rate"),
        "answer": "All listed combinations with B1 (blood in sputum present) rank among the highest proportions of TB. This suggests a strong association between blood in sputum and higher TB detection rates when accompanied by fever and sputum secretion level."
    },
    "Does the absence of blood in sputum (B0) still indicate a TB risk?": {
        "plot_code": lambda df: px.bar(combo_tb_rate, x="Symptom_Combination", y="Class_Num", title="Top Symptom Combinations (Fatigue + Sputum + Blood) with High TB Rate"),
        "answer": "Several combinations with B0 are present in the top ranks, though they tend to have slightly lower TB proportions compared to those with B1, indicating that TB can occur without blood in sputum, though with a lower likelihood among these combinations."
    },
    "Does higher cough severity always mean greater TB probability?": {
        "plot_code": lambda df: px.imshow(
            heat_data,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Probability of TB by Cough Severity and Weight Loss"
        ),
        "answer": "No. Moderate cough severity with low weight loss actually shows a slightly higher probability than high cough severity with the same weight loss."
    },
    "Does a history of previous Tuberculosis significantly raise the possibility of a patient to be diagnosed with Tuberculosis again?": {
        "plot_code": lambda df: px.bar(tb_risk_combo, x="RiskCombo", y="Class_Num", title="Top Smoking + Previous TB + Symptom Combinations with High TB Rate"),
        "answer": "Tuberculosis history when combined with other factors like smoking regularity, fatigue level, cough severity and Fever level shows that a prior episode is a substantial risk factor for future Tuberculosis."
    },
    "Is there any specific age group where Tuberculosis is more likely to appear?": {
        "plot_code": lambda df: px.scatter(
            df, x="Age", y="Symptom_Score", color="Class",
            title="Age vs Symptom Group Score by TB Class",
            labels={"Symptom_Score": "Symptom Group Score"}
        ),
        "answer": "The points are spread relatively evenly across all ages and symptom group scores, indicating that higher or lower symptom group scores are not concentrated in specific age ranges."
    },
    # If more questions are needed, add here...
}

# List of questions (update for up to 10 total)
question_list = list(question_dict.keys())

st.header("Detailed Symptom & Risk Analysis")
selected_question = st.selectbox(
    "Select a question to explore:",
    question_list
)

if selected_question:
    plot_func = question_dict[selected_question]["plot_code"]
    st.plotly_chart(plot_func(df), use_container_width=True)
    st.info(question_dict[selected_question]["answer"])

# Continue with other analyses or features as needed...
