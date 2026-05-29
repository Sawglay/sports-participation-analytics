import streamlit as st
import pandas as pd
import numpy as np
import statistics

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy.stats import norm
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─────────────────────────────────────────────
   ROOT DESIGN SYSTEM
───────────────────────────────────────────── */
:root {
    --bg-primary:      #0a0f1c;
    --bg-secondary:    #111827;
    --bg-card:         rgba(17, 24, 39, 0.82);
    --bg-hover:        rgba(30, 41, 59, 0.95);

    --accent-primary:  #4f8cff;
    --accent-secondary:#00c2a8;
    --accent-warning:  #f5b942;
    --accent-danger:   #ff6b6b;

    --text-primary:    #f3f4f6;
    --text-secondary:  #9ca3af;
    --text-muted:      #6b7280;

    --border:          rgba(255,255,255,0.08);
    --shadow:          0 8px 30px rgba(0,0,0,0.32);

    --radius-sm:       10px;
    --radius-md:       16px;
    --radius-lg:       22px;
}

/* ─────────────────────────────────────────────
   GLOBAL APP
───────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(circle at top left, rgba(79,140,255,0.12), transparent 35%),
        radial-gradient(circle at bottom right, rgba(0,194,168,0.08), transparent 30%),
        var(--bg-primary);

    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding: 2rem 3rem 3rem;
    max-width: 1600px;
}

/* ─────────────────────────────────────────────
   TYPOGRAPHY
───────────────────────────────────────────── */
h1, h2, h3, h4, h5 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    font-weight: 700 !important;
}

p, li, span, label, div {
    font-family: 'Inter', sans-serif !important;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ─────────────────────────────────────────────
   SIDEBAR
───────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid var(--border);
    backdrop-filter: blur(14px);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--accent-primary) !important;
    font-weight: 700 !important;
}

/* ─────────────────────────────────────────────
   PROFESSIONAL DASHBOARD HEADER
───────────────────────────────────────────── */
.dashboard-banner {
    background:
        linear-gradient(145deg,
        rgba(17,24,39,0.95),
        rgba(15,23,42,0.92));

    border: 1px solid var(--border);
    border-radius: var(--radius-lg);

    padding: 2rem 2.5rem;
    margin-bottom: 2rem;

    box-shadow: var(--shadow);
    backdrop-filter: blur(14px);

    position: relative;
    overflow: hidden;
}

.dashboard-banner::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;

    width: 100%;
    height: 4px;

    background: linear-gradient(
        90deg,
        var(--accent-primary),
        var(--accent-secondary)
    );
}

.dashboard-banner h1 {
    font-size: 2.6rem !important;
    margin-bottom: 0.35rem;
}

.dashboard-banner p {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.01em;
}

/* ─────────────────────────────────────────────
   METRIC CARDS
───────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--bg-card);

    border: 1px solid var(--border);
    border-radius: var(--radius-md);

    padding: 1.2rem 1.4rem !important;

    backdrop-filter: blur(12px);
    box-shadow: var(--shadow);

    transition: all 0.25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(79,140,255,0.35);
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

/* ─────────────────────────────────────────────
   ALERT BOXES
───────────────────────────────────────────── */
.stAlert {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    backdrop-filter: blur(10px);
}

.stInfo {
    background: rgba(79,140,255,0.12) !important;
    border-left: 4px solid var(--accent-primary) !important;
}

.stSuccess {
    background: rgba(0,194,168,0.12) !important;
    border-left: 4px solid var(--accent-secondary) !important;
}

.stWarning {
    background: rgba(245,185,66,0.12) !important;
    border-left: 4px solid var(--accent-warning) !important;
}

.stError {
    background: rgba(255,107,107,0.12) !important;
    border-left: 4px solid var(--accent-danger) !important;
}

/* ─────────────────────────────────────────────
   TABS
───────────────────────────────────────────── */
[data-testid="stTabs"] {
    margin-top: 0.5rem;
}

[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;

    background: transparent;
    border-radius: 10px 10px 0 0;

    color: var(--text-secondary);
    font-size: 0.95rem !important;
    font-weight: 600;

    padding: 0.8rem 1.3rem;
    transition: all 0.2s ease;
}

[data-testid="stTabs"] button:hover {
    color: var(--text-primary);
    background: rgba(255,255,255,0.03);
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-primary) !important;

    border-bottom: 2px solid var(--accent-primary) !important;
    background: rgba(79,140,255,0.08);
}

/* ─────────────────────────────────────────────
   DATAFRAME
───────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow);
}

/* ─────────────────────────────────────────────
   BUTTONS
───────────────────────────────────────────── */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        135deg,
        var(--accent-primary),
        #3b82f6
    ) !important;

    color: white !important;

    border: none !important;
    border-radius: 12px !important;

    padding: 0.65rem 1.4rem !important;

    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;

    transition: all 0.25s ease;
    box-shadow: 0 4px 16px rgba(79,140,255,0.25);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.05);
}

/* ─────────────────────────────────────────────
   INPUTS
───────────────────────────────────────────── */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background: rgba(17,24,39,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

/* ─────────────────────────────────────────────
   SECTION DIVIDER
───────────────────────────────────────────── */
.section-divider {
    border-top: 1px solid var(--border);
    margin: 2rem 0 1.5rem;
    opacity: 0.7;
}

/* ─────────────────────────────────────────────
   SCROLLBAR
───────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.12);
    border-radius: 10px;
}

::-webkit-scrollbar-track {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="dash-banner">
  <h1>UNIVERSITY SPORT INVESTMENT ANALYTICS</h1>
  <p>Popularity · Stress · Physical Load · Machine Learning · CRISP-DM Framework &nbsp;|&nbsp;
     Siam University · Group 4 &nbsp;|&nbsp;
     <strong style="color:#f0c060">n = </strong> filtered responses
  </p>
</div>
""", unsafe_allow_html=True)

# st.title("Optimizing University Sport Investment Using Popularity And Stress Analysis")
st.info('Upload Your Cleaned Data')
uploaded_file = st.file_uploader('Upload Cleaned_Data.xlsx', type=['xlsx'])

if uploaded_file is None:
    st.write('Please upload your Cleaned_Data.xlsx file to continue.')
    st.stop()

df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
df = df.drop(columns=[c for c in df.columns if 'unnamed' in c.lower()])
df = df.rename(columns={'AcitvityLevel': 'ActivityLevel'})

# Removing 'Prefer Not To Say' from dataset
df = df[df['Gender'] != 'Prefer Not To Say']
df = df.reset_index(drop=True)

st.dataframe(df)
st.info('The dataset covers key lifestyle and academic variables that may influence student stress. "Prefer Not To Say" gender responses have been removed so that the gender analysis compares only Male and Female groups. All 128 remaining responses are used in the full analysis.')

st.header('Descriptive Statistics')

df_selected = df[['Stress_num', 'SportsHrs_num', 'Sleep_num', 'Height_num']]

#calculate and display stats for each column using a loop
for column in df_selected.columns:
    st.subheader(column + ' Statistics')
    data = df_selected[column]
    st.write('Count:', len(data))
    st.write('Mean:', round(np.mean(data), 2))
    st.write('Median:', round(np.median(data), 2))
    st.write('Mode:', statistics.mode(data))
    st.write('Standard Deviation:', round(np.std(data), 2))
    st.write('Min:', min(data))
    st.write('Max:', max(data))

#IQR and fence values for Stress
Q1 = np.percentile(df['Stress_num'], 25)
Q3 = np.percentile(df['Stress_num'], 75)
IQR = Q3 - Q1

st.subheader('Stress Level - IQR and Fence Values')
st.write('Q1:', Q1)
st.write('Q3:', Q3)
st.write('IQR:', IQR)
st.write('Lower Fence:', Q1 - 1.5 * IQR)
st.write('Upper Fence:', Q3 + 1.5 * IQR)
st.info('The IQR of 1.0 shows the middle 50% of stress scores falls between 3 and 4, indicating that most students experience moderate-to-high stress. The upper fence is 5.5, which exceeds the maximum possible score of 5 — meaning no students qualify as high-stress outliers; all high scores are genuine. The lower fence is 1.5, so only students scoring below 1.5 would be outliers on the low end. This confirms the stress distribution is real and not distorted by anomalies.')

st.subheader('Summary Table')
st.dataframe(df_selected.describe())
st.info('Summary: Stress scores average around 3.4 out of 5, indicating moderate-to-high stress across the sample. The median of 3 and mode of 3 confirm that the most common experience is mid-range stress. Sport hours average just 2.73 hours per week with a low median of 1.5, suggesting most students participate minimally. Sleep averages 6 hours per night, below the recommended 8 hours, indicating mild sleep deprivation is common across the sample.')


df['HighStress'] = (df['Stress_num'] >= 4).astype(int)

df['Workload_num'] = df['AcademicWorkload'].map({'Low': 1, 'Medium': 2, 'High': 3})
df['Activity_num'] = df['ActivityLevel'].map({'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4})
df['Gender_bin'] = (df['Gender'] == 'Female').astype(int)

st.header('Sport Popularity Analysis')
st.write('Which sports do the most students participate in?')

#bar chart - sport popularity
sport_counts = df['Sport'].value_counts().reset_index()
sport_counts.columns = ['Sport', 'Count']
sport_counts['Percentage'] = round(sport_counts['Count'] / len(df) * 100, 1)



fig_bar_sport = px.bar(sport_counts, x='Sport', y='Count',
                       color='Sport',
                       text='Percentage',
                       title='Sport Popularity - Number of Students per Sport')
fig_bar_sport.update_traces(texttemplate='%{text}%', textposition='outside')
st.plotly_chart(fig_bar_sport)

st.write('Key Finding: Badminton is the most popular sport with 59 students (43%).')
st.write('Basketball is second with 36 students (26%).')
st.write('Badminton + Basketball together cover 69% of all sport participants.')
st.write('Majority of participation is concentrated in just 2 sports.')
st.info('Interpretation: Sport participation is highly concentrated in two sports. Badminton alone accounts for 43% of students, making it the dominant activity on campus. Basketball follows at 26%. Every other sport individually accounts for less than 15% of students. This concentration pattern is critical for investment decisions — any resources directed at minority sports would benefit fewer than 1 in 7 students, making them poor value compared to investing in Badminton and Basketball infrastructure.')


st.header('Workload vs Stress Analysis')
st.write('RQ1: Does academic workload significantly increase student stress?')


#bar chart - average stress by workload
wl_stress = df.groupby('AcademicWorkload')['Stress_num'].mean().reindex(
    ['Low', 'Medium', 'High']).reset_index()
wl_stress.columns = ['AcademicWorkload', 'Average Stress']
fig_bar_wl = px.bar(wl_stress, x='AcademicWorkload', y='Average Stress',
                    color='AcademicWorkload',
                    title='Average Stress Level by Academic Workload')
st.plotly_chart(fig_bar_wl)
st.info('Interpretation: The bar chart shows a clear upward staircase pattern — average stress rises consistently from Low (3.0) to Medium (3.14) to High (4.08) workload groups. This visual trend strongly suggests a positive relationship between workload and stress. The jump from Medium to High is notably larger than from Low to Medium, indicating that high workload has a disproportionately strong effect on stress compared to moderate workload.')

#ANOVA statistical test
group_Low    = df[df['AcademicWorkload'] == 'Low']['Stress_num']
group_Medium = df[df['AcademicWorkload'] == 'Medium']['Stress_num']
group_High   = df[df['AcademicWorkload'] == 'High']['Stress_num']
f_stat, p_value_anova = stats.f_oneway(group_Low, group_Medium, group_High)

st.subheader('One-Way ANOVA - Workload vs Stress')
st.write('Low Workload    Mean Stress:', round(group_Low.mean(), 2))
st.write('Medium Workload Mean Stress:', round(group_Medium.mean(), 2))
st.write('High Workload   Mean Stress:', round(group_High.mean(), 2))
st.write('F-statistic:', round(f_stat, 4))
st.write('p-value    :', round(p_value_anova, 6))
if p_value_anova < 0.05:
    st.write('Result: SIGNIFICANT (p < 0.05)')
    st.write('Conclusion: Academic workload significantly increases student stress.')
    st.write('High workload students are 37% more stressed than low workload students.')
    st.write('This is the main stress driver confirmed by statistical testing.')
st.info('ANOVA Interpretation: The one-way ANOVA tests whether mean stress differs significantly across the three workload groups. The F-statistic of 10.33 and p-value of 0.00007 confirm the differences are statistically significant — far below the 0.05 threshold. This means workload group differences in stress did not occur by chance. With High workload students averaging 4.08 stress vs 3.0 for Low workload students, academic pressure is the strongest and most statistically reliable predictor of stress in this entire dataset.')


st.header('Sport vs Stress Analysis')
st.write('RQ2: Does sport participation reduce student stress?')
