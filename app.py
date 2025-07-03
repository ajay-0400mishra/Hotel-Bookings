# 1. Imports (all needed libraries)
import plotly.express as px
from wordcloud import WordCloud
from st_aggrid import AgGrid, GridOptionsBuilder
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns

# 2. (Optional) Add logo or banner
# from PIL import Image
# logo = Image.open("hotel_logo.png")
# st.sidebar.image(logo, width=120)

# 3. Streamlit config and title
st.set_page_config(page_title="Hotel Pricing Insights Dashboard", layout="wide")
st.title("AI‑Powered Hotel Pricing Insights Dashboard")

# 4. Load data
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_hotel_pricing_survey.csv")
df = load_data()

# 5. KPI calculation
kpi1 = df['Trust AI'].eq('Yes').mean() * 100
lead_time_map = {'Same day':0, '1–3 days':2, '4–7 days':5, '8–14 days':11, '15+ days':20}
kpi2 = df['Advance Booking Days'].map(lead_time_map).mean()
kpi3 = df['ADR Budget'].map({
    '<2000':1500,'2000–4000':3000,'4000–7000':5500,'7000–10000':8500,'>10000':12000}).mean()

l1, l2, l3 = st.columns(3)
l1.metric("AI-Trust (%)", f"{kpi1:0.1f} %")
l2.metric("Avg. Lead-Time (days)", f"{kpi2:0.1f}")
l3.metric("Mean ADR (₹)", f"{kpi3:,.0f}")

# 6. Sidebar for filters and tab selection
st.sidebar.header("GLOBAL FILTERS")
age_filter = st.sidebar.multiselect(
    "Age Group", options=df['Age Group'].unique(),
    default=df['Age Group'].unique()
)
hotel_filter = st.sidebar.multiselect(
    "Hotel Type", options=df['Preferred Hotel Type'].unique(),
    default=df['Preferred Hotel Type'].unique()
)
lead_filter = st.sidebar.slider(
    "Lead-time (days)",
    min_value=0, max_value=20, value=(0, 20)
)
df_filt = df.copy()
df_filt = df_filt[
    df_filt['Age Group'].isin(age_filter) &
    df_filt['Preferred Hotel Type'].isin(hotel_filter)
]
df_filt = df_filt[
    df_filt['Advance Booking Days'].map(lead_time_map).between(*lead_filter)
]

st.sidebar.header("Navigation")
tabs = st.sidebar.radio(
    "Go to", 
    (
        "Data Visualization", 
        "Classification", 
        "Clustering", 
        "Association Rules", 
        "Regression"
    ),
    key="main_tabs"  # Prevents duplication errors
)

# 7. Helper functions (needed for plotting, etc.)
def describe_plot(fig, caption):
    st.pyplot(fig)
    st.markdown(f"**Insight:** {caption}")

def encode_label(series):
    mapping = {"Yes": 1, "Maybe": 0, "No": 0}
    return series.map(mapping).fillna(0).astype(int)

# --------- Data Visualization Tab ---------
if tabs == "Data Visualization":
    st.header("Key Descriptive Insights")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df_filt, x="Age Group", ax=ax1)
    plt.xticks(rotation=45)
    describe_plot(fig1, "Age distribution shows majority of respondents are 25–44, ideal early‑career travelers.")

    fig2, ax2 = plt.subplots()
    sns.countplot(data=df_filt, x="Preferred Hotel Type", hue="Trust AI", ax=ax2)
    describe_plot(fig2, "Higher AI trust among city‑hotel lovers suggests urban travelers are more tech‑savvy.")

    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df_filt, x="Stay Length (Nights)", y="Annual Income", ax=ax3)
    describe_plot(fig3, "Longer stays correlate with higher incomes – premium opportunities for extended bundles.")

    fig4, ax4 = plt.subplots()
    sns.histplot(df_filt["Annual Income"], bins=50, kde=True, ax=ax4)
    describe_plot(fig4, "Income distribution is highly skewed right with visible outliers – pricing must handle extremes.")

    fig5, ax5 = plt.subplots()
    sns.countplot(data=df_filt, x="Use Price Comparison", ax=ax5)
    describe_plot(fig5, "Nearly half always use price comparators, validating need for price‑prediction transparency.")

    fig6, ax6 = plt.subplots()
    comp = df_filt.groupby("Advance Booking Days")["ADR Budget"].value_counts().unstack().fillna(0)
    comp.plot(kind='bar', stacked=True, ax=ax6)
    describe_plot(fig6, "Late bookers (<3 days) skew towards high ADR budgets – opportunity for last‑minute deals.")

    fig7, ax7 = plt.subplots()
    sns.countplot(data=df_filt, x="Subscription Willingness", hue="Refer Platform", ax=ax7)
    describe_plot(fig7, "Users willing to pay ₹199–₹299+ are most likely to refer, indicating sweet‑spot pricing plan.")

    fig8, ax8 = plt.subplots()
    sns.countplot(data=df_filt, x="Booking Device", ax=ax8)
    describe_plot(fig8, "Mobile dominates bookings, so mobile‑first UX is crucial.")

    fig9, ax9 = plt.subplots()
    sns.countplot(data=df_filt, x="Delayed Booking for Drop", hue="Trust AI", ax=ax9)
    describe_plot(fig9, "AI‑trusting users less likely to delay for price drops – market behaviour we can monetise.")

    fig10, ax10 = plt.subplots()
    sns.countplot(data=df_filt, x="Pay More for Sustainability", ax=ax10)
    describe_plot(fig10, "60% would pay premium for eco‑stays – integrate sustainability filter & pricing.")

    # Correlation Heatmap
    fig_corr, ax_corr = plt.subplots(figsize=(8,6))
    num_vars = df_filt.select_dtypes(exclude='object').corr()
    sns.heatmap(num_vars, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    describe_plot(fig_corr, "Correlation heat-map highlights which numeric factors move together.")

    # Word Cloud
    text_blob = " ".join(df_filt['Additional Features'].dropna().astype(str))
    if text_blob.strip():
        word_cloud = WordCloud(background_color='white', width=800, height=400).generate(text_blob)
        fig_wc, ax_wc = plt.subplots(figsize=(8,4))
        ax_wc.imshow(word_cloud, interpolation='bilinear')
        ax_wc.axis('off')
        describe_plot(fig_wc, "Word-cloud surfaces popular requested extras—loyalty rewards & AI chatbot dominate.")

# --------- Classification Tab ---------
elif tabs == "Classification":
    st.header("Predicting AI‑Trust Using Classification Models")
    target_col = "Trust AI"
    y = encode_label(df_filt[target_col])
    X = df_filt.drop(columns=[target_col])

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = []
    fitted_models = {}
    for name, model in models.items():
        clf = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results.append({
            "Model": name,
            "Train Acc": accuracy_score(y_train, clf.predict(X_train)),
            "Test Acc": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred)
        })
        fitted_models[name] = clf

    st.dataframe(pd.DataFrame(results).set_index("Model").round(3))

    algo_choice = st.selectbox("Select algorithm to view Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, fitted_models[algo_choice].predict(X_test))
    st.write("Confusion Matrix (rows: actual, cols: predicted)")
    st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

    # ROC Curve
    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots()
    for name, clf in fitted_models.items():
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # Upload new data
    st.subheader("Predict on New Data")
    uploaded = st.file_uploader("Upload CSV with same columns (without 'Trust AI')", type="csv")
    if uploaded:
        new_df = pd.read_csv(uploaded)
        model_sel = st.selectbox("Choose model for prediction", list(models.keys()), key="pred_model")
        preds = fitted_models[model_sel].predict(new_df)
        new_df["Trust_AI_Pred"] = preds
        st.dataframe(new_df.head())
        st.download_button("Download Predictions", data=new_df.to_csv(index=False).encode(), file_name="trust_ai_predictions.csv")

# --------- Clustering Tab ---------
elif tabs == "Clustering":
    st.header("Customer Segmentation with K‑Means")
    df_enc = pd.get_dummies(df_filt.select_dtypes(include="object"))
    k_default = 4
    max_k = st.slider("Choose number of clusters (k)", 2, 10, k_default)
    inertia_list = []
    for k in range(2, 11):
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans_temp.fit(df_enc)
        inertia_list.append(kmeans_temp.inertia_)
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(2, 11), inertia_list, marker="o")
    ax_elbow.set_xlabel("k")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.set_title("Elbow Curve")
    st.pyplot(fig_elbow)

    kmeans = KMeans(n_clusters=max_k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(df_enc)
    df_clustered = df_filt.copy()
    df_clustered["Cluster"] = clusters

    st.subheader("Cluster Personas (Top Modes)")
    persona = df_clustered.groupby("Cluster").agg({
        "Age Group": lambda x: x.mode().iloc[0],
        "Preferred Hotel Type": lambda x: x.mode().iloc[0],
        "Advance Booking Days": lambda x: x.mode().iloc[0],
        "Stay Length (Nights)": lambda x: x.mode().iloc[0],
        "ADR Budget": lambda x: x.mode().iloc[0],
        "Use Price Comparison": lambda x: x.mode().iloc[0],
    })
    gb = GridOptionsBuilder.from_dataframe(persona.reset_index())
    gb.configure_pagination(paginationAutoPageSize=True)
    AgGrid(persona.reset_index(), gridOptions=gb.build(), height=250, theme='alpine')

    import plotly.graph_objects as go
    # Add cluster-level metrics for radar
    df_clustered["ADR_Budget_Numeric"] = df_clustered["ADR Budget"].map({
        '<2000': 1500, '2000–4000': 3000, '4000–7000': 5500, '7000–10000': 8500, '>10000': 12000})
    radar_vars = ['AI_Trust_Pct', 'Mean_LeadTime', 'Mean_ADR']
    radar_df = df_clustered.groupby('Cluster').apply(
        lambda g: pd.Series({
            'AI_Trust_Pct': g['Trust AI'].eq('Yes').mean() * 100,
            'Mean_LeadTime': g['Advance Booking Days'].map(lead_time_map).mean(),
            'Mean_ADR': g['ADR_Budget_Numeric'].mean()
        })
    ).reset_index()
    fig_radar = go.Figure()
    for _, row in radar_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=row[radar_vars].values,
            theta=radar_vars,
            fill='toself',
            name=f"Cluster {int(row['Cluster'])}"
        ))
    fig_radar.update_layout(showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)
    st.download_button("Download Clustered Data", data=df_clustered.to_csv(index=False).encode(), file_name="clustered_data.csv")

# --------- Association Rules Tab ---------
elif tabs == "Association Rules":
    st.header("Association Rule Mining")
    col_choice = st.multiselect("Select columns for Apriori (comma-separated fields)", ["Platforms Used", "Booking Challenges", "Desired Features"])
    min_support = st.slider("Minimum Support", 0.01, 0.3, 0.05, 0.01)
    min_conf = st.slider("Minimum Confidence", 0.1, 0.9, 0.5, 0.05)
    if col_choice:
        basket = df_filt[col_choice].apply(lambda row: ", ".join(row.dropna()), axis=1)
        unique_items = set()
        for items in basket:
            unique_items.update([i.strip() for i in items.split(",")])
        encoded = pd.DataFrame([{item: (item in rec.split(", ")) for item in unique_items} for rec in basket])
        freq_items = apriori(encoded, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
        rules = rules.sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# --------- Regression Tab ---------
elif tabs == "Regression":
    st.header("ADR Budget Prediction")
    budget_map = {
        "<2000": 1500,
        "2000–4000": 3000,
        "4000–7000": 5500,
        "7000–10000": 8500,
        ">10000": 12000
    }
    df_reg = df_filt.copy()
    df_reg["ADR_Budget_Numeric"] = df_reg["ADR Budget"].map(budget_map)
    y_reg = df_reg["ADR_Budget_Numeric"]
    X_reg = df_reg.drop(columns=["ADR Budget", "ADR_Budget_Numeric"])
    cat_cols_r = X_reg.select_dtypes(include="object").columns.tolist()
    num_cols_r = X_reg.select_dtypes(exclude="object").columns.tolist()

    pre_r = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_r),
        ("num", StandardScaler(), num_cols_r)
    ])

    models_r = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    res_reg = []
    for name, reg in models_r.items():
        pipe = Pipeline(steps=[("prep", pre_r), ("model", reg)])
        pipe.fit(X_train_r, y_train_r)
        res_reg.append({
            "Model": name,
            "Train R2": pipe.score(X_train_r, y_train_r),
            "Test R2": pipe.score(X_test_r, y_test_r)
        })
    st.dataframe(pd.DataFrame(res_reg).set_index("Model").round(3))

    best_name = max(res_reg, key=lambda x: x["Test R2"])["Model"]
    st.subheader(f"Actual vs Predicted ADR Budget using {best_name}")
    best_pipe = Pipeline(steps=[("prep", pre_r), ("model", models_r[best_name])])
    best_pipe.fit(X_train_r, y_train_r)
    preds = best_pipe.predict(X_test_r)
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(y_test_r, preds)
    ax_scatter.set_xlabel("Actual")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.set_title("Actual vs Predicted ADR")
    st.pyplot(fig_scatter)
