import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ================= LOAD ARTIFACTS =================
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# Load dataset for insights
df = pd.read_csv("Telco_Cusomer_Churn.csv")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Create plotting copy (IMPORTANT FIX)
df_plot = df.copy()
df_plot["Churn"] = df_plot["Churn"].map({0: "No Churn", 1: "Churn"})

# ================= TITLE =================
st.markdown(
    "<h1 style='text-align:center;'>📊 Customer Churn Prediction Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Prediction • Insights • Model Explainability</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ================= TABS =================
tab1, tab2, tab3 = st.tabs(
    ["🔮 Predict Churn", "📈 Customer Insights", "🧠 Model Insights"]
)

# =====================================================
# 🔮 TAB 1: PREDICTION
# =====================================================
with tab1:
    st.subheader("🧾 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges (₹)", 20, 150, 70)

    with col2:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col3:
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No"]
        )

    if st.button("🚀 Predict Churn Risk"):
        # Build input exactly like training data
        input_df = pd.DataFrame(0, index=[0], columns=features)

        input_df["tenure"] = tenure
        input_df["MonthlyCharges"] = monthly_charges
        input_df["PaperlessBilling"] = 1 if paperless == "Yes" else 0
        input_df["TechSupport"] = 1 if tech_support == "Yes" else 0
        input_df["OnlineSecurity"] = 1 if online_security == "Yes" else 0

        # Contract encoding (drop_first=True logic)
        if contract == "One year" and "Contract_One year" in input_df.columns:
            input_df["Contract_One year"] = 1
        elif contract == "Two year" and "Contract_Two year" in input_df.columns:
            input_df["Contract_Two year"] = 1
        # Month-to-month → baseline → do nothing

        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]

        st.metric("Churn Probability", f"{prob:.2%}")
        st.progress(int(prob * 100))

        if prob < 0.3:
            st.success("🟢 Low Risk: Customer likely to stay")
        elif prob < 0.6:
            st.warning("🟠 Medium Risk: Consider retention offers")
        else:
            st.error("🔴 High Risk: Immediate retention action required")

# =====================================================
# 📈 TAB 2: CUSTOMER INSIGHTS (GRAPHS + INSIGHTS)
# =====================================================
with tab2:
    st.subheader("📊 Exploratory Data Analysis")

    # ---- Tenure vs Churn ----
    st.markdown("### 🔹 Tenure vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(
        x="Churn",
        y="tenure",
        data=df_plot,
        palette={"No Churn": "#2ecc71", "Churn": "#e74c3c"},
        ax=ax
    )
    st.pyplot(fig)

    st.info(
        "📌 **Insight:** Customers who churn usually leave within the first few months. "
        "This indicates that early engagement and onboarding are critical."
    )

    # ---- Monthly Charges vs Churn ----
    st.markdown("### 🔹 Monthly Charges vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(
        x="Churn",
        y="MonthlyCharges",
        data=df_plot,
        palette={"No Churn": "#3498db", "Churn": "#f39c12"},
        ax=ax
    )
    st.pyplot(fig)

    st.info(
        "📌 **Insight:** Customers paying higher monthly charges are more likely to churn, "
        "indicating price sensitivity and perceived value issues."
    )

    # ---- Contract Type vs Churn ----
    st.markdown("### 🔹 Contract Type vs Churn")
    fig, ax = plt.subplots()
    sns.countplot(
        x="Contract",
        hue="Churn",
        data=df_plot,
        palette={"No Churn": "#27ae60", "Churn": "#c0392b"},
        ax=ax
    )
    st.pyplot(fig)

    st.info(
        "📌 **Insight:** Month-to-month contracts show the highest churn, while long-term "
        "contracts significantly improve customer retention."
    )

# =====================================================
# 🧠 TAB 3: MODEL INSIGHTS
# =====================================================
with tab3:
    st.subheader("🧠 Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance = (
            pd.Series(model.feature_importances_, index=features)
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots()
        importance.plot(kind="barh", color="#8e44ad", ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)

        st.success(
            "📌 **Insight:** Tenure, Monthly Charges, and Contract Type are the strongest "
            "drivers of churn. Improving these areas can significantly reduce churn."
        )
    else:
        st.warning("Feature importance not available for this model.")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>Built with ❤️ using Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
