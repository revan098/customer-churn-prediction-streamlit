# customer-churn-prediction-streamlit
machine-learning, churn-prediction, streamlit, data-analytics, classification
# 📊 Customer Churn Prediction using Machine Learning

## 📌 Project Overview
Customer churn is a major challenge for subscription-based businesses such as telecom, SaaS, and banking companies.  
This project builds an **end-to-end Customer Churn Prediction system** that identifies customers who are likely to leave the service, enabling businesses to take **proactive retention actions**.

The solution includes **data analysis, machine learning models, and an interactive Streamlit dashboard** for real-time prediction and insights.

---

## 🎯 Problem Statement
To predict whether a customer will churn based on their:
- Usage behavior
- Contract details
- Billing information
- Service subscriptions

---

## 📂 Dataset
- **Telco Customer Churn Dataset**
- Each row represents a customer
- Target variable: `Churn` (Yes / No)

### Key Features:
- Tenure
- Monthly Charges
- Contract Type
- Payment Method
- Tech Support & Online Security
- Paperless Billing

---

## 🛠️ Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost**
- **Seaborn & Matplotlib**
- **Streamlit**

---

## 🔍 Project Workflow
1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering & Encoding  
4. Model Building  
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
5. Model Evaluation using ROC-AUC  
6. Feature Importance Analysis  
7. Streamlit App Deployment  

---

## 📈 Key Insights
- Customers with **low tenure** are more likely to churn
- **Month-to-month contracts** show the highest churn rate
- Higher **monthly charges** increase churn probability
- Customers using **value-added services** churn less

---

## 🚀 Model Performance
- XGBoost achieved the best performance
- High ROC-AUC score for identifying high-risk churn customers
- Balanced focus on recall and business impact

---

## 🖥️ Streamlit Application
The Streamlit dashboard provides:
- Real-time churn probability prediction
- Risk categorization (Low / Medium / High)
- Interactive EDA visualizations
- Model feature importance insights

---

## 💼 Business Impact
This system helps businesses:
- Identify at-risk customers early
- Design targeted retention strategies
- Reduce revenue loss due to churn
- Improve customer lifetime value

---

## ▶️ How to Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
