import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_processing import load_data, preprocess_data
from utils.classification import predict_expense_category
from utils.regression import recommend_budget
from utils.lstm_forecast import forecast_lstm
from utils.anomaly import detect_anomalies
import re
from PIL import Image
import pytesseract

# 🕰️ Conversion Rate and Currency Selection
currency = st.sidebar.selectbox("Choose Currency", ["USD", "INR"])
conversion_rate = 1 if currency == "USD" else 83.2
symbol = "$" if currency == "USD" else "₹"

# 📋 App title
st.title("Personal Finance Assistant - AI Agent (₹ Edition)")

# 📂 Load and preprocess data
expense_data = load_data("data/expense_data.csv")
budget_data = load_data("data/budget_data.csv")
forecast_data = load_data("data/expense_forecast_data.csv")

# Convert amounts to selected currency
expense_data['Amount_Conv'] = expense_data['Amount'] * conversion_rate

# 🔹 Sidebar for navigation
option = st.sidebar.selectbox(
    "Choose a feature:",
    ["Expense Categorization", "Budget Recommendations", "Expense Forecasting", "Visual Analytics", "Anomaly Detection", "OCR Receipt Reader", "Saving Goals"]
)

if option == "Expense Categorization":
    st.header("Predictive Expense Categorization")
    description = st.text_input("Enter expense description:")
    if st.button("Classify Expense"):
        category = predict_expense_category(description)
        st.write(f"Predicted Category: **{category}**")

elif option == "Budget Recommendations":
    st.header("Budget Recommendations")
    income = st.number_input(f"Enter your monthly income ({symbol}):")
    if st.button("Get Budget Recommendation"):
        recommendation = recommend_budget(income / conversion_rate, budget_data) * conversion_rate
        st.write(f"Recommended Budget: **{symbol}{recommendation:,.2f}**")

elif option == "Expense Forecasting":
    st.header("Expense Forecasting with LSTM (Deep Learning)")
    months = st.number_input("Enter months to forecast:", min_value=1, max_value=12)
    if st.button("Forecast Expenses"):
        forecast = forecast_lstm(expense_data, months) * conversion_rate
        st.line_chart(forecast)

elif option == "Visual Analytics":
    st.header("Visual Expense Analysis")
    if not expense_data.empty:
        fig1 = px.pie(expense_data, values='Amount_Conv', names='Category', title='Spending by Category')
        st.plotly_chart(fig1)

        expense_data['Date'] = pd.to_datetime(expense_data['Date'])
        monthly = expense_data.groupby(expense_data['Date'].dt.to_period("M"))['Amount_Conv'].sum()
        st.line_chart(monthly)

elif option == "Anomaly Detection":
    st.header("Detect Anomalies in Spending")
    if st.button("Run Anomaly Detection"):
        anomalies = detect_anomalies(expense_data[['Amount_Conv']])
        st.dataframe(anomalies)

elif option == "OCR Receipt Reader":
    st.header("Extract Text from Receipt Image")
    image = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg"])
    if image:
        text = pytesseract.image_to_string(Image.open(image))
        st.text_area("Extracted Text", text)

elif option == "Saving Goals":
    st.header("Set and Track Saving Goals")
    goal = st.number_input(f"Enter your monthly saving goal ({symbol}):")
    spent = expense_data['Amount_Conv'].sum()
    progress = max(0, min(100, 100 - (spent / goal * 100))) if goal else 0
    st.progress(progress)
    st.write(f"Progress: {100 - progress:.2f}% spent")

# 📄 CSV Export
st.sidebar.download_button("📄 Download CSV", data=expense_data.to_csv(), file_name="expenses.csv")
