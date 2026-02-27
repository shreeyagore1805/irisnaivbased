import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Naive Bayes Classifier with Descriptive Analysis")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # ---------------- DESCRIPTIVE ANALYSIS ---------------- #
    st.header("Descriptive Analysis")

    st.subheader("Dataset Shape")
    st.write("Rows:", data.shape[0], "Columns:", data.shape[1])

    st.subheader("Data Types")
    st.write(data.dtypes)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Correlation matrix (numeric only)
    st.subheader("Correlation Matrix")
    st.write(data.corr(numeric_only=True))

    # ---------------- MODEL SECTION ---------------- #
    st.header("Model Training")

    target = st.selectbox("Select Target Column", data.columns)

    features = st.multiselect(
        "Select Feature Columns",
        [col for col in data.columns if col != target]
    )

    test_size = st.slider("Select Test Size (%)", 10, 50, 30) / 100

    if st.button("Train Model"):

        if len(features) == 0:
            st.warning("Please select at least one feature.")
        else:
            X = data[features]
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            model = GaussianNB()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            st.success(f"Model Accuracy: {round(acc * 100, 2)}%")

            st.subheader("Confusion Matrix")
            st.write(cm)

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))