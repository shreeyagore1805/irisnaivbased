import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    mean_squared_error,
    r2_score,
    accuracy_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ML App with EDA", layout="centered")
st.title("Machine Learning App (Classification & Regression)")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is None:
    st.info("Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(file)

# ---------------- DATA PREVIEW ----------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ================= EDA =================
st.header("Exploratory Data Analysis")

numeric_cols = df.select_dtypes(include="number")

if not numeric_cols.empty:
    st.subheader("Summary Statistics")
    st.dataframe(numeric_cols.describe())

    col = st.selectbox("Select Numeric Column", numeric_cols.columns)

    # Histogram
    st.markdown("**Histogram**")
    fig, ax = plt.subplots()
    ax.hist(df[col], bins=20)
    st.pyplot(fig)

    # Boxplot
    st.markdown("**Box Plot**")
    fig, ax = plt.subplots()
    ax.boxplot(df[col], vert=False)
    st.pyplot(fig)

    # Correlation Heatmap
    if len(numeric_cols.columns) > 1:
        st.markdown("**Correlation Heatmap**")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ================= MODEL SECTION =================
st.header("Model Training")

model_type = st.radio("Choose Model Type", ["Classification", "Regression"])

target_col = st.selectbox("Select Target Column", df.columns)

feature_cols = st.multiselect(
    "Select Feature Columns",
    [col for col in df.columns if col != target_col and df[col].dtype != "object"]
)

train_percent = st.slider("Training Data Percentage", 60, 90, 70)

if model_type == "Classification":
    classifier_name = st.selectbox(
        "Choose Classifier",
        ["Naive Bayes", "Logistic Regression", "KNN", "Decision Tree"]
    )

# ================= TRAIN MODEL =================
if st.button("Train Model"):

    if len(feature_cols) == 0:
        st.warning("Select at least one feature column.")
        st.stop()

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(100 - train_percent) / 100, random_state=42
    )

    # ---------- Classification ----------
    if model_type == "Classification":

        y_train = y_train.astype(str)
        y_test = y_test.astype(str)

        if classifier_name == "Naive Bayes":
            model = GaussianNB()
        elif classifier_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif classifier_name == "KNN":
            model = KNeighborsClassifier()
        else:
            model = DecisionTreeClassifier(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Accuracy")
        st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        st.pyplot(fig)

    # ---------- Regression ----------
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        st.subheader("Regression Metrics")
        st.write("MSE:", round(mse, 4))
        st.write("RMSE:", round(mse ** 0.5, 4))
        st.write("R²:", round(r2_score(y_test, y_pred), 4))

    st.success("Model trained successfully.")
