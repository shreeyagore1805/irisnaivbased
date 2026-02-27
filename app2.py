import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

st.title("Naive Bayes Classifier with Descriptive Analysis & Visualization") 
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"]) 
if uploaded_file is not None:
 data = pd.read_csv(uploaded_file) 
 st.header("Descriptive Analysis")
 st.subheader("Dataset Shape") 
 st.write("Rows:", data.shape[0], "Columns:", data.shape[1]) 
 st.subheader("Dataset Preview") 
 st.dataframe(data.head()) 
 st.subheader("Summary Statistics") 
 st.write(data.describe()) 
# ----------- VISUALIZATION SECTION ----------- # 

 st.header("Data Visualization") 
 numeric_cols = data.select_dtypes(include=np.number).columns 
 # Histogram 
 st.subheader("Histogram") 
 hist_col = st.selectbox("Select Column for Histogram", numeric_cols) 
 fig1, ax1 = plt.subplots() 
 sns.histplot(data[hist_col], kde=True, ax=ax1) 
 st.pyplot(fig1) 
 # Boxplot 
 st.subheader("Boxplot") 
 box_col = st.selectbox("Select Column for Boxplot", numeric_cols) 
 fig2, ax2 = plt.subplots() 
 sns.boxplot(y=data[box_col], ax=ax2) 
 st.pyplot(fig2) 
 # Correlation Heatmap 
 st.subheader("Correlation Heatmap") 
 fig3, ax3 = plt.subplots() 
 sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3) 
 st.pyplot(fig3) 
 # Target Distribution 
 st.subheader("Target Distribution") 
 target_col = st.selectbox("Select Target for Distribution", data.columns) 
 fig4, ax4 = plt.subplots() 
 sns.countplot(x=data[target_col], ax=ax4) 
 plt.xticks(rotation=45) 
 st.pyplot(fig4) 
 # ----------- MODEL SECTION ----------- # 
 st.header("Model Training") 
 target = st.selectbox("Select Target Column for Model", data.columns) 
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




