import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title(" Machine Learning Model Comparison Dashboard")

# Load saved results
scores = pickle.load(open("scores.pkl", "rb"))

# ---------------- MODEL COMPARISON TABLE ----------------
models = ["Decision Tree", "Random Forest"]

data = {
    "Model": models,
    "Accuracy": [
        scores["Decision Tree"]["metrics"]["Accuracy"],
        scores["Random Forest"]["metrics"]["Accuracy"]
    ],
    "Precision": [
        scores["Decision Tree"]["metrics"]["Precision"],
        scores["Random Forest"]["metrics"]["Precision"]
    ],
    "Recall": [
        scores["Decision Tree"]["metrics"]["Recall"],
        scores["Random Forest"]["metrics"]["Recall"]
    ],
    "F1 Score": [
        scores["Decision Tree"]["metrics"]["F1"],
        scores["Random Forest"]["metrics"]["F1"]
    ],
    "ROC-AUC": [
        scores["Decision Tree"]["metrics"]["ROC-AUC"],
        scores["Random Forest"]["metrics"]["ROC-AUC"]
    ],
    "Training Time (sec)": [
        scores["Decision Tree"]["time"],
        scores["Random Forest"]["time"]
    ]
}

df = pd.DataFrame(data)

st.subheader(" Model Comparison Table")
st.dataframe(df)

# ---------------- BAR CHART ----------------
st.subheader(" Accuracy Comparison Chart")
st.bar_chart(df.set_index("Model")[["Accuracy"]])

# ---------------- CONFUSION MATRIX ----------------
st.subheader(" Confusion Matrix")

st.write("Decision Tree Confusion Matrix")
st.write(scores["Decision Tree"]["metrics"]["Confusion Matrix"])

st.write("Random Forest Confusion Matrix")
st.write(scores["Random Forest"]["metrics"]["Confusion Matrix"])

# ---------------- LIVE PREDICTION ----------------
st.subheader(" Live Weather Prediction")

model = pickle.load(open("random.pkl", "rb"))

temperature = st.number_input("Temperature", 0, 50, 25)
humidity = st.number_input("Humidity", 0, 100, 50)
wind = st.number_input("Wind Speed", 0, 50, 10)

if st.button("Predict Rain"):
    input_data = np.array([[temperature, humidity, wind]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🌧 It will Rain")
    else:
        st.success("☀ No Rain Expected")