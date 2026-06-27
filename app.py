import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Weather Prediction Dashboard",
    page_icon="🌦",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    color:#1f77b4;
    font-size:40px;
    font-weight:bold;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:2px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES ----------------
scores = pickle.load(open("scores.pkl","rb"))
model = pickle.load(open("random.pkl","rb"))
df_dataset = pd.read_csv("weather_dataset.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Dataset Analysis",
        "Model Comparison",
        "Prediction"
    ]
)

# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.markdown(
        "<h1 class='title'>🌦 Weather Prediction System</h1>",
        unsafe_allow_html=True
    )

    st.image(
        "https://images.unsplash.com/photo-1504608524841-42fe6f032b4b",
        use_container_width=True
    )

    st.write("---")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("Dataset Records",len(df_dataset))

    with col2:
        st.metric(
            "Decision Tree Accuracy",
            f"{scores['Decision Tree']['metrics']['Accuracy']:.2f}"
        )

    with col3:
        st.metric(
            "Random Forest Accuracy",
            f"{scores['Random Forest']['metrics']['Accuracy']:.2f}"
        )

    st.write("---")

    st.subheader("Project Description")

    st.write("""
    This project predicts whether rain will occur using:

    ✔ Decision Tree Algorithm

    ✔ Random Forest Algorithm

    The system compares both models using:

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Training Time

    A live prediction module is also included.
    """)


# ==================================================
# DATASET ANALYSIS
# ==================================================

elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(df_dataset.head())

    st.subheader("Dataset Shape")

    st.write(df_dataset.shape)

    st.subheader("Statistical Summary")

    st.write(df_dataset.describe())

    st.subheader("Temperature Distribution")

    st.bar_chart(df_dataset["Temperature"].value_counts())

    st.subheader("Humidity Distribution")

    st.line_chart(df_dataset["Humidity"])


# ==================================================
# MODEL COMPARISON
# ==================================================

elif page == "Model Comparison":

    st.title("🤖 Model Comparison")

    models = ["Decision Tree","Random Forest"]

    data = {

        "Model":models,

        "Accuracy":[
            scores["Decision Tree"]["metrics"]["Accuracy"],
            scores["Random Forest"]["metrics"]["Accuracy"]
        ],

        "Precision":[
            scores["Decision Tree"]["metrics"]["Precision"],
            scores["Random Forest"]["metrics"]["Precision"]
        ],

        "Recall":[
            scores["Decision Tree"]["metrics"]["Recall"],
            scores["Random Forest"]["metrics"]["Recall"]
        ],

        "F1 Score":[
            scores["Decision Tree"]["metrics"]["F1"],
            scores["Random Forest"]["metrics"]["F1"]
        ],

        "ROC-AUC":[
            scores["Decision Tree"]["metrics"]["ROC-AUC"],
            scores["Random Forest"]["metrics"]["ROC-AUC"]
        ],

        "Training Time":[
            scores["Decision Tree"]["time"],
            scores["Random Forest"]["time"]
        ]

    }

    df = pd.DataFrame(data)

    st.dataframe(df)

    st.subheader("Accuracy Comparison")

    st.bar_chart(df.set_index("Model")["Accuracy"])

    st.subheader("Precision Comparison")

    st.bar_chart(df.set_index("Model")["Precision"])

    st.subheader("Recall Comparison")

    st.bar_chart(df.set_index("Model")["Recall"])

    st.subheader("F1 Score Comparison")

    st.bar_chart(df.set_index("Model")["F1 Score"])

    st.subheader("Confusion Matrix")

    col1,col2 = st.columns(2)

    with col1:

        st.write("Decision Tree")

        st.write(
            scores["Decision Tree"]["metrics"]["Confusion Matrix"]
        )

    with col2:

        st.write("Random Forest")

        st.write(
            scores["Random Forest"]["metrics"]["Confusion Matrix"]
        )


# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "Prediction":

    st.title("🌧 Live Weather Prediction")

    col1,col2,col3 = st.columns(3)

    with col1:

        temperature = st.number_input(
            "Temperature",
            0,
            50,
            25
        )

    with col2:

        humidity = st.number_input(
            "Humidity",
            0,
            100,
            50
        )

    with col3:

        wind = st.number_input(
            "Wind Speed",
            0,
            50,
            10
        )

    if st.button("Predict Weather"):

        input_data = np.array([
            [temperature,humidity,wind]
        ])

        prediction = model.predict(input_data)

        if prediction[0] == 1:

            st.success("🌧 Rain Expected")

        else:

            st.success("☀ No Rain Expected")