
import streamlit as st
import pandas as pd
import joblib

# Load model
saved = joblib.load("model.pkl")
model = saved["model"]
features = saved["features"]

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
sibsp = st.number_input("Siblings / Spouses", 0, 10, 0)
parch = st.number_input("Parents / Children", 0, 10, 0)

if st.button("Predict"):

    data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "SibSp": sibsp,
        "Parch": parch
    }])

    data = pd.get_dummies(data, dtype=int)
    data = data.reindex(columns=features, fill_value=0)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.success("🎉 Passenger would have SURVIVED!")
    else:
        st.error("Passenger would NOT have survived.")

    st.write(f"Survival probability: {probability:.2%}")
