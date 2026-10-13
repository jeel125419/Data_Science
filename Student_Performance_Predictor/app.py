import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Performance Predictor")

st.write("Welcome to my first Machine Learning App!")

def reset_form_fields():
    # Setting values to None inside a callback clears the input fields successfully
    st.session_state.math_score = None
    st.session_state.reading_score = None
    st.session_state.writing_score = None
    # Wipe the prediction state if it exists
    if "prediction_data" in st.session_state:
        del st.session_state.prediction_data

with st.form("prediction_form", clear_on_submit=False):

    math_score = st.number_input("Enter maths score:", min_value=0, max_value=100, step=1, value=None, placeholder="Enter maths score", key="math_score")
    reading_score = st.number_input("Enter reading score:", min_value=0, max_value=100, step=1, value=None, placeholder="Enter reading score", key="reading_score")
    writing_score = st.number_input("Enter writing score:", min_value=0, max_value=100, step=1, value=None, placeholder="Enter writing score", key="writing_score")

    col1, col2 = st.columns([1, 4])

    with col1:
        predict_btn = st.form_submit_button("Predict")

    with col2:
        reset_btn = st.form_submit_button("Reset", on_click=reset_form_fields)

if predict_btn:

    if math_score is None or reading_score is None or writing_score is None:
        st.warning("⚠️ Please fill in all fields before predicting.")

    else:

        data = pd.DataFrame({
            "math score": [math_score],
            "reading score": [reading_score],
            "writing score": [writing_score]
        })

        data_scale = scaler.transform(data)

        prediction = model.predict(data_scale)

        st.write("🎉 Predicted Performance: ")

        if prediction[0] == 1:
            st.success("✅ Pass.")
        else:
            st.error("❌ Fail.")

        prob = model.predict_proba(data_scale)

        st.write(f"Confidence: {(prob.max() * 100):.2f}%")
        st.progress(int(round(prob.max() * 100, 2)))


st.divider()
st.subheader("About")
st.write("This application predicts whether a student is likely to pass based on Math, Reading and Writing scores.")
st.caption("Model: Logistic Regression")

