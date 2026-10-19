# Student Performance Predictor

A Machine Learning web application built with Python and Streamlit that predicts whether a student is likely to **Pass or Fail** based on their Maths, Reading, and Writing scores.

## Live Demo

🔗 **Try the app:** [Live Link](https://tkue3t24d3dtet7vapbccv.streamlit.app/)

## Project Overview

The Student Performance Predictor allows users to enter a student's marks in three subjects and receive a predicted performance result.

The application uses a trained Logistic Regression model to make predictions and displays the model's prediction confidence.

## Features

- Enter Maths, Reading, and Writing scores.
- Validate marks between 0 and 100.
- Predict student performance as Pass or Fail.
- Display prediction confidence as a percentage.
- Reset inputs to make another prediction.
- Interactive web interface built with Streamlit.

## Technologies Used

- **Python** — Programming language
- **Pandas** — Data handling
- **Scikit-learn** — Machine Learning and feature scaling
- **Joblib** — Loading the trained model and scaler
- **Streamlit** — Web application interface

## How It Works

1. The user enters Maths, Reading, and Writing scores.
2. The input data is organized into a Pandas DataFrame.
3. The saved scaler transforms the input features.
4. The trained Logistic Regression model predicts the student's performance.
5. The application displays the predicted Pass/Fail result and prediction confidence.

## Model Information

- **Algorithm:** Logistic Regression
- **Input Features:** Maths score, Reading score, Writing score
- **Output:** Pass or Fail
- **Preprocessing:** Feature scaling using a saved scaler

The model and scaler are loaded from `model.pkl` and `scaler.pkl`.

## Author

**Jeel Katua**

- GitHub: [jeel125419](https://github.com/jeel125419)

