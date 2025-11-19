# AQI Forecasting (ML Module)

This repository contains the **machine learning pipeline** for predicting PM2.5 values using:
- Historical air quality data
- Realtime pollutant data from OpenWeatherMap API
- RandomForest regression model

## 🚀 Features
- Cleaned + processed dataset
- Feature engineering (lags, hour, day-of-week)
- Realtime data integration
- Trained RandomForest model
- Prediction pipeline for backend/frontend teams
- Streamlit demo app

## 📁 Project Structure
src/
preprocess.py
train_model.py
model.py
realtime.py
models/
data/
processed_data.csv
history.csv
app.py