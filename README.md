# 🌦️ Real-Time Weather Predictive Analytics Dashboard

An end-to-end **Machine Learning & Predictive Analytics project** that leverages **real-time weather and air quality data** to generate insights, predictions, and comparative analytics using a **futuristic visualization dashboard**.

This project goes beyond basic weather apps by focusing on **prediction, trend analysis, and data-driven insights**.

---

<img width="1919" height="915" alt="image" src="https://github.com/user-attachments/assets/c0c67aef-08c8-44b1-9760-cae14cdbeb18" />
---
live link: https://skycast-vr38.onrender.com
---
## 🚀 Key Features

- 🌍 Real-time weather & AQI data using live APIs  
- 🤖 Machine Learning-based prediction models  
- 📊 Advanced analytics & comparison dashboards  
- ⚡ Futuristic, dynamic data visualizations  
- 🏙️ Multi-city weather and pollution analysis  
- 📈 Trend forecasting & variability analysis  

---
## clone git clone https://github.com/Amit046/real-time-weather-predictive-analytics.git
---

## 🧠 Project Objective

Most weather applications only display current conditions.

This project aims to:
- Predict **future temperature & weather behavior**
- Analyze **pollution patterns & AQI trends**
- Compare **cities across multiple metrics**
- Visualize **historical trends & correlations**
- Convert raw data into **actionable insights**

---

## 🏗️ Project Architecture

```
real-time-weather-predictive-analytics/
│
├── app.py
├── dashboard.py
├── advanced_analytics.py
├── train_model.py
├── predict.py
├── data_collector.py
├── auto_data_collector.py
├── generate_sample_data.py
├── config.py
│
├── data/
│   └── weather_data.csv
│
├── models/
│   ├── temperature_model.joblib
│   ├── humidity_model.joblib
│   ├── weather_classifier.joblib
│   └── scaler.joblib
│
├── templates/
│   ├── index.html
│   └── dashboard.html
│
├── requirements.txt
└── README.md
```

---

## 🤖 Machine Learning Models Used

- Random Forest Regressor (Temperature & Humidity)
- Weather Classification Model
- StandardScaler for normalization

---

## 📊 Analytics & Visualizations

- City-wise temperature comparison  
- AQI & PM2.5 pollution analysis  
- Time-series trend analysis  
- Weather distribution charts  
- Correlation & variability analysis  
- Multi-metric radar charts  

---

## 🌐 Real-Time Data Source

- OpenWeather API (Weather + AQI)

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python generate_sample_data.py
python train_model.py
python app.py
```

Open:
- http://127.0.0.1:5000
- http://127.0.0.1:5000/dashboard

---

## 🔐 Environment Variables

Create `.env` file:

```
OPENWEATHER_API_KEY=your_api_key_here
DEFAULT_CITY=Delhi
```

---

## 👤 Author

**Amit**  
GitHub: https://github.com/Amit046
