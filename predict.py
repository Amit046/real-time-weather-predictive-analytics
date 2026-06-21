"""
WeatherPredictor
- Loads trained ML models
- Makes current + ML-enhanced predictions
- 10-day forecast via Open-Meteo (free)
"""

import joblib
import numpy as np
from datetime import datetime
import os
from config import Config
from data_collector import WeatherDataCollector

# ── Feature lists MUST match train_model.py exactly ───────────────────────
TEMP_FEATURES     = ['feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity',
                     'wind_speed', 'clouds', 'pm2_5', 'pm10', 'hour', 'month']
WEATHER_FEATURES  = ['temperature', 'humidity', 'pressure', 'wind_speed', 'clouds',
                     'pm2_5', 'aqi', 'hour', 'month']
HUMIDITY_FEATURES = ['temperature', 'pressure', 'wind_speed', 'clouds',
                     'pm2_5', 'hour', 'month']


class WeatherPredictor:

    def __init__(self):
        self.collector = WeatherDataCollector()
        self._load_models()

    # ------------------------------------------------------------------ #
    def _load_models(self):
        try:
            self.temp_model    = joblib.load(Config.TEMP_MODEL_FILE)
            self.weather_clf   = joblib.load(Config.WEATHER_MODEL_FILE)
            self.humidity_model= joblib.load(Config.HUMIDITY_MODEL_FILE)
            self.temp_scaler   = joblib.load(Config.SCALER_FILE)

            # Humidity scaler — new separate file; graceful fallback
            if os.path.exists(Config.HUM_SCALER_FILE):
                self.hum_scaler = joblib.load(Config.HUM_SCALER_FILE)
            else:
                print("⚠️  hum_scaler.joblib not found — retrain models.")
                self.hum_scaler = None

            print("✅ All models loaded successfully!")
        except FileNotFoundError as e:
            print(f"❌ Model file missing: {e}")
            print("   Run train_model.py first.")
            self.temp_model = self.weather_clf = self.humidity_model = None
            self.temp_scaler = self.hum_scaler = None

    # ------------------------------------------------------------------ #
    #  HELPERS                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_aqi_category(aqi):
        return {1: "Good", 2: "Fair", 3: "Moderate",
                4: "Poor", 5: "Very Poor"}.get(int(aqi) if aqi else 0, "Unknown")

    @staticmethod
    def get_health_advice(aqi, pm25):
        aqi   = aqi   or 0
        pm25  = pm25  or 0
        if aqi >= 4 or pm25 > 55:
            return ("⚠️ Air quality is poor. Avoid outdoor activities. "
                    "Wear a mask if going outside.")
        elif aqi == 3:
            return ("⚡ Moderate air quality. Sensitive groups should "
                    "limit outdoor activities.")
        return "✅ Air quality is good. Safe for outdoor activities."

    def _safe(self, live, key, default=0):
        """Safely get a numeric value from live data dict."""
        v = live.get(key)
        return float(v) if v is not None else default

    # ------------------------------------------------------------------ #
    #  CURRENT + ML PREDICTION                                             #
    # ------------------------------------------------------------------ #
    def predict_for_city(self, city_name):
        """Fetch live weather and run ML predictions."""
        print(f"\n🌍 Fetching live data for {city_name} …")
        live = self.collector.fetch_weather_data(city_name)

        if not live:
            return {"error": f"Could not fetch data for {city_name}"}

        now = datetime.now()
        hour  = now.hour
        month = now.month

        # ── Build feature vectors using CONSISTENT field names ────────
        feat_temp = np.array([[
            self._safe(live, 'feels_like'),
            self._safe(live, 'temp_min'),
            self._safe(live, 'temp_max'),
            self._safe(live, 'pressure', 1013),
            self._safe(live, 'humidity'),
            self._safe(live, 'wind_speed'),
            self._safe(live, 'clouds'),
            self._safe(live, 'pm2_5'),
            self._safe(live, 'pm10'),
            hour,
            month,
        ]])  # shape (1, 11) — matches TEMP_FEATURES

        feat_weather = np.array([[
            self._safe(live, 'temperature'),
            self._safe(live, 'humidity'),
            self._safe(live, 'pressure', 1013),
            self._safe(live, 'wind_speed'),
            self._safe(live, 'clouds'),
            self._safe(live, 'pm2_5'),
            self._safe(live, 'aqi', 1),
            hour,
            month,
        ]])  # shape (1, 9) — matches WEATHER_FEATURES

        feat_humidity = np.array([[
            self._safe(live, 'temperature'),
            self._safe(live, 'pressure', 1013),
            self._safe(live, 'wind_speed'),
            self._safe(live, 'clouds'),
            self._safe(live, 'pm2_5'),
            hour,
            month,
        ]])  # shape (1, 7) — matches HUMIDITY_FEATURES

        aqi_val = live.get('aqi') or 1

        result = {
            'city':      city_name,
            'timestamp': str(live.get('timestamp', datetime.now())),
            'current': {
                'temperature': round(self._safe(live, 'temperature'), 1),
                'feels_like':  round(self._safe(live, 'feels_like'),  1),
                'temp_min':    round(self._safe(live, 'temp_min'),     1),
                'temp_max':    round(self._safe(live, 'temp_max'),     1),
                'humidity':    int(self._safe(live, 'humidity')),
                'pressure':    int(self._safe(live, 'pressure', 1013)),
                'wind_speed':  round(self._safe(live, 'wind_speed'),   1),
                'clouds':      int(self._safe(live, 'clouds')),
                'weather':     live.get('weather_main', 'Unknown'),
                'description': live.get('weather_description', ''),
                'aqi':         int(aqi_val),
                'aqi_category': self.get_aqi_category(aqi_val),
                'pm2_5':       round(self._safe(live, 'pm2_5'), 2),
                'pm10':        round(self._safe(live, 'pm10'),  2),
            },
            'health_advice': self.get_health_advice(aqi_val, live.get('pm2_5')),
        }

        # ── ML Predictions ────────────────────────────────────────────
        ml = {}
        if self.temp_model and self.temp_scaler:
            try:
                fs = self.temp_scaler.transform(feat_temp)
                pred_temp = float(self.temp_model.predict(fs)[0])
                ml['predicted_temperature'] = round(pred_temp, 2)
                ml['temp_difference'] = round(
                    pred_temp - self._safe(live, 'temperature'), 2)
            except Exception as e:
                print(f"Temp prediction error: {e}")

        if self.humidity_model and self.hum_scaler:
            try:
                fs = self.hum_scaler.transform(feat_humidity)
                pred_hum = float(self.humidity_model.predict(fs)[0])
                ml['predicted_humidity'] = round(pred_hum, 1)
            except Exception as e:
                print(f"Humidity prediction error: {e}")

        if self.weather_clf:
            try:
                code = int(self.weather_clf.predict(feat_weather)[0])
                # Use stored class names if available
                if hasattr(self.weather_clf, 'classes_names_'):
                    classes = self.weather_clf.classes_names_
                else:
                    classes = ['Clear', 'Clouds', 'Drizzle', 'Haze',
                               'Mist', 'Rain', 'Snow', 'Thunderstorm']
                ml['predicted_weather'] = (classes[code]
                                           if code < len(classes) else 'Unknown')
            except Exception as e:
                print(f"Weather classification error: {e}")

        if ml:
            result['ml_predictions'] = ml

        return result

    # ------------------------------------------------------------------ #
    #  10-DAY FORECAST                                                      #
    # ------------------------------------------------------------------ #
    def get_10day_forecast(self, city_name):
        """Return 10-day forecast list using Open-Meteo (no key needed)."""
        forecast = self.collector.fetch_10day_forecast(city_name)
        if not forecast:
            return {"error": f"Could not fetch forecast for {city_name}"}
        return {
            'city':     city_name,
            'forecast': forecast,
        }

    # ------------------------------------------------------------------ #
    #  CLI DISPLAY                                                          #
    # ------------------------------------------------------------------ #
    def display_predictions(self, p):
        if 'error' in p:
            print(f"\n❌ {p['error']}")
            return

        print("\n" + "=" * 70)
        print(f"📍 WEATHER REPORT: {p['city']}")
        print("=" * 70)

        c = p['current']
        print(f"\n🌡️  CURRENT CONDITIONS:")
        print(f"   Temperature : {c['temperature']}°C  "
              f"(Feels like: {c['feels_like']}°C)")
        print(f"   Range       : {c['temp_min']}°C – {c['temp_max']}°C")
        print(f"   Weather     : {c['weather']} — {c['description']}")
        print(f"   Humidity    : {c['humidity']}%")
        print(f"   Pressure    : {c['pressure']} hPa")
        print(f"   Wind Speed  : {c['wind_speed']} m/s")
        print(f"   Cloud Cover : {c['clouds']}%")

        print(f"\n💨 AIR QUALITY:")
        print(f"   AQI   : {c['aqi_category']} (Level {c['aqi']})")
        print(f"   PM2.5 : {c['pm2_5']} µg/m³")
        print(f"   PM10  : {c['pm10']} µg/m³")

        if 'ml_predictions' in p:
            ml = p['ml_predictions']
            print(f"\n🤖 ML PREDICTIONS:")
            if 'predicted_temperature' in ml:
                print(f"   Predicted Temp     : {ml['predicted_temperature']}°C "
                      f"(Δ {ml.get('temp_difference', 0):+.2f}°C)")
            if 'predicted_humidity' in ml:
                print(f"   Predicted Humidity : {ml['predicted_humidity']}%")
            if 'predicted_weather' in ml:
                print(f"   Predicted Condition: {ml['predicted_weather']}")

        print(f"\n💡 HEALTH ADVICE:\n   {p['health_advice']}")
        print("\n" + "=" * 70)


# ── Standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = WeatherPredictor()
    city = input("\nEnter city (Enter = Delhi): ").strip() or "Delhi"

    print("\n--- Current + ML ---")
    pred = predictor.predict_for_city(city)
    predictor.display_predictions(pred)

    print("\n--- 10-Day Forecast ---")
    fc = predictor.get_10day_forecast(city)
    if 'forecast' in fc:
        for d in fc['forecast']:
            print(f"{d['date']}  {d['icon']} {d['condition']:20s}  "
                  f"{d['temp_min']}–{d['temp_max']}°C  "
                  f"Rain {d['precip_prob']}%  "
                  f"💨{d['wind_max']} km/h")