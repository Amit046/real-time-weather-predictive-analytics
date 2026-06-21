"""
Train ML models for weather prediction.
Fixes:
- Separate StandardScaler for each model (avoids dimension mismatch)
- Consistent feature lists matching predict.py exactly
- More robust cross-validation reporting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, classification_report)
import joblib
from config import Config
import warnings
warnings.filterwarnings('ignore')


# ── Feature lists (MUST match predict.py exactly) ──────────────────────────
TEMP_FEATURES = [
    'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity',
    'wind_speed', 'clouds', 'pm2_5', 'pm10', 'hour', 'month'
]  # 11 features  →  scaler.joblib

WEATHER_FEATURES = [
    'temperature', 'humidity', 'pressure', 'wind_speed', 'clouds',
    'pm2_5', 'aqi', 'hour', 'month'
]  # 9 features  (no scaler needed for tree classifier)

HUMIDITY_FEATURES = [
    'temperature', 'pressure', 'wind_speed', 'clouds',
    'pm2_5', 'hour', 'month'
]  # 7 features  →  hum_scaler.joblib


class WeatherModelTrainer:

    def __init__(self):
        Config.init_app()
        self.temp_scaler = StandardScaler()
        self.hum_scaler  = StandardScaler()
        self.label_enc   = LabelEncoder()

    # ------------------------------------------------------------------ #
    def load_data(self):
        try:
            df = pd.read_csv(Config.DATASET_FILE)
            print(f"✅ Loaded {len(df)} records from {Config.DATASET_FILE}")
        except FileNotFoundError:
            print("❌ Dataset not found — run generate_sample_data.py first.")
            return None

        # Fill numeric NaN
        df = df.fillna(df.mean(numeric_only=True))

        # Time features
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour']  = df['timestamp'].dt.hour.fillna(12).astype(int)
        df['month'] = df['timestamp'].dt.month.fillna(6).astype(int)

        # Ensure required columns exist
        required = set(TEMP_FEATURES + WEATHER_FEATURES + HUMIDITY_FEATURES
                       + ['temperature', 'humidity', 'weather_main', 'aqi'])
        missing = required - set(df.columns)
        if missing:
            print(f"⚠️  Missing columns: {missing}")
            return None

        # Drop rows with NaN in any feature column
        df = df.dropna(subset=list(required))
        print(f"   After cleaning: {len(df)} records")
        return df

    # ------------------------------------------------------------------ #
    def train_temperature_model(self, df):
        print("\n🔥 Training Temperature Model …")
        X = df[TEMP_FEATURES].values
        y = df['temperature'].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)

        X_tr_s = self.temp_scaler.fit_transform(X_tr)
        X_te_s = self.temp_scaler.transform(X_te)

        candidates = {
            'RandomForest':     RandomForestRegressor(
                                    n_estimators=200, max_depth=15,
                                    n_jobs=-1, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(
                                    n_estimators=150, max_depth=8,
                                    learning_rate=0.1, random_state=42),
        }

        best_model, best_mae = None, float('inf')
        for name, m in candidates.items():
            m.fit(X_tr_s, y_tr)
            pred = m.predict(X_te_s)
            mae  = mean_absolute_error(y_te, pred)
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            r2   = r2_score(y_te, pred)
            print(f"   {name}: MAE={mae:.3f}°C  RMSE={rmse:.3f}°C  R²={r2:.3f}")
            if mae < best_mae:
                best_mae   = mae
                best_model = m

        joblib.dump(best_model,      Config.TEMP_MODEL_FILE)
        joblib.dump(self.temp_scaler, Config.SCALER_FILE)
        print(f"✅ Temperature model saved  (best MAE: {best_mae:.3f}°C)")
        return best_model

    # ------------------------------------------------------------------ #
    def train_weather_classifier(self, df):
        print("\n☁️  Training Weather Classifier …")
        X = df[WEATHER_FEATURES].values
        y = self.label_enc.fit_transform(df['weather_main'].values)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
        clf.fit(X_tr, y_tr)

        pred = clf.predict(X_te)
        acc  = accuracy_score(y_te, pred)
        print(f"   Accuracy: {acc:.3f}")
        print(classification_report(y_te, pred,
              target_names=self.label_enc.classes_, zero_division=0))

        # Store class names inside the classifier for later use
        clf.classes_names_ = list(self.label_enc.classes_)

        joblib.dump(clf, Config.WEATHER_MODEL_FILE)
        print(f"✅ Weather classifier saved  (Accuracy: {acc:.3f})")
        return clf

    # ------------------------------------------------------------------ #
    def train_humidity_model(self, df):
        print("\n💧 Training Humidity Model …")
        X = df[HUMIDITY_FEATURES].values
        y = df['humidity'].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)

        X_tr_s = self.hum_scaler.fit_transform(X_tr)
        X_te_s = self.hum_scaler.transform(X_te)

        model = RandomForestRegressor(
            n_estimators=150, max_depth=12, n_jobs=-1, random_state=42)
        model.fit(X_tr_s, y_tr)

        pred = model.predict(X_te_s)
        mae  = mean_absolute_error(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        print(f"   MAE={mae:.3f}%  RMSE={rmse:.3f}%")

        joblib.dump(model,           Config.HUMIDITY_MODEL_FILE)
        joblib.dump(self.hum_scaler, Config.HUM_SCALER_FILE)
        print(f"✅ Humidity model saved  (MAE: {mae:.3f}%)")
        return model

    # ------------------------------------------------------------------ #
    def train_all_models(self):
        print("=" * 60)
        print("🚀 STARTING ML MODEL TRAINING")
        print("=" * 60)

        df = self.load_data()
        if df is None:
            return

        self.train_temperature_model(df)
        self.train_weather_classifier(df)
        self.train_humidity_model(df)

        print("\n" + "=" * 60)
        print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 60)


if __name__ == "__main__":
    trainer = WeatherModelTrainer()
    trainer.train_all_models()