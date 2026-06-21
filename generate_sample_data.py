"""
Generate realistic sample weather data for ML training.
Generates 1000 records with proper seasonal/daily variation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config
import random

random.seed(42)
np.random.seed(42)

CITY_PROFILES = {
    'Delhi':     {'base_temp': 25, 'aqi_weights': [0.05, 0.15, 0.30, 0.35, 0.15]},
    'Mumbai':    {'base_temp': 28, 'aqi_weights': [0.10, 0.30, 0.40, 0.15, 0.05]},
    'Bangalore': {'base_temp': 22, 'aqi_weights': [0.30, 0.45, 0.20, 0.05, 0.00]},
    'Chennai':   {'base_temp': 30, 'aqi_weights': [0.10, 0.35, 0.35, 0.15, 0.05]},
    'Kolkata':   {'base_temp': 27, 'aqi_weights': [0.05, 0.15, 0.30, 0.35, 0.15]},
    'Hyderabad': {'base_temp': 26, 'aqi_weights': [0.15, 0.35, 0.35, 0.10, 0.05]},
    'Pune':      {'base_temp': 24, 'aqi_weights': [0.25, 0.45, 0.25, 0.05, 0.00]},
    'Ahmedabad': {'base_temp': 28, 'aqi_weights': [0.10, 0.30, 0.35, 0.20, 0.05]},
}

WEATHER_TYPES = {
    'Clear':       {'hum_range': (25, 55), 'cloud_range': (0, 15),  'prob': 0.22},
    'Clouds':      {'hum_range': (45, 75), 'cloud_range': (40, 85), 'prob': 0.22},
    'Rain':        {'hum_range': (75, 95), 'cloud_range': (85, 100),'prob': 0.17},
    'Drizzle':     {'hum_range': (70, 90), 'cloud_range': (70, 95), 'prob': 0.12},
    'Mist':        {'hum_range': (80, 95), 'cloud_range': (60, 90), 'prob': 0.12},
    'Haze':        {'hum_range': (55, 80), 'cloud_range': (20, 60), 'prob': 0.10},
    'Thunderstorm':{'hum_range': (80, 98), 'cloud_range': (90, 100),'prob': 0.05},
}

WEATHER_DESCS = {
    'Clear':        ['clear sky', 'sunny'],
    'Clouds':       ['few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds'],
    'Rain':         ['light rain', 'moderate rain', 'heavy intensity rain'],
    'Drizzle':      ['light intensity drizzle', 'drizzle', 'heavy intensity drizzle'],
    'Mist':         ['mist'],
    'Haze':         ['haze'],
    'Thunderstorm': ['thunderstorm with light rain', 'thunderstorm with rain',
                     'thunderstorm with heavy rain'],
}

PM25_BY_AQI = {1: (5, 30), 2: (30, 60), 3: (60, 90), 4: (90, 150), 5: (150, 300)}


def generate_sample_data(num_records: int = 1000) -> pd.DataFrame:
    Config.init_app()

    print("=" * 60)
    print("🎲 GENERATING SAMPLE WEATHER DATA")
    print("=" * 60)

    cities   = list(CITY_PROFILES.keys())
    w_types  = list(WEATHER_TYPES.keys())
    w_probs  = [WEATHER_TYPES[w]['prob'] for w in w_types]

    data       = []
    start_date = datetime.now() - timedelta(days=60)

    for i in range(num_records):
        city    = random.choice(cities)
        profile = CITY_PROFILES[city]
        ts      = start_date + timedelta(hours=i * (60 / (num_records / 24)))

        hour  = ts.hour
        month = ts.month

        # Daily temperature cycle (peak ~14:00)
        daily_cycle = np.sin((hour - 6) * np.pi / 12) * 5
        # Seasonal cycle (India: hotter in May-Jun, cooler Dec-Jan)
        seasonal    = np.sin((month - 3) * np.pi / 6) * 4
        base_temp   = profile['base_temp'] + daily_cycle + seasonal
        temperature = round(base_temp + np.random.normal(0, 1.5), 2)

        weather_main = np.random.choice(w_types, p=w_probs)
        wt           = WEATHER_TYPES[weather_main]
        description  = random.choice(WEATHER_DESCS[weather_main])

        humidity = int(np.clip(
            np.random.randint(*wt['hum_range']) + np.random.normal(0, 3),
            10, 100))
        clouds   = int(np.clip(
            np.random.randint(*wt['cloud_range']) + np.random.normal(0, 5),
            0, 100))

        aqi   = np.random.choice([1, 2, 3, 4, 5], p=profile['aqi_weights'])
        pm2_5 = round(np.random.uniform(*PM25_BY_AQI[aqi]), 2)
        pm10  = round(max(0, pm2_5 * 1.5 + np.random.normal(0, 8)), 2)

        feels_offset = (humidity - 50) * 0.05 - (clouds * 0.02)
        feels_like   = round(temperature + feels_offset + np.random.normal(0, 0.5), 2)
        temp_min     = round(temperature - np.random.uniform(1.5, 3.5), 2)
        temp_max     = round(temperature + np.random.uniform(1.5, 3.5), 2)

        record = {
            'timestamp':           ts,
            'city':                city,
            'temperature':         temperature,
            'feels_like':          feels_like,
            'temp_min':            temp_min,
            'temp_max':            temp_max,
            'pressure':            int(1013 + np.random.normal(0, 8)),
            'humidity':            humidity,
            'wind_speed':          round(np.random.uniform(0.3, 9), 2),
            'wind_deg':            np.random.randint(0, 360),
            'clouds':              clouds,
            'weather_main':        weather_main,
            'weather_description': description,
            'aqi':                 int(aqi),
            'pm2_5':               pm2_5,
            'pm10':                pm10,
            'co':  round(np.random.uniform(200, 1200), 2),
            'no2': round(np.random.uniform(5,   80),   2),
            'o3':  round(np.random.uniform(10,  120),  2),
            'so2': round(np.random.uniform(2,   40),   2),
            'hour':  hour,
            'month': month,
        }
        data.append(record)

    df = pd.DataFrame(data)
    df.to_csv(Config.DATASET_FILE, index=False)

    print(f"\n✅ Generated {num_records} records")
    print(f"📁 Saved to: {Config.DATASET_FILE}")
    print(f"\n📊 Summary:")
    print(f"   Cities          : {df['city'].nunique()}")
    print(f"   Date range      : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"   Temp range      : {df['temperature'].min():.1f}°C – {df['temperature'].max():.1f}°C")
    print(f"   Weather types   : {list(df['weather_main'].unique())}")
    print(f"   AQI distribution: {df['aqi'].value_counts().sort_index().to_dict()}")
    print(f"\n🚀 Now run: python train_model.py")
    print("=" * 60)
    return df


if __name__ == "__main__":
    df = generate_sample_data(num_records=1000)
    print("\n📋 Preview:")
    print(df[['timestamp', 'city', 'temperature', 'humidity',
              'weather_main', 'aqi', 'pm2_5']].head(10).to_string(index=False))