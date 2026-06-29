from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from config import Config
from predict import WeatherPredictor
from data_collector import WeatherDataCollector
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

Config.init_app()
predictor = WeatherPredictor()
collector  = WeatherDataCollector()

# ── Load sample/historical data for fallback ──────────────────────────────
try:
    sample_data = pd.read_csv(Config.DATASET_FILE)
    print(f"✅ Loaded {len(sample_data)} historical records")
    has_data = True
except Exception:
    print("⚠️  No historical data — run generate_sample_data.py")
    sample_data = None
    has_data    = False


# ── Helpers ───────────────────────────────────────────────────────────────
def get_aqi_category(aqi):
    return {1: "Good", 2: "Fair", 3: "Moderate",
            4: "Poor",  5: "Very Poor"}.get(int(aqi), "Unknown")


def build_sample_response(city):
    """Build a response dict from sample CSV data (fallback when API unavailable)."""
    if sample_data is None or len(sample_data) == 0:
        return None

    df = sample_data[sample_data['city'] == city]
    if len(df) == 0:
        df = sample_data

    row = df.iloc[-1].to_dict()

    aqi = int(row.get('aqi', 1))
    pm25 = float(row.get('pm2_5', 0) or 0)

    return {
        'city':      city,
        'timestamp': str(row.get('timestamp', '')),
        'source':    'sample_data',
        'current': {
            'temperature': float(row.get('temperature', 0)),
            'feels_like':  float(row.get('feels_like', 0)),
            'temp_min':    float(row.get('temp_min', 0)),
            'temp_max':    float(row.get('temp_max', 0)),
            'humidity':    int(row.get('humidity', 0)),
            'pressure':    int(row.get('pressure', 1013)),
            'wind_speed':  float(row.get('wind_speed', 0)),
            'clouds':      int(row.get('clouds', 0)),
            'weather':     str(row.get('weather_main', 'Unknown')),
            'description': str(row.get('weather_description', '')),
            'aqi':         aqi,
            'aqi_category': get_aqi_category(aqi),
            'pm2_5':       float(row.get('pm2_5', 0) or 0),
            'pm10':        float(row.get('pm10', 0) or 0),
        },
        'health_advice': predictor.get_health_advice(aqi, pm25),
    }


# ── Routes ────────────────────────────────────────────────────────────────
# @app.route('/')
# def index():
#     return render_template('index.html')

# MAINTENANCE_MODE = True

# @app.route('/')
# def index():
#     if MAINTENANCE_MODE:
#         return render_template("maintenance.html")

#     return render_template("index.html")


@app.route('/')
def index():
    return "<!DOCTYPE html><html><head></head><body></body></html>"

@app.route('/dashboard')
def dashboard():
    try:
        from dashboard import create_dashboard
        create_dashboard()
        return render_template('dashboard.html')
    except Exception as e:
        return (f"<h2>Dashboard error: {e}</h2>"
                f"<p>Make sure data/weather_data.csv exists.</p>"), 500


# ── /api/predict  (current weather + ML) ─────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json() or {}
        city = body.get('city', Config.DEFAULT_CITY).strip()

        # 1. Try live API
        result = predictor.predict_for_city(city)
        if 'error' not in result:
            result['source'] = 'live_api'
            return jsonify({'success': True, 'data': result})

        # 2. Fallback to sample data
        print(f"⚠️  Live API failed for {city}: {result.get('error')}")
        fallback = build_sample_response(city)
        if fallback:
            fallback['warning'] = ('Live API unavailable — showing sample data. '
                                   'Check your OPENWEATHER_API_KEY.')
            return jsonify({'success': True, 'data': fallback})

        return jsonify({'success': False,
                        'error': (f'No data for {city}. '
                                  'Check API key or run generate_sample_data.py')}), 404

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── /api/forecast/<city>  (10-day) ────────────────────────────────────────
@app.route('/api/forecast/<city>', methods=['GET'])
def forecast(city):
    """10-day forecast via Open-Meteo (free, always works)."""
    try:
        result = predictor.get_10day_forecast(city)
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 404
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── /api/weather/<city>  (raw current) ────────────────────────────────────
@app.route('/api/weather/<city>', methods=['GET'])
def get_weather(city):
    try:
        data = collector.fetch_weather_data(city)
        if not data:
            return jsonify({'success': False,
                            'error': f'Could not fetch weather for {city}'}), 404
        # Convert datetime to string for JSON
        if 'timestamp' in data:
            data['timestamp'] = str(data['timestamp'])
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ── /api/cities ────────────────────────────────────────────────────────────
@app.route('/api/cities', methods=['GET'])
def get_cities():
    return jsonify({'success': True, 'cities': Config.CITIES})


# ── /api/chat ──────────────────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        body    = request.get_json() or {}
        message = body.get('message', '').lower()
        city    = body.get('city', Config.DEFAULT_CITY).strip()

        # Get prediction (live or fallback)
        predictions = predictor.predict_for_city(city)
        if 'error' in predictions:
            predictions = build_sample_response(city)
        if not predictions:
            return jsonify({'success': False,
                            'reply': 'No data available. Run generate_sample_data.py first!'})

        current = predictions['current']

        # Forecast intent
        if any(w in message for w in ['forecast', '10 day', '10-day', 'week', 'next']):
            fc = predictor.get_10day_forecast(city)
            if 'forecast' in fc:
                lines = [f"📅 10-Day Forecast for {city}:\n"]
                for d in fc['forecast'][:10]:
                    lines.append(
                        f"{d['date']}  {d['icon']} {d['condition']}  "
                        f"{d['temp_min']}–{d['temp_max']}°C  "
                        f"Rain: {d['precip_prob']}%")
                return jsonify({'success': True, 'reply': '\n'.join(lines),
                                'data': fc})

        # Weather intent
        if any(w in message for w in ['weather', 'temperature', 'temp',
                                       'forecast', 'climate', 'how is']):
            reply  = f"🌍 Weather in {city}:\n\n"
            reply += f"🌡️ Temperature : {current['temperature']}°C (Feels like {current['feels_like']}°C)\n"
            reply += f"📊 Range       : {current['temp_min']}°C – {current['temp_max']}°C\n"
            reply += f"☁️ Condition   : {current['weather']} — {current['description']}\n"
            reply += f"💧 Humidity    : {current['humidity']}%\n"
            reply += f"💨 Wind        : {current['wind_speed']} m/s\n\n"
            reply += f"🏭 Air Quality : {current['aqi_category']} (AQI {current['aqi']})\n"
            reply += f"   PM2.5: {current['pm2_5']} µg/m³\n\n"
            if 'ml_predictions' in predictions:
                ml = predictions['ml_predictions']
                reply += f"🤖 ML Predicted Temp: {ml.get('predicted_temperature','N/A')}°C\n\n"
            reply += f"💡 {predictions.get('health_advice', '')}"
            return jsonify({'success': True, 'reply': reply, 'data': predictions})

        # AQI intent
        if any(w in message for w in ['aqi', 'air quality', 'pollution',
                                       'pm2.5', 'pm10']):
            reply  = f"💨 Air Quality in {city}:\n\n"
            reply += f"AQI Level : {current['aqi_category']} ({current['aqi']})\n"
            reply += f"PM2.5     : {current['pm2_5']} µg/m³\n"
            reply += f"PM10      : {current['pm10']} µg/m³\n\n"
            reply += f"💡 {predictions.get('health_advice', '')}"
            return jsonify({'success': True, 'reply': reply, 'data': predictions})

        return jsonify({
            'success': True,
            'reply': ("Hi! I can help with:\n"
                      "• Current weather & temperature\n"
                      "• 10-day forecast\n"
                      "• Air quality (AQI, PM2.5)\n"
                      "• Health advice\n\n"
                      "Try: 'weather in Delhi' or '10-day forecast Mumbai'")
        })

    except Exception as e:
        return jsonify({'success': False,
                        'reply': f"Sorry, something went wrong: {e}"}), 500


# ── /api/health ────────────────────────────────────────────────────────────
@app.route('/api/health')
def health_check():
    api_key_ok = bool(Config.OPENWEATHER_API_KEY)
    models_ok  = predictor.temp_model is not None
    return jsonify({
        'status':         'healthy',
        'api_key_set':    api_key_ok,
        'models_loaded':  models_ok,
        'data_available': has_data,
        'message':        'WeatherML API running',
    })


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 WeatherML Flask Server")
    print("=" * 60)
    print(f"  Main      : http://127.0.0.1:5000")
    print(f"  Dashboard : http://127.0.0.1:5000/dashboard")
    print(f"  API Key   : {'✅ Set' if Config.OPENWEATHER_API_KEY else '❌ Missing (.env)'}")
    print(f"  Models    : {'✅ Loaded' if predictor.temp_model else '❌ Run train_model.py'}")
    print(f"  Data      : {'✅ Available' if has_data else '❌ Run generate_sample_data.py'}")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)