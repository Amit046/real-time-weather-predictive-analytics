from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from config import Config
from predict import WeatherPredictor
from data_collector import WeatherDataCollector
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Initialize
Config.init_app()
predictor = WeatherPredictor()
collector = WeatherDataCollector()

# Load sample data for testing
try:
    sample_data = pd.read_csv(Config.DATASET_FILE)
    print(f"✅ Loaded {len(sample_data)} sample records")
    has_data = True
except:
    print("⚠️ No sample data found")
    sample_data = None
    has_data = False

def get_sample_city_data(city):
    """Get sample data for testing if API fails"""
    if sample_data is None or len(sample_data) == 0:
        return None
    
    city_data = sample_data[sample_data['city'] == city]
    if len(city_data) == 0:
        city_data = sample_data
    
    latest = city_data.iloc[-1].to_dict()
    
    return {
        'city': city,
        'timestamp': str(latest['timestamp']),
        'current': {
            'temperature': float(latest['temperature']),
            'feels_like': float(latest['feels_like']),
            'humidity': int(latest['humidity']),
            'pressure': int(latest['pressure']),
            'wind_speed': float(latest['wind_speed']),
            'clouds': int(latest['clouds']),
            'weather': str(latest['weather_main']),
            'description': str(latest['weather_description']),
            'aqi': int(latest['aqi']),
            'aqi_category': get_aqi_category(int(latest['aqi'])),
            'pm2_5': float(latest['pm2_5']),
            'pm10': float(latest['pm10'])
        }
    }

def get_aqi_category(aqi):
    """Convert AQI number to category"""
    categories = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    return categories.get(aqi, "Unknown")

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Show analytics dashboard"""
    try:
        from dashboard import create_dashboard
        create_dashboard()
        return render_template('dashboard.html')
    except Exception as e:
        return f"Error creating dashboard: {e}. Make sure you have data in data/weather_data.csv", 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Get weather prediction for a city"""
    try:
        data = request.get_json()
        city = data.get('city', Config.DEFAULT_CITY)
        
        # Try real API first
        try:
            predictions = predictor.predict_for_city(city)
            if 'error' not in predictions:
                return jsonify({
                    'success': True,
                    'data': predictions
                })
        except:
            pass
        
        # Fallback to sample data
        if has_data:
            sample_predictions = get_sample_city_data(city)
            if sample_predictions:
                sample_predictions['health_advice'] = predictor.get_health_advice(
                    sample_predictions['current']['aqi'],
                    sample_predictions['current']['pm2_5']
                )
                return jsonify({
                    'success': True,
                    'data': sample_predictions
                })
        
        return jsonify({
            'success': False,
            'error': f'Could not fetch data for {city}. Check API key or generate sample data.'
        }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/weather/<city>', methods=['GET'])
def get_weather(city):
    """Get current weather for a city"""
    try:
        weather_data = collector.fetch_weather_data(city)
        
        if not weather_data:
            return jsonify({
                'success': False,
                'error': f'Could not fetch weather data for {city}'
            }), 404
        
        return jsonify({
            'success': True,
            'data': weather_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of available cities"""
    return jsonify({
        'success': True,
        'cities': Config.CITIES
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat interface for weather queries"""
    try:
        data = request.get_json()
        message = data.get('message', '').lower()
        city = data.get('city', Config.DEFAULT_CITY)
        
        # Try real prediction first
        try:
            predictions = predictor.predict_for_city(city)
            if 'error' in predictions:
                raise Exception("API failed")
        except:
            # Use sample data
            if has_data:
                predictions = get_sample_city_data(city)
                predictions['health_advice'] = predictor.get_health_advice(
                    predictions['current']['aqi'],
                    predictions['current']['pm2_5']
                )
            else:
                return jsonify({
                    'success': False,
                    'reply': 'No data available. Please run generate_sample_data.py first!'
                })
        
        current = predictions['current']
        
        if any(word in message for word in ['weather', 'temperature', 'temp', 'forecast', 'climate']):
            reply = f"🌍 Weather in {city}:\n\n"
            reply += f"🌡️ Temperature: {current['temperature']}°C (Feels like {current['feels_like']}°C)\n"
            reply += f"☁️ Condition: {current['weather']} - {current['description']}\n"
            reply += f"💧 Humidity: {current['humidity']}%\n"
            reply += f"💨 Wind: {current['wind_speed']} m/s\n\n"
            reply += f"🏭 Air Quality: {current['aqi_category']} (AQI: {current['aqi']})\n"
            reply += f"PM2.5: {current['pm2_5']} µg/m³\n\n"
            
            if 'ml_predictions' in predictions:
                ml_pred = predictions['ml_predictions']
                reply += f"🤖 ML Prediction: Temperature will be around {ml_pred['predicted_temperature']}°C\n\n"
            
            reply += f"💡 {predictions.get('health_advice', '')}"
            
            return jsonify({
                'success': True,
                'reply': reply,
                'data': predictions
            })
        
        elif any(word in message for word in ['aqi', 'air quality', 'pollution', 'pm2.5']):
            reply = f"💨 Air Quality in {city}:\n\n"
            reply += f"AQI Level: {current['aqi_category']} ({current['aqi']})\n"
            reply += f"PM2.5: {current['pm2_5']} µg/m³\n"
            reply += f"PM10: {current['pm10']} µg/m³\n\n"
            reply += f"💡 {predictions.get('health_advice', '')}"
            
            return jsonify({
                'success': True,
                'reply': reply,
                'data': predictions
            })
        
        else:
            return jsonify({
                'success': True,
                'reply': "Hi! I can help you with:\n• Weather information\n• Temperature forecasts\n• Air quality (AQI)\n• Health advice\n\nJust ask me about weather or AQI for any city!"
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'reply': f"Sorry, something went wrong: {str(e)}"
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Weather ML API is running',
        'data_available': has_data
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Weather ML Flask Server")
    print("="*60)
    print(f"📍 Main App: http://127.0.0.1:5000")
    print(f"📊 Dashboard: http://127.0.0.1:5000/dashboard")
    print(f"✅ Data: {'Available' if has_data else 'Not available - run generate_sample_data.py'}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)