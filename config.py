import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # API Keys
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
    
    # OpenWeatherMap API URLs
    WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
    AIR_POLLUTION_API_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
    FORECAST_API_URL = "https://api.openweathermap.org/data/2.5/forecast"
    GEO_API_URL = "http://api.openweathermap.org/geo/1.0/direct"
    
    # Open-Meteo (FREE, no API key needed) — used for 10-day forecast
    OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    
    # Default Settings
    DEFAULT_CITY = os.getenv('DEFAULT_CITY', 'Delhi')
    
    # File Paths
    DATA_DIR = 'data'
    MODEL_DIR = 'models'
    DATASET_FILE = os.path.join(DATA_DIR, 'weather_data.csv')
    
    # Model Files
    TEMP_MODEL_FILE    = os.path.join(MODEL_DIR, 'temperature_model.joblib')
    WEATHER_MODEL_FILE = os.path.join(MODEL_DIR, 'weather_classifier.joblib')
    HUMIDITY_MODEL_FILE= os.path.join(MODEL_DIR, 'humidity_model.joblib')
    SCALER_FILE        = os.path.join(MODEL_DIR, 'scaler.joblib')
    HUM_SCALER_FILE    = os.path.join(MODEL_DIR, 'hum_scaler.joblib')   # NEW: separate scaler
    
    # Cities to track
    CITIES = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata',
              'Hyderabad', 'Pune', 'Ahmedabad']
    
    @staticmethod
    def init_app():
        """Initialize application directories"""
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)