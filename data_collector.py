import requests
import pandas as pd
from datetime import datetime
import os
from config import Config


class WeatherDataCollector:
    """Collects real-time weather + AQI data from OpenWeatherMap,
       and 10-day forecasts from Open-Meteo (free, no key required)."""

    def __init__(self):
        self.api_key = Config.OPENWEATHER_API_KEY

    # ------------------------------------------------------------------ #
    #  COORDINATES                                                          #
    # ------------------------------------------------------------------ #
    def get_coordinates(self, city_name):
        """Lat/lon via OpenWeatherMap Geo API (needs key),
           falls back to Open-Meteo Geocoding (free)."""
        # Try OWM first
        if self.api_key:
            try:
                url = (f"{Config.GEO_API_URL}?q={city_name}"
                       f"&limit=1&appid={self.api_key}")
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json()
                if data:
                    return data[0]['lat'], data[0]['lon']
            except Exception as e:
                print(f"OWM geo failed for {city_name}: {e}")

        # Free fallback: Open-Meteo Geocoding
        try:
            url = (f"{Config.OPEN_METEO_GEO_URL}"
                   f"?name={city_name}&count=1&language=en&format=json")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            results = data.get('results', [])
            if results:
                return results[0]['latitude'], results[0]['longitude']
        except Exception as e:
            print(f"Open-Meteo geo failed for {city_name}: {e}")

        return None, None

    # ------------------------------------------------------------------ #
    #  CURRENT WEATHER  (OpenWeatherMap)                                   #
    # ------------------------------------------------------------------ #
    def fetch_weather_data(self, city_name):
        """Fetch current weather + AQI from OWM.
        Returns a flat dict with CONSISTENT field names used everywhere."""
        if not self.api_key:
            print("⚠️  No OWM API key — cannot fetch live weather.")
            return None

        try:
            lat, lon = self.get_coordinates(city_name)
            if lat is None:
                return None

            # Weather
            w_url = (f"{Config.WEATHER_API_URL}"
                     f"?lat={lat}&lon={lon}&appid={self.api_key}&units=metric")
            wr = requests.get(w_url, timeout=10)
            wr.raise_for_status()
            w = wr.json()

            # AQI
            a_url = (f"{Config.AIR_POLLUTION_API_URL}"
                     f"?lat={lat}&lon={lon}&appid={self.api_key}")
            ar = requests.get(a_url, timeout=10)
            ar.raise_for_status()
            a = ar.json()

            return self._parse_current(city_name, w, a)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching OWM data for {city_name}: {e}")
            return None

    def _parse_current(self, city, w, a):
        """Parse OWM response into a flat, consistently-named dict."""
        try:
            main       = w.get('main', {})
            wind       = w.get('wind', {})
            clouds_d   = w.get('clouds', {})
            weather_d  = w.get('weather', [{}])[0]
            aqi_list   = a.get('list', [{}])[0]
            aqi_main   = aqi_list.get('main', {})
            components = aqi_list.get('components', {})

            return {
                # identifiers
                'timestamp':           datetime.now(),
                'city':                city,
                # temperature group
                'temperature':         main.get('temp'),
                'feels_like':          main.get('feels_like'),
                'temp_min':            main.get('temp_min'),
                'temp_max':            main.get('temp_max'),
                # atmosphere
                'pressure':            main.get('pressure'),
                'humidity':            main.get('humidity'),
                # wind
                'wind_speed':          wind.get('speed'),
                'wind_deg':            wind.get('deg', 0),
                # sky
                'clouds':              clouds_d.get('all'),
                # weather condition — CONSISTENT names used by predict.py
                'weather_main':        weather_d.get('main'),
                'weather_description': weather_d.get('description'),
                # air quality
                'aqi':                 aqi_main.get('aqi'),
                'pm2_5':               components.get('pm2_5'),
                'pm10':                components.get('pm10'),
                'co':                  components.get('co'),
                'no2':                 components.get('no2'),
                'o3':                  components.get('o3'),
                'so2':                 components.get('so2'),
            }
        except Exception as e:
            print(f"Error parsing OWM data: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  10-DAY FORECAST  (Open-Meteo — FREE, no key)                        #
    # ------------------------------------------------------------------ #
    def fetch_10day_forecast(self, city_name):
        """Fetch a 10-day hourly forecast from Open-Meteo (free API).
        Returns a list of daily summary dicts."""
        try:
            lat, lon = self.get_coordinates(city_name)
            if lat is None:
                print(f"Could not resolve coordinates for {city_name}")
                return None

            params = {
                'latitude':  lat,
                'longitude': lon,
                'daily': [
                    'temperature_2m_max',
                    'temperature_2m_min',
                    'precipitation_sum',
                    'precipitation_probability_max',
                    'weathercode',
                    'windspeed_10m_max',
                    'relative_humidity_2m_max',
                    'relative_humidity_2m_min',
                    'uv_index_max',
                    'apparent_temperature_max',
                    'apparent_temperature_min',
                ],
                'timezone':    'auto',
                'forecast_days': 10,
            }

            r = requests.get(Config.OPEN_METEO_FORECAST_URL,
                             params=params, timeout=15)
            r.raise_for_status()
            raw = r.json()

            return self._parse_10day(raw)

        except Exception as e:
            print(f"Error fetching 10-day forecast for {city_name}: {e}")
            return None

    def _parse_10day(self, raw):
        """Convert Open-Meteo daily response into clean list of dicts."""
        daily   = raw.get('daily', {})
        dates   = daily.get('time', [])
        n       = len(dates)
        if n == 0:
            return []

        def safe(key, i, default=0):
            lst = daily.get(key, [])
            return lst[i] if i < len(lst) and lst[i] is not None else default

        results = []
        for i in range(n):
            wcode      = int(safe('weathercode', i))
            condition, icon = self._wmo_to_condition(wcode)
            results.append({
                'date':              dates[i],
                'temp_max':          round(safe('temperature_2m_max', i), 1),
                'temp_min':          round(safe('temperature_2m_min', i), 1),
                'temp_avg':          round((safe('temperature_2m_max', i) +
                                            safe('temperature_2m_min', i)) / 2, 1),
                'feels_max':         round(safe('apparent_temperature_max', i), 1),
                'feels_min':         round(safe('apparent_temperature_min', i), 1),
                'humidity_max':      int(safe('relative_humidity_2m_max', i)),
                'humidity_min':      int(safe('relative_humidity_2m_min', i)),
                'precipitation':     round(safe('precipitation_sum', i), 1),
                'precip_prob':       int(safe('precipitation_probability_max', i)),
                'wind_max':          round(safe('windspeed_10m_max', i), 1),
                'uv_index':          round(safe('uv_index_max', i), 1),
                'weather_code':      wcode,
                'condition':         condition,
                'icon':              icon,
            })
        return results

    @staticmethod
    def _wmo_to_condition(code):
        """Map WMO weather interpretation codes to human label + emoji."""
        mapping = {
            0:  ('Clear Sky',          '☀️'),
            1:  ('Mainly Clear',       '🌤️'),
            2:  ('Partly Cloudy',      '⛅'),
            3:  ('Overcast',           '☁️'),
            45: ('Fog',                '🌫️'),
            48: ('Icy Fog',            '🌫️'),
            51: ('Light Drizzle',      '🌦️'),
            53: ('Moderate Drizzle',   '🌦️'),
            55: ('Heavy Drizzle',      '🌧️'),
            61: ('Light Rain',         '🌧️'),
            63: ('Moderate Rain',      '🌧️'),
            65: ('Heavy Rain',         '🌧️'),
            71: ('Light Snow',         '❄️'),
            73: ('Moderate Snow',      '🌨️'),
            75: ('Heavy Snow',         '❄️'),
            77: ('Snow Grains',        '🌨️'),
            80: ('Light Showers',      '🌦️'),
            81: ('Moderate Showers',   '🌧️'),
            82: ('Heavy Showers',      '⛈️'),
            85: ('Snow Showers',       '🌨️'),
            86: ('Heavy Snow Showers', '❄️'),
            95: ('Thunderstorm',       '⛈️'),
            96: ('Thunderstorm+Hail',  '⛈️'),
            99: ('Heavy Thunderstorm', '⛈️'),
        }
        return mapping.get(code, ('Unknown', '🌤️'))

    # ------------------------------------------------------------------ #
    #  BULK COLLECT & SAVE                                                  #
    # ------------------------------------------------------------------ #
    def collect_and_save(self, cities=None):
        """Collect current data for multiple cities and append to CSV."""
        if cities is None:
            cities = Config.CITIES

        all_data = []
        for city in cities:
            print(f"Fetching data for {city}...")
            data = self.fetch_weather_data(city)
            if data:
                all_data.append(data)

        if all_data:
            df = pd.DataFrame(all_data)
            if os.path.exists(Config.DATASET_FILE):
                existing = pd.read_csv(Config.DATASET_FILE)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(Config.DATASET_FILE, index=False)
            print(f"\n✅ Data saved! Total records: {len(df)}")
            return df

        return None


# ------------------------------------------------------------------ #
#  STANDALONE TEST                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    Config.init_app()
    collector = WeatherDataCollector()

    city = input("City (Enter = Delhi): ").strip() or "Delhi"

    print("\n--- Current weather ---")
    current = collector.fetch_weather_data(city)
    if current:
        print(current)
    else:
        print("No current data (check API key).")

    print("\n--- 10-day forecast ---")
    forecast = collector.fetch_10day_forecast(city)
    if forecast:
        for day in forecast:
            print(f"{day['date']}  {day['icon']}  {day['condition']}"
                  f"  {day['temp_min']}–{day['temp_max']}°C"
                  f"  Rain: {day['precip_prob']}%")
    else:
        print("Could not fetch forecast.")