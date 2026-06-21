"""
Dynamic Analytics Dashboard
Reads fresh data from CSV every time — no hardcoded values.
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime
from config import Config


def create_dashboard():
    try:
        df = pd.read_csv(Config.DATASET_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        print(f"✅ Loaded {len(df)} records for dashboard")
    except Exception as e:
        print(f"❌ Could not load data: {e}")
        return None

    # ── Stats ──────────────────────────────────────────────────────────
    stats = {
        'total_records': len(df),
        'cities':        int(df['city'].nunique()),
        'avg_temp':      round(float(df['temperature'].mean()), 2),
        'max_temp':      round(float(df['temperature'].max()), 2),
        'min_temp':      round(float(df['temperature'].min()), 2),
        'avg_humidity':  round(float(df['humidity'].mean()), 2),
        'avg_aqi':       round(float(df['aqi'].mean()), 2),
    }

    # ── Per-city aggregates ────────────────────────────────────────────
    temp_by_city     = df.groupby('city')['temperature'].mean().round(2).sort_values(ascending=False)
    aqi_by_city      = df.groupby('city')['aqi'].mean().round(2).sort_values(ascending=False)
    humidity_by_city = df.groupby('city')['humidity'].mean().round(2).sort_values(ascending=False)
    pm25_by_city     = df.groupby('city')['pm2_5'].mean().round(2).sort_values(ascending=False)
    wind_by_city     = df.groupby('city')['wind_speed'].mean().round(2)

    weather_dist = df['weather_main'].value_counts().to_dict()
    aqi_dist     = df['aqi'].value_counts().sort_index().to_dict()

    # ── Hourly averages (real data) ────────────────────────────────────
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    hourly_temp = (df.groupby('hour')['temperature']
                     .mean().round(2).reindex(range(24), fill_value=stats['avg_temp'])
                     .tolist())
    hourly_hum  = (df.groupby('hour')['humidity']
                     .mean().round(2).reindex(range(24), fill_value=stats['avg_humidity'])
                     .tolist())

    # ── Radar dataset for top 3 cities ────────────────────────────────
    top3 = list(temp_by_city.head(3).index)
    # Normalise values to 0-100 for radar
    temp_max_val  = float(temp_by_city.max())   or 1
    hum_max_val   = float(humidity_by_city.max()) or 1
    pm25_max_val  = float(pm25_by_city.max())   or 1
    wind_max_val  = float(wind_by_city.max())   or 1

    radar_datasets = []
    colours = ['rgba(0,255,136,0.4)', 'rgba(0,255,255,0.4)', 'rgba(255,170,0,0.4)']
    border  = ['#00ff88', '#00ffff', '#ffaa00']
    for idx, city in enumerate(top3):
        radar_datasets.append({
            'label':           city,
            'data': [
                round(float(temp_by_city.get(city, 0))     / temp_max_val  * 100, 1),
                round(float(humidity_by_city.get(city, 0)) / hum_max_val   * 100, 1),
                round(float(aqi_by_city.get(city, 0))      / 5             * 100, 1),
                round(float(pm25_by_city.get(city, 0))     / pm25_max_val  * 100, 1),
                round(float(wind_by_city.get(city, 0))     / wind_max_val  * 100, 1),
            ],
            'backgroundColor': colours[idx],
            'borderColor':     border[idx],
            'borderWidth': 2,
            'pointBackgroundColor': border[idx],
        })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WeatherML – Analytics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:'Rajdhani',sans-serif;background:#000;color:#00ff88;overflow-x:hidden}}
        .grid-bg{{position:fixed;top:0;left:0;width:100%;height:100%;
            background:linear-gradient(90deg,rgba(0,255,136,.03) 1px,transparent 1px),
                       linear-gradient(rgba(0,255,136,.03) 1px,transparent 1px);
            background-size:50px 50px;animation:gs 20s linear infinite;z-index:0}}
        @keyframes gs{{0%{{transform:translate(0,0)}}100%{{transform:translate(50px,50px)}}}}
        .orb{{position:fixed;border-radius:50%;filter:blur(80px);opacity:.35;animation:fl 20s ease-in-out infinite;z-index:0}}
        .orb1{{width:500px;height:500px;background:radial-gradient(circle,#00ff88,transparent);top:-250px;left:-250px}}
        .orb2{{width:400px;height:400px;background:radial-gradient(circle,#00ffff,transparent);bottom:-200px;right:-200px;animation-delay:-10s}}
        @keyframes fl{{0%,100%{{transform:translate(0,0) scale(1)}}50%{{transform:translate(80px,80px) scale(1.15)}}}}
        .container{{position:relative;z-index:1;max-width:1900px;margin:0 auto;padding:20px}}
        .top-bar{{background:rgba(0,0,0,.85);backdrop-filter:blur(20px);
            border-bottom:2px solid rgba(0,255,136,.3);padding:20px 40px;
            display:flex;justify-content:space-between;align-items:center;
            margin-bottom:30px;box-shadow:0 0 30px rgba(0,255,136,.2)}}
        .logo{{font-family:'Orbitron',sans-serif;font-size:1.8em;font-weight:900;
            letter-spacing:3px;text-shadow:0 0 20px rgba(0,255,136,.8)}}
        .top-meta{{color:rgba(0,255,136,.6);font-size:.9em}}
        .nav-btn{{padding:10px 25px;background:linear-gradient(135deg,rgba(0,255,136,.2),rgba(0,255,255,.2));
            border:2px solid #00ff88;color:#00ff88;border-radius:8px;
            font-weight:700;text-decoration:none;transition:all .3s;
            text-transform:uppercase;letter-spacing:1px}}
        .nav-btn:hover{{background:#00ff88;color:#000;box-shadow:0 0 30px rgba(0,255,136,.8)}}
        .stats-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:20px;margin-bottom:30px}}
        .stat-card{{background:rgba(0,0,0,.7);backdrop-filter:blur(20px);
            border:2px solid rgba(0,255,136,.3);border-radius:15px;padding:25px;
            text-align:center;transition:all .4s;position:relative;overflow:hidden}}
        .stat-card:hover{{border-color:#00ff88;box-shadow:0 0 40px rgba(0,255,136,.4);transform:translateY(-5px)}}
        .stat-icon{{font-size:2.2em;margin-bottom:10px}}
        .stat-value{{font-size:2.3em;font-weight:700;font-family:'Orbitron',sans-serif;
            text-shadow:0 0 20px rgba(0,255,136,.8);margin:8px 0}}
        .stat-label{{color:rgba(0,255,136,.6);font-size:.85em;text-transform:uppercase;letter-spacing:1px}}
        .charts-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:25px;margin-bottom:25px}}
        .chart-card{{background:rgba(0,0,0,.7);backdrop-filter:blur(20px);
            border:2px solid rgba(0,255,136,.3);border-radius:20px;padding:30px;transition:all .4s}}
        .chart-card:hover{{border-color:#00ff88;box-shadow:0 0 50px rgba(0,255,136,.3)}}
        .chart-card.full{{grid-column:1/-1}}
        .chart-header{{display:flex;align-items:center;gap:15px;margin-bottom:25px;
            padding-bottom:15px;border-bottom:2px solid rgba(0,255,136,.2)}}
        .chart-icon{{font-size:1.8em}}
        .chart-title{{font-family:'Orbitron',sans-serif;font-size:1.2em;
            font-weight:600;text-transform:uppercase;letter-spacing:2px}}
        .chart-container{{position:relative;height:340px}}
        .chart-container.large{{height:440px}}
        ::-webkit-scrollbar{{width:10px}}
        ::-webkit-scrollbar-track{{background:rgba(0,255,136,.05)}}
        ::-webkit-scrollbar-thumb{{background:linear-gradient(135deg,#00ff88,#00ffff);border-radius:10px}}
        @media(max-width:1400px){{.stats-grid{{grid-template-columns:repeat(3,1fr)}}}}
        @media(max-width:1024px){{.charts-grid,.stats-grid{{grid-template-columns:1fr}}}}
    </style>
</head>
<body>
<div class="grid-bg"></div>
<div class="orb orb1"></div><div class="orb orb2"></div>
<div class="container">
    <div class="top-bar">
        <div class="logo">⚡ ANALYTICS DASHBOARD</div>
        <div class="top-meta">Generated: {datetime.now().strftime('%d %b %Y %H:%M')}</div>
        <a href="/" class="nav-btn">← Back to App</a>
    </div>

    <div class="stats-grid">
        <div class="stat-card"><div class="stat-icon">📊</div>
            <div class="stat-value">{stats['total_records']}</div>
            <div class="stat-label">Total Records</div></div>
        <div class="stat-card"><div class="stat-icon">🏙️</div>
            <div class="stat-value">{stats['cities']}</div>
            <div class="stat-label">Cities</div></div>
        <div class="stat-card"><div class="stat-icon">🌡️</div>
            <div class="stat-value">{stats['avg_temp']}°</div>
            <div class="stat-label">Avg Temperature</div></div>
        <div class="stat-card"><div class="stat-icon">💧</div>
            <div class="stat-value">{stats['avg_humidity']}%</div>
            <div class="stat-label">Avg Humidity</div></div>
        <div class="stat-card"><div class="stat-icon">💨</div>
            <div class="stat-value">{stats['avg_aqi']}</div>
            <div class="stat-label">Avg AQI</div></div>
    </div>

    <div class="charts-grid">
        <div class="chart-card">
            <div class="chart-header"><span class="chart-icon">🌡️</span>
                <span class="chart-title">Temperature by City</span></div>
            <div class="chart-container"><canvas id="tempChart"></canvas></div>
        </div>
        <div class="chart-card">
            <div class="chart-header"><span class="chart-icon">💨</span>
                <span class="chart-title">AQI by City</span></div>
            <div class="chart-container"><canvas id="aqiChart"></canvas></div>
        </div>
        <div class="chart-card">
            <div class="chart-header"><span class="chart-icon">💧</span>
                <span class="chart-title">Humidity by City</span></div>
            <div class="chart-container"><canvas id="humChart"></canvas></div>
        </div>
        <div class="chart-card">
            <div class="chart-header"><span class="chart-icon">☁️</span>
                <span class="chart-title">Weather Distribution</span></div>
            <div class="chart-container"><canvas id="weatherChart"></canvas></div>
        </div>
        <div class="chart-card full">
            <div class="chart-header"><span class="chart-icon">📈</span>
                <span class="chart-title">Hourly Temperature & Humidity Averages</span></div>
            <div class="chart-container large"><canvas id="hourlyChart"></canvas></div>
        </div>
        <div class="chart-card">
            <div class="chart-header"><span class="chart-icon">🏭</span>
                <span class="chart-title">PM2.5 Pollution by City</span></div>
            <div class="chart-container"><canvas id="pm25Chart"></canvas></div>
        </div>
        <div class="chart-card">
            <div class="chart-header"><span class="chart-icon">🎯</span>
                <span class="chart-title">City Radar (Top 3)</span></div>
            <div class="chart-container"><canvas id="radarChart"></canvas></div>
        </div>
    </div>
</div>

<script>
Chart.defaults.color = '#00ff88';
Chart.defaults.borderColor = 'rgba(0,255,136,0.15)';
Chart.defaults.font.family = "'Rajdhani', sans-serif";

const COLORS = [
    'rgba(0,255,136,0.7)','rgba(0,255,255,0.7)','rgba(255,170,0,0.7)',
    'rgba(255,0,68,0.7)','rgba(136,0,255,0.7)','rgba(255,100,0,0.7)',
    'rgba(0,180,255,0.7)','rgba(200,255,0,0.7)'
];
const BORDERS = ['#00ff88','#00ffff','#ffaa00','#ff0044','#8800ff','#ff6400','#00b4ff','#c8ff00'];

const scaleOpts = {{
    y: {{ grid:{{color:'rgba(0,255,136,0.1)'}}, ticks:{{color:'#00ff88'}} }},
    x: {{ grid:{{display:false}},             ticks:{{color:'#00ff88'}} }}
}};

// Temperature bar chart
new Chart('tempChart', {{
    type:'bar',
    data:{{
        labels:{json.dumps(list(temp_by_city.index))},
        datasets:[{{
            label:'Avg Temp (°C)',
            data:{json.dumps(list(temp_by_city.values.tolist()))},
            backgroundColor:COLORS,
            borderColor:BORDERS,
            borderWidth:2,borderRadius:8
        }}]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}}}},scales:scaleOpts}}
}});

// AQI polar
new Chart('aqiChart', {{
    type:'polarArea',
    data:{{
        labels:{json.dumps(list(aqi_by_city.index))},
        datasets:[{{
            data:{json.dumps(list(aqi_by_city.values.tolist()))},
            backgroundColor:COLORS,borderColor:BORDERS,borderWidth:2
        }}]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{position:'right',labels:{{color:'#00ff88',padding:12}}}}}},
        scales:{{r:{{ticks:{{color:'#00ff88',backdropColor:'transparent'}},
                     grid:{{color:'rgba(0,255,136,0.2)'}}}}}}}}
}});

// Humidity line
new Chart('humChart', {{
    type:'line',
    data:{{
        labels:{json.dumps(list(humidity_by_city.index))},
        datasets:[{{
            label:'Avg Humidity (%)',
            data:{json.dumps(list(humidity_by_city.values.tolist()))},
            backgroundColor:'rgba(0,255,255,0.15)',borderColor:'#00ffff',
            borderWidth:3,fill:true,tension:0.4,
            pointRadius:6,pointBackgroundColor:'#00ffff',pointBorderColor:'#000',pointBorderWidth:2
        }}]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{labels:{{color:'#00ff88'}}}}}},scales:scaleOpts}}
}});

// Weather doughnut
new Chart('weatherChart', {{
    type:'doughnut',
    data:{{
        labels:{json.dumps(list(weather_dist.keys()))},
        datasets:[{{
            data:{json.dumps(list(weather_dist.values()))},
            backgroundColor:COLORS,borderColor:'#000',borderWidth:3
        }}]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{position:'bottom',labels:{{color:'#00ff88',padding:12}}}}}}}}
}});

// Hourly line
new Chart('hourlyChart', {{
    type:'line',
    data:{{
        labels:Array.from({{length:24}},(_,i)=>i+':00'),
        datasets:[
            {{label:'Temperature (°C)',data:{json.dumps(hourly_temp)},
              borderColor:'#00ff88',backgroundColor:'rgba(0,255,136,0.15)',
              borderWidth:3,fill:true,tension:0.4,yAxisID:'y'}},
            {{label:'Humidity (%)',data:{json.dumps(hourly_hum)},
              borderColor:'#00ffff',backgroundColor:'rgba(0,255,255,0.15)',
              borderWidth:3,fill:true,tension:0.4,yAxisID:'y1'}}
        ]
    }},
    options:{{
        responsive:true,maintainAspectRatio:false,
        interaction:{{mode:'index',intersect:false}},
        plugins:{{legend:{{labels:{{color:'#00ff88',padding:20}}}}}},
        scales:{{
            y:{{type:'linear',position:'left',
               grid:{{color:'rgba(0,255,136,0.1)'}},ticks:{{color:'#00ff88'}},
               title:{{display:true,text:'Temperature (°C)',color:'#00ff88'}}}},
            y1:{{type:'linear',position:'right',grid:{{display:false}},
                ticks:{{color:'#00ffff'}},
                title:{{display:true,text:'Humidity (%)',color:'#00ffff'}}}},
            x:{{grid:{{display:false}},ticks:{{color:'#00ff88'}}}}
        }}
    }}
}});

// PM2.5 bar
new Chart('pm25Chart', {{
    type:'bar',
    data:{{
        labels:{json.dumps(list(pm25_by_city.index))},
        datasets:[{{
            label:'PM2.5 (µg/m³)',
            data:{json.dumps(list(pm25_by_city.values.tolist()))},
            backgroundColor:'rgba(255,0,68,0.7)',borderColor:'#ff0044',
            borderWidth:2,borderRadius:8
        }}]
    }},
    options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}}}},
        scales:{{
            y:{{beginAtZero:true,grid:{{color:'rgba(0,255,136,0.1)'}},ticks:{{color:'#00ff88'}}}},
            x:{{grid:{{display:false}},ticks:{{color:'#00ff88'}}}}
        }}}}
}});

// Radar
new Chart('radarChart', {{
    type:'radar',
    data:{{
        labels:['Temperature','Humidity','AQI','PM2.5','Wind'],
        datasets:{json.dumps(radar_datasets)}
    }},
    options:{{
        responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{labels:{{color:'#00ff88',padding:12}}}}}},
        scales:{{r:{{
            beginAtZero:true,max:100,
            ticks:{{color:'#00ff88',backdropColor:'transparent'}},
            grid:{{color:'rgba(0,255,136,0.2)'}},
            pointLabels:{{color:'#00ff88',font:{{size:12}}}}
        }}}}
    }}
}});
</script>
</body>
</html>"""

    out_path = 'templates/dashboard.html'
    os.makedirs('templates', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Dashboard generated: {out_path}")
    return out_path


if __name__ == "__main__":
    import os
    Config.init_app()
    create_dashboard()
    print("🚀 Dashboard ready at /dashboard")